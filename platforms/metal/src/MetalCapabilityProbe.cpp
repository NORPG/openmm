/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * Portions copyright (c) 2026 Stanford University and the Authors.           *
 * -------------------------------------------------------------------------- */

#include "MetalCapabilityProbeInternal.h"
#include "MetalArray.h"
#include "MetalContext.h"
#include "MetalFixedPoint.h"
#include "MetalKernelSources.h"
#include "MetalProgram.h"
#include "openmm/OpenMMException.h"
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <vector>

using namespace OpenMM;
using namespace std;

namespace {

MetalFixedPoint64Storage split(uint64_t bits) {
    return {static_cast<uint32_t>(bits), static_cast<uint32_t>(bits >> 32)};
}

float reconstructReference(uint64_t bits) {
    const bool negative = (bits >> 63) != 0;
    const uint64_t magnitude = negative ? uint64_t(0)-bits : bits;
    // Independent CPU oracle: round the complete 64-bit magnitude directly to
    // float, then scale exactly.  No intermediate double or signed overflow.
    const float value = ldexp(static_cast<float>(magnitude), -32);
    return negative ? -value : value;
}

} // namespace

const char* MetalCapabilityProbeResult::getStatusName() const {
    switch (status) {
        case Status::NotRun: return "not-run";
        case Status::Supported: return "supported";
        case Status::InitializationFailed: return "initialization-failed";
        case Status::CompilationFailed: return "compilation-failed";
        case Status::PipelineCreationFailed: return "pipeline-creation-failed";
        case Status::ExecutionFailed: return "execution-failed";
        case Status::ValidationFailed: return "validation-failed";
    }
    return "unknown";
}

string OpenMM::detail::getSplitFixedPointProbeSource() {
    return MetalKernelSources::fixedPoint+"\n"+MetalKernelSources::capabilityProbe;
}

MetalCapabilityProbeResult OpenMM::detail::runSplitFixedPointProbe(
        MetalQueue& queue, const string& source, int blockSize) {
    MetalCapabilityProbeResult result;
    result.status = MetalCapabilityProbeResult::Status::CompilationFailed;
    try {
        // Always compile on the selected runtime, even in an offline-metallib
        // build.  An offline compiler or device-family flag is not this probe.
        MetalProgramOptions options;
        options.languageVersionMajor = 3;
        options.languageVersionMinor = 0;
        options.fastMathEnabled = true;
        MetalProgram program(queue, source, options);

        result.status = MetalCapabilityProbeResult::Status::PipelineCreationFailed;
        ComputeKernel writers = program.createKernel("probeSplitFixedPointWriters");
        ComputeKernel readers = program.createKernel("probeSplitFixedPointReaders");

        result.status = MetalCapabilityProbeResult::Status::InitializationFailed;
        const uint32_t numAtoms = 5, paddedNumAtoms = 8, numWriters = 2053;
        const uint32_t dispatchCount = 2;
        const size_t valueCount = 3*paddedNumAtoms;
        struct Contribution {
            float value;
            uint64_t bits;
        };
        const float unit = ldexp(1.0f, -32);
        const vector<Contribution> cases = {
            {unit, UINT64_C(1)}, {-unit, UINT64_MAX},
            {1.25f, UINT64_C(0x0000000140000000)},
            {-1.5f, UINT64_C(0xfffffffe80000000)},
            {1073741824.0f, UINT64_C(0x4000000000000000)},
            {-1073741824.0f, UINT64_C(0xc000000000000000)},
            {1.5f*unit, UINT64_C(1)}, {-1.5f*unit, UINT64_MAX},
            {-2147483648.0f, UINT64_C(0x8000000000000000)},
            {0.0f, UINT64_C(0)}, {0.75f*unit, UINT64_C(0)},
            {-0.75f*unit, UINT64_C(0)}
        };
        const uint64_t seeds[] = {UINT64_MAX, UINT64_C(0), UINT64_C(1),
                                 UINT64_C(0x7fffffffffffffff), UINT64_C(0x8000000000000000)};
        vector<uint64_t> expected(valueCount, UINT64_C(0x123456789abcdef0));
        vector<MetalFixedPoint64Storage> initial(valueCount);
        for (uint32_t axis = 0; axis < 3; axis++) {
            for (uint32_t atom = 0; atom < paddedNumAtoms; atom++) {
                const size_t index = atom+axis*paddedNumAtoms;
                if (atom < numAtoms)
                    expected[index] = seeds[(atom+axis)%numAtoms];
                initial[index] = split(expected[index]);
            }
        }
        vector<float> contributions(3*numWriters);
        for (uint32_t writer = 0; writer < numWriters; writer++) {
            for (uint32_t axis = 0; axis < 3; axis++) {
                const Contribution& contribution = cases[(writer+3*axis)%cases.size()];
                contributions[3*writer+axis] = contribution.value;
                // Unsigned host arithmetic is the modulo-2^64 accumulation
                // oracle, including carry, high-word wrap, and negative values.
                expected[writer%numAtoms+axis*paddedNumAtoms] += dispatchCount*contribution.bits;
            }
        }
        MetalArray words(queue, valueCount, sizeof(MetalFixedPoint64Storage), "split probe words");
        MetalArray input(queue, contributions.size(), sizeof(float), "split probe contributions");
        MetalArray raw(queue, valueCount, sizeof(MetalFixedPoint64Storage), "split probe raw reads");
        MetalArray reconstructed(queue, paddedNumAtoms, sizeof(MetalFloat4), "split probe reconstruction");
        const vector<MetalFloat4> poison(paddedNumAtoms,
                {numeric_limits<float>::quiet_NaN(), numeric_limits<float>::quiet_NaN(),
                 numeric_limits<float>::quiet_NaN(), numeric_limits<float>::quiet_NaN()});
        words.upload(initial.data(), false);
        input.upload(contributions.data(), false);
        raw.clear();
        reconstructed.upload(poison.data(), false);

        result.status = MetalCapabilityProbeResult::Status::ExecutionFailed;
        writers->addArg(words);
        writers->addArg(input);
        writers->addArg(numAtoms);
        writers->addArg(paddedNumAtoms);
        writers->addArg(numWriters);
        readers->addArg(words);
        readers->addArg(raw);
        readers->addArg(reconstructed);
        readers->addArg(paddedNumAtoms);
        for (uint32_t pass = 0; pass < dispatchCount; pass++)
            writers->execute(numWriters, blockSize);
        // No CPU wait between the writer dispatches and the GPU consumer.
        readers->execute(paddedNumAtoms);
        queue.waitUntilIdle(); // Also surface errors from every async writer.

        vector<MetalFixedPoint64Storage> actualRaw, actualWords;
        vector<MetalFloat4> actualReconstructed;
        raw.download(actualRaw);
        words.download(actualWords);
        reconstructed.download(actualReconstructed);

        result.status = MetalCapabilityProbeResult::Status::ValidationFailed;
        for (uint32_t atom = 0; atom < paddedNumAtoms; atom++) {
            const MetalFloat4& value = actualReconstructed[atom];
            const float components[] = {value.x, value.y, value.z};
            for (uint32_t axis = 0; axis < 3; axis++) {
                const size_t index = atom+axis*paddedNumAtoms;
                const MetalFixedPoint64Storage expectedWords = split(expected[index]);
                if (actualRaw[index].lo != expectedWords.lo || actualRaw[index].hi != expectedWords.hi ||
                        actualWords[index].lo != expectedWords.lo || actualWords[index].hi != expectedWords.hi ||
                        components[axis] != reconstructReference(expected[index])) {
                    stringstream message;
                    message << "Split Q32.32 result mismatch at atom " << atom << ", axis " << axis;
                    throw OpenMMException(message.str());
                }
            }
            if (value.w != 0.0f)
                throw OpenMMException("Split Q32.32 reconstruction metadata mismatch");
        }
        result.status = MetalCapabilityProbeResult::Status::Supported;
        result.diagnostic = "MSL 3.0 runtime compile, pipelines, two contending writer dispatches, "
                "and ordered GPU reconstruction validated using 32-bit atomics; not native 64-bit atomic support";
    }
    catch (const exception& e) {
        result.diagnostic = string(result.getStatusName())+": "+e.what();
        // Drain only this isolated queue.  Never discard errors from a real
        // simulation queue or allow async work to escape the probe lifetime.
        try {
            queue.waitUntilIdle();
        }
        catch (const exception& cleanupError) {
            result.diagnostic += string("; probe queue completion: ")+cleanupError.what();
        }
    }
    return result;
}
