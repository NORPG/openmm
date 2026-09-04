#include "MetalContext.h"
#include "MetalFixedPoint.h"
#include "MetalPlatform.h"
#include "openmm/System.h"
#include "openmm/common/ComputeArray.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/internal/ThreadPool.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

using namespace OpenMM;
using namespace std;

static_assert(is_base_of<ComputeContext, MetalContext>::value,
              "MetalContext must implement ComputeContext");

namespace {

void testFixedPointHostABI() {
    ASSERT_EQUAL(8, static_cast<int>(sizeof(MetalFixedPoint64Storage)));
    ASSERT_EQUAL(8, static_cast<int>(alignof(MetalFixedPoint64Storage)));
    ASSERT_EQUAL(0, static_cast<int>(offsetof(MetalFixedPoint64Storage, lo)));
    ASSERT_EQUAL(4, static_cast<int>(offsetof(MetalFixedPoint64Storage, hi)));
}

float reconstructFixedPointOnHost(const MetalFixedPoint64Storage& value) {
    const uint64_t bits = (static_cast<uint64_t>(value.hi) << 32) | value.lo;
    const bool negative = (value.hi & 0x80000000u) != 0u;
    const uint64_t magnitude = negative ? (~bits)+1u : bits;
    const float converted = static_cast<float>(magnitude)*ldexp(1.0f, -32);
    return negative ? -converted : converted;
}

void testFixedPointHelpers(MetalContext& context) {
    struct ConversionCase {
        float input;
        uint32_t lo;
        uint32_t hi;
        float reconstructed;
    };
    const float unit = ldexp(1.0f, -32);
    const float predecessorOfOne = nextafter(1.0f, 0.0f);
    const float twoTo31 = ldexp(1.0f, 31);
    const float largestValidFloat = nextafter(twoTo31, 0.0f);
    const vector<ConversionCase> cases = {
        {0.0f, 0x00000000u, 0x00000000u, 0.0f},
        {unit, 0x00000001u, 0x00000000u, unit},
        {-unit, 0xffffffffu, 0xffffffffu, -unit},
        {0.5f, 0x80000000u, 0x00000000u, 0.5f},
        {-0.5f, 0x80000000u, 0xffffffffu, -0.5f},
        {predecessorOfOne, 0xffffff00u, 0x00000000u, predecessorOfOne},
        {-predecessorOfOne, 0x00000100u, 0xffffffffu, -predecessorOfOne},
        {1.0f, 0x00000000u, 0x00000001u, 1.0f},
        {-1.0f, 0x00000000u, 0xffffffffu, -1.0f},
        {1.5f, 0x80000000u, 0x00000001u, 1.5f},
        {-1.5f, 0x80000000u, 0xfffffffeu, -1.5f},
        {123.75f, 0xc0000000u, 0x0000007bu, 123.75f},
        {-123.75f, 0x40000000u, 0xffffff84u, -123.75f},
        {largestValidFloat, 0x00000000u, 0x7fffff80u, largestValidFloat},
        {-twoTo31, 0x00000000u, 0x80000000u, -twoTo31},
        {0.75f*unit, 0x00000000u, 0x00000000u, 0.0f},
        {-0.75f*unit, 0x00000000u, 0x00000000u, 0.0f},
        {1.5f*unit, 0x00000001u, 0x00000000u, unit},
        {-1.5f*unit, 0xffffffffu, 0xffffffffu, -unit}
    };

    const string source = R"MSL(
kernel void convertFixedPointHelpers(device const float* input [[buffer(0)]],
                                     device uint2* converted [[buffer(1)]],
                                     device uint2* split [[buffer(2)]],
                                     device float* reconstructed [[buffer(3)]],
                                     constant uint& count [[buffer(4)]],
                                     uint index [[thread_position_in_grid]]) {
    if (index >= count)
        return;
    const float value = input[index];
    const float integral = trunc(value);
    const float scaledFraction = (value-integral)*0x1.0p32f;
    const uint fractionalMagnitude = uint(fabs(scaledFraction));
    converted[index] = realToFixedPoint(value);
    split[index] = splitFixedPoint(int(integral), fractionalMagnitude,
                                   scaledFraction <= -1.0f);
    reconstructed[index] = reconstructSignedFixedPoint(converted[index]);
}

kernel void loadFixedPointHelpers(device const uint2* input [[buffer(0)]],
                                  device uint2* loaded [[buffer(1)]],
                                  device float* reconstructed [[buffer(2)]],
                                  constant uint& count [[buffer(3)]],
                                  uint index [[thread_position_in_grid]]) {
    if (index >= count)
        return;
    loaded[index] = loadFixedPoint(input, index);
    reconstructed[index] = loadSignedFixedPoint(input, index);
}

kernel void loadFixedPointPlanes(device const uint2* input [[buffer(0)]],
                                 device float4* output [[buffer(1)]],
                                 constant uint& atom [[buffer(2)]],
                                 constant uint& paddedNumAtoms [[buffer(3)]],
                                 uint index [[thread_position_in_grid]]) {
    if (index == 0u)
        output[0] = float4(loadFixedPoint3(input, atom, paddedNumAtoms), 0.0f);
}
)MSL";
    ComputeProgram program = context.compileProgram(source);

    vector<float> inputValues;
    for (const auto& test : cases)
        inputValues.push_back(test.input);
    ComputeArray input;
    ComputeArray converted;
    ComputeArray split;
    ComputeArray reconstructed;
    input.initialize<float>(context, cases.size(), "fixed-point conversion input");
    converted.initialize<MetalFixedPoint64Storage>(context, cases.size(),
                                                    "converted fixed-point words");
    split.initialize<MetalFixedPoint64Storage>(context, cases.size(),
                                               "split fixed-point words");
    reconstructed.initialize<float>(context, cases.size(),
                                    "reconstructed fixed-point values");
    input.upload(inputValues);

    ComputeKernel convert = program->createKernel("convertFixedPointHelpers");
    const uint32_t conversionCount = static_cast<uint32_t>(cases.size());
    convert->addArg(input);
    convert->addArg(converted);
    convert->addArg(split);
    convert->addArg(reconstructed);
    convert->addArg(conversionCount);
    convert->execute(conversionCount);

    vector<MetalFixedPoint64Storage> convertedWords;
    vector<MetalFixedPoint64Storage> splitWords;
    vector<float> reconstructedValues;
    converted.download(convertedWords);
    split.download(splitWords);
    reconstructed.download(reconstructedValues);
    for (size_t i = 0; i < cases.size(); i++) {
        ASSERT_EQUAL(cases[i].lo, convertedWords[i].lo);
        ASSERT_EQUAL(cases[i].hi, convertedWords[i].hi);
        ASSERT_EQUAL(cases[i].lo, splitWords[i].lo);
        ASSERT_EQUAL(cases[i].hi, splitWords[i].hi);
        ASSERT_EQUAL(cases[i].reconstructed, reconstructedValues[i]);
    }

    vector<MetalFixedPoint64Storage> rawValues = {
        {0x00000000u, 0x00000000u},
        {0x00000001u, 0x00000000u},
        {0xffffffffu, 0xffffffffu},
        {0xffffffffu, 0x00000000u},
        {0x00000001u, 0xffffffffu},
        {0x40000000u, 0x00000001u},
        {0x80000000u, 0xfffffffeu},
        {0x00000000u, 0x80000000u},
        {0x2cb49649u, 0x01cfca81u},
        {0xd34b69b7u, 0xfe30357eu},
        // Exact half ties with even and odd retained significands.
        {0x01000001u, 0x00000000u},
        {0x01000003u, 0x00000000u},
        // Rounding carries out of the 24-bit significand.
        {0x01ffffffu, 0x00000000u},
        // The discarded portion begins exactly at the low/high word boundary.
        {0x80000000u, 0x00800000u},
        {0x80000000u, 0x00800001u}
    };
    uint32_t randomState = 0x12345678u;
    for (int i = 0; i < 2048; i++) {
        randomState = 1664525u*randomState+1013904223u;
        const uint32_t lo = randomState;
        randomState = 1664525u*randomState+1013904223u;
        rawValues.push_back({lo, randomState});
    }
    vector<float> expectedRawValues;
    for (const auto& value : rawValues)
        expectedRawValues.push_back(reconstructFixedPointOnHost(value));

    ComputeArray rawInput;
    ComputeArray rawLoaded;
    ComputeArray rawReconstructed;
    rawInput.initialize<MetalFixedPoint64Storage>(context, rawValues.size(),
                                                  "raw fixed-point input");
    rawLoaded.initialize<MetalFixedPoint64Storage>(context, rawValues.size(),
                                                   "loaded fixed-point words");
    rawReconstructed.initialize<float>(context, rawValues.size(),
                                       "loaded fixed-point values");
    rawInput.upload(rawValues);
    ComputeKernel load = program->createKernel("loadFixedPointHelpers");
    const uint32_t rawCount = static_cast<uint32_t>(rawValues.size());
    load->addArg(rawInput);
    load->addArg(rawLoaded);
    load->addArg(rawReconstructed);
    load->addArg(rawCount);
    load->execute(rawCount);

    vector<MetalFixedPoint64Storage> loadedWords;
    vector<float> loadedValues;
    rawLoaded.download(loadedWords);
    rawReconstructed.download(loadedValues);
    for (size_t i = 0; i < rawValues.size(); i++) {
        ASSERT_EQUAL(rawValues[i].lo, loadedWords[i].lo);
        ASSERT_EQUAL(rawValues[i].hi, loadedWords[i].hi);
        ASSERT_EQUAL(expectedRawValues[i], loadedValues[i]);
    }
    ASSERT_EQUAL(30395010.0f, loadedValues[8]);
    ASSERT_EQUAL(-30395010.0f, loadedValues[9]);
    ASSERT_EQUAL(ldexp(1.0f, -8), loadedValues[10]);
    ASSERT_EQUAL(ldexp(8388610.0f, -31), loadedValues[11]);
    ASSERT_EQUAL(ldexp(1.0f, -7), loadedValues[12]);
    ASSERT_EQUAL(8388608.0f, loadedValues[13]);
    ASSERT_EQUAL(8388610.0f, loadedValues[14]);

    ArrayInterface& longForces = context.getLongForceBuffer();
    const uint32_t paddedNumAtoms = static_cast<uint32_t>(context.getPaddedNumAtoms());
    const uint32_t atom = 17;
    vector<MetalFixedPoint64Storage> planeValues(longForces.getSize(), {0u, 0u});
    planeValues[atom] = {0x40000000u, 0x00000001u};
    planeValues[atom+paddedNumAtoms] = {0x80000000u, 0xfffffffeu};
    planeValues[atom+2*paddedNumAtoms] = {0x00000001u, 0x00000000u};
    for (uint32_t axis = 0; axis < 3; axis++)
        planeValues[3*atom+axis] = {0x00000000u, 0x0000007fu+axis};
    longForces.upload(planeValues);

    ComputeArray planeOutput;
    planeOutput.initialize<MetalFloat4>(context, 1, "loaded fixed-point planes");
    ComputeKernel loadPlanes = program->createKernel("loadFixedPointPlanes");
    loadPlanes->addArg(longForces);
    loadPlanes->addArg(planeOutput);
    loadPlanes->addArg(atom);
    loadPlanes->addArg(paddedNumAtoms);
    loadPlanes->execute(1);
    vector<MetalFloat4> loadedPlanes;
    planeOutput.download(loadedPlanes);
    ASSERT_EQUAL(1.25f, loadedPlanes[0].x);
    ASSERT_EQUAL(-1.5f, loadedPlanes[0].y);
    ASSERT_EQUAL(unit, loadedPlanes[0].z);
    ASSERT_EQUAL(0.0f, loadedPlanes[0].w);
}

void testLongForceBuffer() {
    System system;
    for (int i = 0; i < ComputeContext::TileSize+1; i++)
        system.addParticle(1.0);
    MetalContext context(system, NULL, 0);
    ASSERT_EQUAL(2*ComputeContext::TileSize, context.getPaddedNumAtoms());

    ArrayInterface& buffer = context.getLongForceBuffer();
    const size_t paddedNumAtoms = static_cast<size_t>(context.getPaddedNumAtoms());
    const size_t elementCount = 3*paddedNumAtoms;
    ASSERT(buffer.isInitialized());
    ASSERT(dynamic_cast<MetalArray*>(&buffer) != NULL);
    ASSERT(&buffer == &context.getLongForceBuffer());
    ASSERT(&buffer != &context.getFloatForceBuffer());
    ASSERT(&buffer != &context.getForceBuffers());
    ASSERT_EQUAL(elementCount, buffer.getSize());
    ASSERT_EQUAL(static_cast<int>(sizeof(MetalFixedPoint64Storage)), buffer.getElementSize());
    ASSERT_EQUAL(3*paddedNumAtoms*sizeof(MetalFixedPoint64Storage),
                 buffer.getSize()*static_cast<size_t>(buffer.getElementSize()));
    ASSERT(&buffer.getContext() == &context);

    vector<MetalFixedPoint64Storage> values;
    buffer.download(values);
    for (const auto& value : values) {
        ASSERT_EQUAL(0u, value.lo);
        ASSERT_EQUAL(0u, value.hi);
    }

    vector<MetalFixedPoint64Storage> sentinels(elementCount);
    for (size_t i = 0; i < sentinels.size(); i++) {
        const uint32_t index = static_cast<uint32_t>(i);
        sentinels[i] = {0x10203040u+0x01010101u*index,
                        0x89abcdefu^(0x9e3779b9u*(index+1u))};
    }
    buffer.upload(sentinels);
    buffer.download(values);
    for (size_t i = 0; i < values.size(); i++) {
        ASSERT_EQUAL(sentinels[i].lo, values[i].lo);
        ASSERT_EQUAL(sentinels[i].hi, values[i].hi);
    }

    // Direct clear uses an ordered GPU blit over both 32-bit words of every
    // logical 64-bit element.
    context.clearBuffer(buffer);
    buffer.download(values);
    for (const auto& value : values) {
        ASSERT_EQUAL(0u, value.lo);
        ASSERT_EQUAL(0u, value.hi);
    }

    // Exercise the exact GPU-copy save/restore sequence used by Common
    // kernels such as the Monte Carlo barostat rollback path.
    ComputeArray savedLongForces;
    savedLongForces.initialize<MetalFixedPoint64Storage>(context, elementCount,
                                                         "saved long force buffer");
    buffer.upload(sentinels);
    buffer.copyTo(savedLongForces);
    context.clearBuffer(buffer);
    savedLongForces.copyTo(buffer);
    buffer.download(values);
    for (size_t i = 0; i < values.size(); i++) {
        ASSERT_EQUAL(sentinels[i].lo, values[i].lo);
        ASSERT_EQUAL(sentinels[i].hi, values[i].hi);
    }

    context.clearAutoclearBuffers();
    buffer.download(values);
    for (const auto& value : values) {
        ASSERT_EQUAL(0u, value.lo);
        ASSERT_EQUAL(0u, value.hi);
    }

    // Version 2 checkpoints preserve the complete padded long force buffer.
    buffer.upload(sentinels);
    stringstream checkpoint(ios_base::in | ios_base::out | ios_base::binary);
    context.createCheckpoint(checkpoint);
    context.clearBuffer(buffer);
    checkpoint.seekg(0);
    context.loadCheckpoint(checkpoint);
    buffer.download(values);
    for (size_t i = 0; i < values.size(); i++) {
        ASSERT_EQUAL(sentinels[i].lo, values[i].lo);
        ASSERT_EQUAL(sentinels[i].hi, values[i].hi);
    }

    // New readers remain compatible with version 1 checkpoints, which have
    // no long force payload.  Loading one must clear pre-existing scratch data.
    string legacyCheckpointData = checkpoint.str();
    const size_t longForceBytes = elementCount*sizeof(MetalFixedPoint64Storage);
    ASSERT(legacyCheckpointData.size() >= 2*sizeof(uint32_t)+longForceBytes);
    const uint32_t legacyVersion = 1;
    memcpy(&legacyCheckpointData[sizeof(uint32_t)], &legacyVersion, sizeof(legacyVersion));
    legacyCheckpointData.resize(legacyCheckpointData.size()-longForceBytes);
    buffer.upload(sentinels);
    stringstream legacyCheckpoint(legacyCheckpointData,
                                  ios_base::in | ios_base::out | ios_base::binary);
    context.loadCheckpoint(legacyCheckpoint);
    buffer.download(values);
    for (const auto& value : values) {
        ASSERT_EQUAL(0u, value.lo);
        ASSERT_EQUAL(0u, value.hi);
    }
    testFixedPointHelpers(context);
}

void testCoreContextSurface() {
    System system;
    system.addParticle(2.0);
    system.addParticle(0.0);
    MetalContext context(system, NULL, 0);

    ASSERT_EQUAL(1, context.getNumContexts());
    ASSERT_EQUAL(0, context.getContextIndex());
    ASSERT_EQUAL(2, context.getNumParticles());
    ASSERT_EQUAL(1, context.getNumAtomBlocks());
    ASSERT_EQUAL(ComputeContext::TileSize, context.getPosq().getSize());
    ASSERT_EQUAL(ComputeContext::TileSize, context.getVelm().getSize());
    ASSERT_EQUAL(ComputeContext::TileSize, context.getFloatForceBuffer().getSize());
    ASSERT(context.getContextImpl() == NULL);
    ASSERT(context.getAllContexts().size() == 1);
    ASSERT(context.getAllContexts()[0] == &context);
    ASSERT(!context.getIsCPU());
    ASSERT_EQUAL(32, context.getSIMDWidth());
    ASSERT(!context.getSupports64BitGlobalAtomics());
    ASSERT(!context.getSupportsDoublePrecision());
    ASSERT(!context.getUseDoublePrecision());
    ASSERT(!context.getUseMixedPrecision());
    ASSERT(context.getPinnedBuffer() != NULL);
    ASSERT_EQUAL(1, context.getThreadPool().getNumThreads());
    ASSERT(context.getMaxThreadBlockSize() > 0);
    ASSERT(context.computeThreadBlockSize(0.0) > 0);

    context.setCharges({0.25, -0.5});
    context.setPositions({Vec3(1.0, 2.0, 3.0), Vec3(4.0, 5.0, 6.0)});
    context.setVelocities({Vec3(0.1, 0.2, 0.3), Vec3(0.4, 0.5, 0.6)});
    vector<MetalFloat4> posq;
    vector<MetalFloat4> velm;
    context.getPosq().download(posq);
    context.getVelm().download(velm);
    ASSERT_EQUAL_TOL(0.25, posq[0].w, 1e-6);
    ASSERT_EQUAL_TOL(-0.5, posq[1].w, 1e-6);
    ASSERT_EQUAL_TOL(0.5, velm[0].w, 1e-6);
    ASSERT_EQUAL_TOL(0.0, velm[1].w, 1e-6);
    ASSERT(context.requestPosqCharges());
    ASSERT(!context.requestPosqCharges());

    ComputeArray values;
    ComputeArray copied;
    values.initialize<float>(context, 4, "ComputeContext values");
    copied.initialize<float>(context, 4, "ComputeContext copied values");
    vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
    values.upload(input);
    values.copyTo(copied);
    vector<float> output;
    copied.download(output);
    for (int i = 0; i < 4; i++)
        ASSERT_EQUAL_TOL(input[i], output[i], 1e-6);

    context.clearBuffer(copied);
    copied.download(output);
    for (float value : output)
        ASSERT_EQUAL_TOL(0.0, value, 1e-6);
    context.addAutoclearBuffer(values);
    context.clearAutoclearBuffers();
    values.download(output);
    for (float value : output)
        ASSERT_EQUAL_TOL(0.0, value, 1e-6);

    // macOS blit fills require four-byte range boundaries.  Verify that the
    // staged fallback preserves clearBuffer() for arbitrary logical sizes.
    ComputeArray bytes;
    bytes.initialize<uint8_t>(context, 7, "odd-sized clear buffer");
    vector<uint8_t> byteInput(7, 0xa5u);
    vector<uint8_t> byteOutput;
    bytes.upload(byteInput);
    context.clearBuffer(bytes);
    bytes.download(byteOutput);
    for (uint8_t value : byteOutput)
        ASSERT_EQUAL(0u, static_cast<unsigned int>(value));

    const string source = R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void addValue(device float* values [[buffer(0)]],
                     constant uint& count [[buffer(1)]],
                     uint index [[thread_position_in_grid]]) {
    if (index < count)
        values[index] += TEST_INCREMENT;
}
)MSL";
    values.upload(input);
    ComputeProgram program = context.compileProgram(source, {
        {"TEST_INCREMENT", "1.5f"},
        {"magnitude", "1"}
    });
    ComputeKernel kernel = program->createKernel("addValue");
    unsigned int count = 4;
    kernel->addArg(values);
    kernel->addArg(count);
    kernel->execute(count);
    ComputeEvent event = context.createEvent();
    event->enqueue();
    event->wait();
    values.download(output);
    for (int i = 0; i < 4; i++)
        ASSERT_EQUAL_TOL(input[i]+1.5, output[i], 1e-6);

    ComputeQueue sibling = context.createQueue();
    ASSERT(sibling.get() != NULL);
    ASSERT(dynamic_cast<MetalQueue*>(sibling.get()) != NULL);
    context.flushQueue();

    context.addEnergyParameterDerivative("lambda");
    context.addEnergyParameterDerivative("lambda");
    ASSERT_EQUAL(1, context.getEnergyParamDerivNames().size());
    context.getEnergyParamDerivWorkspace()["lambda"] = 2.0;
    ASSERT_EQUAL_TOL(2.0, context.getEnergyParamDerivWorkspace()["lambda"], 1e-6);
}

} // namespace

int main() {
    try {
        testFixedPointHostABI();
        if (!MetalPlatform::isPlatformSupported()) {
            cout << "Test skipped: no supported Metal device is visible" << endl;
            return 0;
        }
        testLongForceBuffer();
        testCoreContextSurface();
    }
    catch (const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
