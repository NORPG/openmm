#include "MetalContext.h"
#include "MetalPlatform.h"
#include "openmm/System.h"
#include "openmm/common/ComputeArray.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/internal/ThreadPool.h"

#include <cstdint>
#include <exception>
#include <iostream>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

using namespace OpenMM;
using namespace std;

static_assert(is_base_of<ComputeContext, MetalContext>::value,
              "MetalContext must implement ComputeContext");

namespace {

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
    ComputeProgram program = context.compileProgram(source, {{"TEST_INCREMENT", "1.5f"}});
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

    bool rejected = false;
    try {
        context.getLongForceBuffer();
    }
    catch (const OpenMMException&) {
        rejected = true;
    }
    ASSERT(rejected);
}

} // namespace

int main() {
    try {
        if (!MetalPlatform::isPlatformSupported()) {
            cout << "Test skipped: no supported Metal device is visible" << endl;
            return 0;
        }
        testCoreContextSurface();
    }
    catch (const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
