/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 *                                                                            *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Ported from the OpenMM CUDA and OpenCL Platforms.                          *
 * Sources: platforms/cuda/src/CudaContext.cpp                                *
 *          platforms/opencl/src/OpenCLContext.cpp                            *
 *                                                                            *
 * Original CUDA/OpenCL Platform code:                                       *
 * Portions copyright (c) 2009-2026 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 *                                                                            *
 * Metal Platform code:                                                       *
 * Portions copyright (c) 2026 Chun-Chi Hung.                                 *
 * Authors: Chun-Chi Hung                                                     *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the               *
 * GNU Lesser General Public License for more details.                        *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public License   *
 * along with this program. If not, see <http://www.gnu.org/licenses/>.       *
 * -------------------------------------------------------------------------- */

#include "MetalContext.h"
#include "MetalEvent.h"
#include "MetalFFT3D.h"
#include "MetalIntegrationUtilities.h"
#include "MetalKernel.h"
#include "MetalKernelSources.h"
#include "MetalLanguagePolicy.h"
#include "MetalNonbondedUtilities.h"
#include "MetalProgram.h"
#include "MetalQueue.h"
#include "MetalSort.h"
#include "MetalSourceAdapter.h"
#include "openmm/MonteCarloFlexibleBarostat.h"
#include "openmm/common/BondedUtilities.h"
#include "openmm/common/ExpressionUtilities.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/ThreadPool.h"
#import <Metal/Metal.h>
#include <IOKit/IOKitLib.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <numeric>
#include <sstream>

using namespace OpenMM;
using namespace std;

/** Read complete source-owned policy declarations without inspecting shader bodies. */
static bool hasSourcePolicy(const string& source, const string& policy) {
    istringstream lines(source);
    string line;
    while (getline(lines, line)) {
        istringstream declaration(line);
        string comment, marker, value, extra;
        if (declaration >> comment >> marker >> value && comment == "//" &&
                marker == "OPENMM_METAL_SOURCE_POLICY" && value == policy &&
                !(declaration >> extra))
            return true;
    }
    return false;
}

/**
 * @brief Match the selected Metal device to its public I/O Registry entry.
 *
 * Metal has no core-count property. gpu-core-count is best-effort driver data,
 * not a Metal API contract: missing, malformed, or implausible values retain
 * the previous launch budget. Never infer a count from a product name.
 */
static int getComputeUnits(id<MTLDevice> device) {
    io_service_t service = IOServiceGetMatchingService(kIOMainPortDefault,
            IORegistryEntryIDMatching(device.registryID));
    if (service == IO_OBJECT_NULL)
        return 0;
    CFTypeRef property = IORegistryEntryCreateCFProperty(service, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
    IOObjectRelease(service);
    int count = 0;
    if (property != nullptr) {
        if (CFGetTypeID(property) == CFNumberGetTypeID() &&
                !CFNumberGetValue((CFNumberRef) property, kCFNumberIntType, &count))
            count = 0;
        CFRelease(property);
    }
    return count > 0 && count <= 1024 ? count : 0;
}

struct MetalContext::Impl {
    id<MTLDevice> device = nil;
    id<MTLBuffer> pinnedBuffer = nil;
    int languageVersion = 300;
    ThreadPool threads;
    Impl() : threads(1) {
    }
};

/** One mode switch covers every linked Context that can share accumulator arrays. */
struct MetalContext::AccumulatorState {
    bool floating;
    vector<MetalContext*> contexts;
    explicit AccumulatorState(bool floating) : floating(floating) {
    }
};

MetalContext::MetalContext(const System& system, ContextImpl* simulation, MetalContext* linked, bool floatingAccumulators) : ComputeContext(system),
        simulation(simulation), accumulatorState(linked == nullptr ?
                make_shared<AccumulatorState>(floatingAccumulators) : linked->accumulatorState),
        initialized(false), numComputeUnits(0), energyWorkspace(0.0) {
    try {
        impl.reset(new Impl());
        @autoreleasepool {
            impl->device = (linked == nullptr ? MTLCreateSystemDefaultDevice() : (__bridge id<MTLDevice>) linked->getDevice());
            if (impl->device == nil)
                throw OpenMMException("No Metal device is available");
            if (![impl->device supportsFamily:MTLGPUFamilyApple7])
                throw OpenMMException("The Metal Platform requires Apple silicon");
            numComputeUnits = linked == nullptr ? getComputeUnits(impl->device) : linked->getNumComputeUnits();
            // Inner CustomCV/ATM contexts share the parent's stream, just as in
            // OpenCL, so state copies and force evaluation stay device-ordered.
            defaultQueue = (linked == nullptr ? createQueue() : linked->getCurrentQueue());
            restoreDefaultQueue();
            numAtoms = system.getNumParticles();
            if (numAtoms > numeric_limits<int>::max()-TileSize)
                throw OpenMMException("Too many atoms for the Metal Platform");
            paddedNumAtoms = max(TileSize, TileSize*((numAtoms+TileSize-1)/TileSize));
            useDoublePrecision = false;
            useMixedPrecision = false;
            numAtomBlocks = paddedNumAtoms/TileSize;
            numThreadBlocks = numComputeUnits == 0 ? 128 : 12*numComputeUnits;
            fixedPointRange.initialize<uint32_t>(*this, 2, "fixedPointRange");
            fixedPointRange.upload(vector<uint32_t>{0, 0});
            system.getDefaultPeriodicBoxVectors(periodicBoxVectors[0], periodicBoxVectors[1], periodicBoxVectors[2]);
            boxIsTriclinic = (periodicBoxVectors[0][1] != 0.0 || periodicBoxVectors[0][2] != 0.0 ||
                    periodicBoxVectors[1][0] != 0.0 || periodicBoxVectors[1][2] != 0.0 ||
                    periodicBoxVectors[2][0] != 0.0 || periodicBoxVectors[2][1] != 0.0);
            for (int i = 0; i < system.getNumForces(); i++)
                if (dynamic_cast<const MonteCarloFlexibleBarostat*>(&system.getForce(i)) != nullptr)
                    boxIsTriclinic = true;
            compilationDefines["OPENMM_METAL"] = "1";
            compilationDefines["OPENMM_METAL_NATIVE_FLOAT_ATOMICS"] = to_string(OPENMM_METAL_NATIVE_FLOAT_ATOMICS);
            compilationDefines["OPENMM_METAL_FAST_MINIMIZE_SHUFFLE"] = to_string(OPENMM_METAL_FAST_MINIMIZE_SHUFFLE);
            compilationDefines["OPENMM_METAL_FAST_CONSTANT_POTENTIAL_REDUCTION"] = to_string(OPENMM_METAL_FAST_CONSTANT_POTENTIAL_REDUCTION);
            compilationDefines["OPENMM_METAL_FAST_CONSTANT_POTENTIAL_CG_REDUCTION"] = to_string(OPENMM_METAL_FAST_CONSTANT_POTENTIAL_CG_REDUCTION);
            compilationDefines["OPENMM_METAL_FAST_CONSTANT_POTENTIAL_MATRIX_REDUCTION"] = to_string(OPENMM_METAL_FAST_CONSTANT_POTENTIAL_MATRIX_REDUCTION);
            compilationDefines["OPENMM_METAL_FAST_CONSTANT_POTENTIAL_MATRIX_BROADCAST"] = to_string(OPENMM_METAL_FAST_CONSTANT_POTENTIAL_MATRIX_BROADCAST);
            compilationDefines["OPENMM_METAL_FAST_CUSTOM_NONBONDED_GROUPS_SHUFFLE"] = to_string(OPENMM_METAL_FAST_CUSTOM_NONBONDED_GROUPS_SHUFFLE);
            compilationDefines["OPENMM_METAL_FAST_CUSTOM_MANY_PARTICLE_BALLOT"] = to_string(OPENMM_METAL_FAST_CUSTOM_MANY_PARTICLE_BALLOT);
            compilationDefines["OPENMM_METAL_FAST_LCPO_BALLOT"] = to_string(OPENMM_METAL_FAST_LCPO_BALLOT);
            compilationDefines["OPENMM_METAL_FAST_FP16_BOUNDS_NEXTAFTER"] = to_string(OPENMM_METAL_FAST_FP16_BOUNDS_NEXTAFTER);
            compilationDefines["OPENMM_METAL_TUNE_FORCE_REQUIRED_THREADS"] = to_string(OPENMM_METAL_TUNE_FORCE_REQUIRED_THREADS);
            compilationDefines["USE_GBSA_CHAIN_RULE_GUARD"] = to_string(OPENMM_METAL_FAST_GBSA_CHAIN_RULE_GUARD);
            compilationDefines["USE_GBSA_RECIPROCAL_REUSE"] = to_string(OPENMM_METAL_FAST_GBSA_RECIPROCAL_REUSE);
            compilationDefines["OPENMM_METAL_FAST_SPARSE_FORCE_AGGREGATION"] = to_string(OPENMM_METAL_FAST_SPARSE_FORCE_AGGREGATION);
            // MSL 3.0 exposes only uint64 min/max, not uint64 add/CAS/load.
            // These operations additionally require Apple8 or newer on macOS.
            compilationDefines["OPENMM_METAL_HAS_UINT64_MIN_MAX"] =
                [impl->device supportsFamily:MTLGPUFamilyApple8] ? "1" : "0";
            compilationDefines["OPENMM_METAL_CHECK_FIXED_POINT_RANGE"] = to_string(!OPENMM_METAL_MINIMIZE_FLOAT_ACCUMULATORS);
            // Preserve the OpenCL periodic-image convention for shared kernels.
            if (getBoxIsTriclinic()) {
                compilationDefines["APPLY_PERIODIC_TO_DELTA(delta)"] =
                    "{"
                    "real scale3 = floor(delta.z*invPeriodicBoxSize.z+0.5f); \\\n"
                    "delta.xyz -= scale3*periodicBoxVecZ.xyz; \\\n"
                    "real scale2 = floor(delta.y*invPeriodicBoxSize.y+0.5f); \\\n"
                    "delta.xy -= scale2*periodicBoxVecY.xy; \\\n"
                    "real scale1 = floor(delta.x*invPeriodicBoxSize.x+0.5f); \\\n"
                    "delta.x -= scale1*periodicBoxVecX.x;}";
                compilationDefines["APPLY_PERIODIC_TO_POS(pos)"] =
                    "{"
                    "real scale3 = floor(pos.z*invPeriodicBoxSize.z); \\\n"
                    "pos.xyz -= scale3*periodicBoxVecZ.xyz; \\\n"
                    "real scale2 = floor(pos.y*invPeriodicBoxSize.y); \\\n"
                    "pos.xy -= scale2*periodicBoxVecY.xy; \\\n"
                    "real scale1 = floor(pos.x*invPeriodicBoxSize.x); \\\n"
                    "pos.x -= scale1*periodicBoxVecX.x;}";
                compilationDefines["APPLY_PERIODIC_TO_POS_WITH_CENTER(pos, center)"] =
                    "{"
                    "real scale3 = floor((pos.z-center.z)*invPeriodicBoxSize.z+0.5f); \\\n"
                    "pos.x -= scale3*periodicBoxVecZ.x; \\\n"
                    "pos.y -= scale3*periodicBoxVecZ.y; \\\n"
                    "pos.z -= scale3*periodicBoxVecZ.z; \\\n"
                    "real scale2 = floor((pos.y-center.y)*invPeriodicBoxSize.y+0.5f); \\\n"
                    "pos.x -= scale2*periodicBoxVecY.x; \\\n"
                    "pos.y -= scale2*periodicBoxVecY.y; \\\n"
                    "real scale1 = floor((pos.x-center.x)*invPeriodicBoxSize.x+0.5f); \\\n"
                    "pos.x -= scale1*periodicBoxVecX.x;}";
            }
            else {
                compilationDefines["APPLY_PERIODIC_TO_DELTA(delta)"] =
                    "delta.xyz -= floor(delta.xyz*invPeriodicBoxSize.xyz+0.5f)*periodicBoxSize.xyz;";
                compilationDefines["APPLY_PERIODIC_TO_POS(pos)"] =
                    "pos.xyz -= floor(pos.xyz*invPeriodicBoxSize.xyz)*periodicBoxSize.xyz;";
                compilationDefines["APPLY_PERIODIC_TO_POS_WITH_CENTER(pos, center)"] =
                    "{"
                    "pos.x -= floor((pos.x-center.x)*invPeriodicBoxSize.x+0.5f)*periodicBoxSize.x; \\\n"
                    "pos.y -= floor((pos.y-center.y)*invPeriodicBoxSize.y+0.5f)*periodicBoxSize.y; \\\n"
                    "pos.z -= floor((pos.z-center.z)*invPeriodicBoxSize.z+0.5f)*periodicBoxSize.z;}";
            }
        }
        initializeNativeMath();
        // Common owns the state arrays and charge/clear/reduction kernels. Keep
        // the eight-byte force capacity stable across scoped minimization.
        initializeKernels();
        energyBuffer.initialize<float>(*this, getNumThreadBlocks()*ThreadBlockSize, "energyBuffer");
        energySum.initialize<float>(*this, numComputeUnits == 0 ? getNumThreadBlocks() : numComputeUnits, "energySum");
        vector<mm_float4> velocities(paddedNumAtoms, mm_float4(0, 0, 0, 0));
        for (int i = 0; i < numAtoms; i++) {
            double mass = system.getParticleMass(i);
            velocities[i].w = (mass == 0.0 ? 0.0f : (float) (1.0/mass));
        }
        velm.upload(velocities);
        clearBuffer(posq);
        clearBuffer(longForceBuffer);
        clearBuffer(energyBuffer);
        addAutoclearBuffer(longForceBuffer);
        addAutoclearBuffer(energyBuffer);
        size_t pinnedBytes = max(longForceBuffer.getSize()*longForceBuffer.getElementSize(),
                energyBuffer.getSize()*energyBuffer.getElementSize());
        pinnedBytes = max(pinnedBytes, posq.getSize()*posq.getElementSize());
        resizePinnedBuffer(pinnedBytes);
        getCurrentMetalQueue().finish();
        accumulatorState->contexts.push_back(this);
    }
    catch (...) {
        delete workThread;
        workThread = nullptr;
        throw;
    }
}

void MetalContext::initializeNativeMath() {
    // Port OpenCL's determineNativeAccuracy probe. Two float4 records replace
    // its float8, which MSL does not expose. The probe always uses safe math.
    MetalArray data(*this, 40, sizeof(mm_float4), "nativeMathAccuracy");
    vector<mm_float4> values(40, mm_float4(0, 0, 0, 0));
    float nextValue = 1e-4f;
    for (int i = 0; i < 20; i++) {
        values[2*i].x = nextValue;
        nextValue *= (float) M_PI;
    }
    data.upload(values);
    ComputeProgram program = compileProgram(MetalKernelSources::nativeMath, {{"OPENMM_METAL_REQUIRE_SAFE_MATH", "1"}});
    ComputeKernel kernel = program->createKernel("determineNativeAccuracy");
    kernel->addArg(data);
    kernel->execute(20);
    data.download(values);
    double errors[5] = {};
    for (int i = 0; i < 20; i++) {
        double v = values[2*i].x;
        double references[] = {sqrt(v), 1.0/sqrt(v), 1.0/v, exp(v), log(v)};
        double actual[] = {values[2*i].y, values[2*i].z, values[2*i].w, values[2*i+1].x, values[2*i+1].y};
        for (int j = 0; j < 5; j++) {
            // OpenCL ignores overflowing exp() samples through a NaN error.
            // Explicitly exclude references outside FP32's range instead;
            // unexpected non-finite results within that range still fail.
            if (isfinite(references[j]) && fabs(references[j]) <= numeric_limits<float>::max())
                errors[j] = !isfinite(actual[j]) ? numeric_limits<double>::infinity() :
                    max(errors[j], fabs(actual[j]-references[j])/fabs(references[j]));
        }
    }
    const char* functions[] = {"SQRT", "RSQRT", "RECIP", "EXP", "LOG"};
    for (int i = 0; i < 5; i++)
        compilationDefines[string("OPENMM_METAL_USE_NATIVE_")+functions[i]] = errors[i] < 1e-6 ? "1" : "0";
}

MetalContext::~MetalContext() {
    auto& contexts = accumulatorState->contexts;
    contexts.erase(remove(contexts.begin(), contexts.end(), this), contexts.end());
    // Drain before destroying utility-owned buffers; destruction must not throw.
    try {
        if (currentQueue)
            getCurrentMetalQueue().finish();
        if (defaultQueue && defaultQueue != currentQueue)
            static_cast<MetalQueue&>(*defaultQueue).finish();
    }
    catch (...) {
    }
    delete workThread;
    for (auto* value : forces)
        delete value;
    for (auto* value : reorderListeners)
        delete value;
    for (auto* value : preComputations)
        delete value;
    for (auto* value : postComputations)
        delete value;
    // Queue destructors finish pending work without throwing from destruction.
    currentQueue.reset();
    defaultQueue.reset();
}

bool MetalContext::getUseFloatingPointAccumulators() const {
    return accumulatorState->floating;
}

void MetalContext::setCheckFixedPointRange(bool enabled) {
    // Uploads are queue-ordered. Keep the sticky bit through every force
    // evaluation in one solve, including constraint and linked-context kernels.
    for (auto* context : accumulatorState->contexts)
        context->fixedPointRange.upload(vector<uint32_t>{enabled ? 1u : 0u, 0u});
}

void MetalContext::checkFixedPointRange() {
    bool overflow = false;
    for (auto* context : accumulatorState->contexts) {
        vector<uint32_t> words;
        context->fixedPointRange.download(words);
        overflow |= words[1] != 0;
    }
    if (overflow)
        throw OpenMMException("Minimization exceeded the GPU Q32.32 force range; CPU fallback is disabled for this platform");
}

void MetalContext::restoreCheckFixedPointRange() noexcept {
    for (auto* context : accumulatorState->contexts) {
        try {
            context->fixedPointRange.upload(vector<uint32_t>{0, 0});
        }
        catch (...) {
        }
    }
}

void MetalContext::setUseFloatingPointAccumulators(bool enabled) {
    if (enabled == accumulatorState->floating)
        return;
    // Drain every queue even when one reports an error. No queued dispatch may
    // still interpret a shared buffer in the old format when the mode changes.
    exception_ptr error;
    for (auto* context : accumulatorState->contexts) {
        for (const auto& queue : {context->currentQueue, context->defaultQueue}) {
            try {
                if (queue)
                    static_cast<MetalQueue&>(*queue).finish();
            }
            catch (...) {
                if (!error)
                    error = current_exception();
            }
        }
    }
    if (error)
        rethrow_exception(error);
    accumulatorState->floating = enabled;
    for (auto* context : accumulatorState->contexts)
        context->setForcesValid(false);
    for (auto* context : accumulatorState->contexts) {
        if (context->simulation != nullptr)
            context->simulation->systemChanged();
    }
    for (auto* context : accumulatorState->contexts)
        context->clearAutoclearBuffers();
}

void MetalContext::restoreFloatingPointAccumulators(bool enabled) noexcept {
    try {
        setUseFloatingPointAccumulators(enabled);
    }
    catch (...) {
        // Preserve the initiating exception, but never leave the host selecting
        // the temporary ABI. A later force evaluation clears its accumulators.
        accumulatorState->floating = enabled;
        for (auto* context : accumulatorState->contexts) {
            context->setForcesValid(false);
            try {
                if (context->simulation != nullptr)
                    context->simulation->systemChanged();
            }
            catch (...) {
            }
        }
    }
}

void* MetalContext::getDevice() const {
    return (__bridge void*) impl->device;
}

int MetalContext::getMetalLanguageVersion() const {
    return impl->languageVersion;
}

ContextImpl* MetalContext::getContextImpl() {
    if (simulation == nullptr)
        throw OpenMMException("The Metal Platform is not attached to a simulation Context");
    return simulation;
}

ComputeQueue MetalContext::createQueue() {
    return ComputeQueue(new MetalQueue(getDevice()));
}

MetalQueue& MetalContext::getCurrentMetalQueue() {
    MetalQueue* queue = dynamic_cast<MetalQueue*>(currentQueue.get());
    if (queue == nullptr)
        throw OpenMMException("The current queue is not a MetalQueue");
    id<MTLCommandQueue> native = (__bridge id<MTLCommandQueue>) queue->getQueue();
    if (native.device != impl->device)
        throw OpenMMException("The current Metal queue belongs to a different device");
    return *queue;
}

MetalArray* MetalContext::createArray() {
    return new MetalArray();
}

MetalArray& MetalContext::unwrap(ArrayInterface& array) const {
    ComputeArray* wrapper = dynamic_cast<ComputeArray*>(&array);
    MetalArray* metalArray = dynamic_cast<MetalArray*>(wrapper == nullptr ? &array : &wrapper->getArray());
    if (metalArray == nullptr || dynamic_cast<MetalContext&>(metalArray->getContext()).getDevice() != getDevice())
        throw OpenMMException("Array argument is not a MetalArray from this device");
    if (dynamic_cast<MetalContext&>(metalArray->getContext()).getUseFloatingPointAccumulators() != getUseFloatingPointAccumulators())
        throw OpenMMException("Cannot share accumulator arrays between ordinary and minimization Metal contexts");
    return *metalArray;
}

ComputeEvent MetalContext::createEvent() {
    return ComputeEvent(new MetalEvent(*this));
}

void MetalContext::downloadFixedPointBuffer(ArrayInterface& array, vector<double>& values) {
    unwrap(array);
    if (!getUseFloatingPointAccumulators()) {
        ComputeContext::downloadFixedPointBuffer(array, values);
        return;
    }
    if (array.getElementSize() != sizeof(float) && array.getElementSize() != sizeof(int64_t))
        throw OpenMMException("Floating accumulator decoding requires four- or eight-byte elements");
    values.resize(array.getSize());
    if (values.empty())
        return;
    // Keep fixed-point allocation capacity while decoding only the logical
    // element count from the temporary packed-float representation.
    vector<float> storage(array.getSize()*array.getElementSize()/sizeof(float));
    array.download(storage.data());
    for (size_t i = 0; i < values.size(); i++)
        values[i] = storage[i];
}

ComputeProgram MetalContext::compileProgram(const string source, const map<string, string>& defines) {
    @autoreleasepool {
        const bool floatingAccumulators = getUseFloatingPointAccumulators();
        string code;
        const bool commonSource = MetalSourceAdapter::isCommonSource(source);
        map<string, string> allDefines = commonSource ? compilationDefines : map<string, string>();
        for (auto& define : defines)
            allDefines[define.first] = define.second;
        // This is an internal selection result, not a caller-provided override.
        allDefines.erase("OPENMM_METAL_USE_TILED_ACQ_REL_BARRIERS");
        if (hasSourcePolicy(source, "PRECISE_MATH"))
            allDefines["OPENMM_METAL_REQUIRE_SAFE_MATH"] = "1";
        if (commonSource && floatingAccumulators)
            allDefines["OPENMM_METAL_FLOAT_ACCUMULATORS"] = "1";
        // Shader arithmetic policy lives in mathPolicy.metal. Keep only the
        // runtime capability and existing numerical-safety decisions here.
        const bool strictMath = allDefines.count("OPENMM_METAL_REQUIRE_SAFE_MATH") != 0;
        const bool fastMath = false;
        allDefines["OPENMM_METAL_USE_FAST_MATH"] = fastMath ? "1" : "0";
        if (@available(macOS 15.0, *))
            allDefines["OPENMM_METAL_HAS_MATH_PRAGMAS"] = "1";
        else
            allDefines["OPENMM_METAL_HAS_MATH_PRAGMAS"] = "0";
        for (auto& define : allDefines)
            code += "#define "+define.first+" "+define.second+"\n";
        code += MetalKernelSources::mathPolicy;
        code += commonSource ? MetalKernelSources::common+MetalKernelSources::gbsaTransport+
                MetalSourceAdapter::translate(source, floatingAccumulators) : source;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.languageVersion = MetalLanguagePolicy::languageVersion(getMetalLanguageVersion());
        if (@available(macOS 15.0, *)) {
            // Keep function arguments/results NaN-safe. A source pragma alone
            // does not remove the compiler's default no-NaN call assumptions.
            // mathPolicy.metal still selects safe/fast arithmetic in the shader.
            options.mathMode = MTLMathModeSafe;
            // MSL's arithmetic pragma does not select library math functions.
            options.mathFloatingPointFunctions = fastMath ? MTLMathFloatingPointFunctionsFast : MTLMathFloatingPointFunctionsPrecise;
        }
        else {
            // Compatibility only: these replacement properties require macOS 15.
            options.fastMathEnabled = fastMath;
        }
        NSError* error = nil;
        id<MTLLibrary> library = [impl->device newLibraryWithSource:[NSString stringWithUTF8String:code.c_str()]
                options:options error:&error];
        if (library == nil)
            throw OpenMMException("Error compiling Metal program: "+
                    (error == nil ? string("unknown error") : string(error.localizedDescription.UTF8String)));
        return ComputeProgram(new MetalProgram(*this, (__bridge void*) library, commonSource,
                source, allDefines, strictMath));
    }
}

int MetalContext::getMaxThreadBlockSize() const {
    return impl->device.maxThreadsPerThreadgroup.width;
}

size_t MetalContext::getMaxThreadgroupMemory() const {
    return impl->device.maxThreadgroupMemoryLength;
}

int MetalContext::computeThreadBlockSize(double memory) const {
    if (!isfinite(memory) || memory < 0)
        throw OpenMMException("Invalid shared memory requirement");
    int maximum = getMaxThreadBlockSize();
    if (memory > 0)
        maximum = min(double(maximum), floor(impl->device.maxThreadgroupMemoryLength/memory));
    if (maximum < getSIMDWidth())
        throw OpenMMException("Shared memory requirement exceeds Metal threadgroup capacity");
    return (maximum/getSIMDWidth())*getSIMDWidth();
}

void* MetalContext::getPinnedBuffer() {
    return impl->pinnedBuffer.contents;
}

void* MetalContext::getPinnedBufferHandle() const {
    return (__bridge void*) impl->pinnedBuffer;
}

ThreadPool& MetalContext::getThreadPool() {
    return impl->threads;
}

void MetalContext::getPeriodicBoxVectors(Vec3& a, Vec3& b, Vec3& c) const {
    a = periodicBoxVectors[0];
    b = periodicBoxVectors[1];
    c = periodicBoxVectors[2];
}

void MetalContext::setPeriodicBoxVectors(const Vec3& a, const Vec3& b, const Vec3& c) {
    periodicBoxVectors[0] = a;
    periodicBoxVectors[1] = b;
    periodicBoxVectors[2] = c;
}

void MetalContext::flushQueue() {
    // Match CudaContext/HipContext: flush also waits for the current stream.
    getCurrentMetalQueue().finish();
}

ArrayInterface& MetalContext::getForceBuffers() {
    throw OpenMMException("Metal does not use floating point force buffers");
}

ArrayInterface& MetalContext::getFloatForceBuffer() {
    throw OpenMMException("Metal does not use floating point force buffers");
}

ComputeSort MetalContext::createSort(ComputeSortImpl::SortTrait* trait, unsigned int length, bool uniform) {
    return ComputeSort(new MetalSort(*this, trait, length, uniform));
}

IntegrationUtilities& MetalContext::getIntegrationUtilities() {
    if (!integration)
        integration.reset(new MetalIntegrationUtilities(*this, system));
    return *integration;
}

ExpressionUtilities& MetalContext::getExpressionUtilities() {
    if (!expression)
        expression.reset(new ExpressionUtilities(*this));
    return *expression;
}

BondedUtilities& MetalContext::getBondedUtilities() {
    if (!bonded)
        bonded.reset(new BondedUtilities(*this));
    return *bonded;
}

NonbondedUtilities& MetalContext::getNonbondedUtilities() {
    if (!nonbonded)
        nonbonded.reset(createNonbondedUtilities());
    return *nonbonded;
}

NonbondedUtilities* MetalContext::createNonbondedUtilities() {
    return new MetalNonbondedUtilities(*this);
}

FFT3D MetalContext::createFFT(int xsize, int ysize, int zsize, bool realToComplex) {
    return FFT3D(new MetalFFT3D(*this, xsize, ysize, zsize, realToComplex));
}

int MetalContext::findLegalFFTDimension(int minimum) {
    return MetalFFT3D::findLegalDimension(minimum);
}

void MetalContext::resizePinnedBuffer(size_t bytes) {
    if (impl->pinnedBuffer.length >= bytes)
        return;
    getCurrentMetalQueue().finish();
    id<MTLBuffer> buffer = [impl->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (buffer == nil)
        throw OpenMMException("Error growing the Metal pinned transfer buffer");
    impl->pinnedBuffer = buffer;
}

void MetalContext::initialize() {
    if (initialized)
        return;
    getBondedUtilities().initialize(system);
    MetalNonbondedUtilities& nb = static_cast<MetalNonbondedUtilities&>(getNonbondedUtilities());
    int energySize = max(getNumThreadBlocks()*ThreadBlockSize, nb.getNumEnergyBuffers());
    if (energyBuffer.getSize() != energySize)
        energyBuffer.resize(energySize);
    if (!energyParamDerivNames.empty()) {
        energyParamDerivBuffer.initialize<float>(*this, energyParamDerivNames.size()*energySize, "energyParamDerivBuffer");
        addAutoclearBuffer(energyParamDerivBuffer);
    }
    resizePinnedBuffer(max(longForceBuffer.getSize()*longForceBuffer.getElementSize(), energyBuffer.getSize()*energyBuffer.getElementSize()));
    findMoleculeGroups();
    nb.initialize(system);
    initialized = true;
}

double MetalContext::reduceEnergy() {
    int blockSize = min(512, reduceEnergyKernel->getMaxBlockSize());
    reduceEnergyKernel->setArg(0, energyBuffer);
    reduceEnergyKernel->setArg(1, energySum);
    reduceEnergyKernel->setArg(2, (int) energyBuffer.getSize());
    reduceEnergyKernel->setArg(3, blockSize);
    reduceEnergyKernel->execute(blockSize*energySum.getSize(), blockSize);
    // Reuse the same pinned workspace as OpenCL's blocking energy readback.
    float* partials = static_cast<float*>(getPinnedBuffer());
    energySum.download(partials, false);
    getCurrentMetalQueue().finish();
    return accumulate(partials, partials+energySum.getSize(), 0.0);
}

mm_float4 MetalContext::getPeriodicBoxSize() const {
    return mm_float4(periodicBoxVectors[0][0], periodicBoxVectors[1][1], periodicBoxVectors[2][2], 0);
}

mm_float4 MetalContext::getInvPeriodicBoxSize() const {
    mm_float4 box = getPeriodicBoxSize();
    return mm_float4(1/box.x, 1/box.y, 1/box.z, 0);
}

mm_float4 MetalContext::getPeriodicBoxVecX() const {
    const Vec3& v = periodicBoxVectors[0];
    return mm_float4(v[0], v[1], v[2], 0);
}

mm_float4 MetalContext::getPeriodicBoxVecY() const {
    const Vec3& v = periodicBoxVectors[1];
    return mm_float4(v[0], v[1], v[2], 0);
}

mm_float4 MetalContext::getPeriodicBoxVecZ() const {
    const Vec3& v = periodicBoxVectors[2];
    return mm_float4(v[0], v[1], v[2], 0);
}
