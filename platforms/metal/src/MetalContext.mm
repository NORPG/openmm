/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 *                                                                            *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Ported from the OpenMM CUDA Platform.                                      *
 * Source: platforms/cuda/src/CudaContext.cpp                                 *
 *                                                                            *
 * Original CUDA Platform code:                                               *
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
#include "MetalProgram.h"
#include "MetalQueue.h"
#include "MetalCommand.h"
#include "openmm/internal/ThreadPool.h"
#import <Metal/Metal.h>
#include <dispatch/dispatch.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>

using namespace OpenMM;
using namespace std;

struct MetalContext::Impl {
    id<MTLDevice> device = nil;
    id<MTL4Compiler> compiler = nil;
    id<MTLBuffer> pinnedBuffer = nil;
    ThreadPool threads;
    Impl() : threads(1) {
    }
};

MetalContext::MetalContext(const System& system) : ComputeContext(system), energyWorkspace(0.0) {
    try {
        impl.reset(new Impl());
        @autoreleasepool {
            impl->device = MTLCreateSystemDefaultDevice();
            if (impl->device == nil)
                throw OpenMMException("No Metal device is available");
            if (![impl->device supportsFamily:MTLGPUFamilyMetal4])
                throw OpenMMException("The Metal Platform requires a Metal 4 capable Apple silicon GPU");
            NSError* error = nil;
            impl->compiler = [impl->device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
            if (impl->compiler == nil)
                throw OpenMMException("Error creating Metal 4 compiler");
            defaultQueue = createQueue();
            restoreDefaultQueue();
            numAtoms = system.getNumParticles();
            if (numAtoms > numeric_limits<int>::max()-TileSize)
                throw OpenMMException("Too many atoms for the Metal Platform");
            paddedNumAtoms = max(TileSize, TileSize*((numAtoms+TileSize-1)/TileSize));
            posq.initialize<mm_float4>(*this, paddedNumAtoms, "posq");
            velm.initialize<mm_float4>(*this, paddedNumAtoms, "velm");
            // Standard CUDA/HIP component planes; no custom fixed-point storage ABI.
            force.initialize<int64_t>(*this, 3*size_t(paddedNumAtoms), "force");
            energyBuffer.initialize<float>(*this, getNumThreadBlocks()*ThreadBlockSize, "energyBuffer");
            atomIndexArray.initialize<int>(*this, paddedNumAtoms, "atomIndex");
            atomIndex.resize(paddedNumAtoms);
            iota(atomIndex.begin(), atomIndex.end(), 0);
            atomIndexArray.upload(atomIndex);
            posCellOffsets.resize(paddedNumAtoms, mm_int4(0, 0, 0, 0));
            vector<mm_float4> velocities(paddedNumAtoms, mm_float4(0, 0, 0, 0));
            for (int i = 0; i < numAtoms; i++) {
                double mass = system.getParticleMass(i);
                velocities[i].w = (mass == 0.0 ? 0.0f : (float) (1.0/mass));
            }
            velm.upload(velocities);
            clearBuffer(posq);
            clearBuffer(force);
            clearBuffer(energyBuffer);
            addAutoclearBuffer(force);
            addAutoclearBuffer(energyBuffer);
            size_t pinnedBytes = max(force.getSize()*force.getElementSize(),
                    energyBuffer.getSize()*energyBuffer.getElementSize());
            impl->pinnedBuffer = [impl->device newBufferWithLength:pinnedBytes options:MTLResourceStorageModeShared];
            if (impl->pinnedBuffer == nil)
                throw OpenMMException("Error creating Metal pinned transfer buffer");
            system.getDefaultPeriodicBoxVectors(periodicBoxVectors[0], periodicBoxVectors[1], periodicBoxVectors[2]);
            getCurrentMetalQueue().finish();
        }
    }
    catch (...) {
        delete workThread;
        workThread = nullptr;
        throw;
    }
}

MetalContext::~MetalContext() {
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

void* MetalContext::getDevice() const {
    return (__bridge void*) impl->device;
}

void* MetalContext::getCompiler() const {
    return (__bridge void*) impl->compiler;
}

ContextImpl* MetalContext::getContextImpl() {
    throw OpenMMException("The Metal Platform is not attached to a simulation Context");
}

ComputeQueue MetalContext::createQueue() {
    return ComputeQueue(new MetalQueue(getDevice()));
}

MetalQueue& MetalContext::getCurrentMetalQueue() {
    MetalQueue* queue = dynamic_cast<MetalQueue*>(currentQueue.get());
    if (queue == nullptr)
        throw OpenMMException("The current queue is not a MetalQueue");
    id<MTL4CommandQueue> native = (__bridge id<MTL4CommandQueue>) queue->getQueue();
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
    if (metalArray == nullptr || &metalArray->getContext() != this)
        throw OpenMMException("Array argument is not a MetalArray from this context");
    return *metalArray;
}

ComputeEvent MetalContext::createEvent() {
    return ComputeEvent(new MetalEvent(*this));
}

ComputeProgram MetalContext::compileProgram(const string source, const map<string, string>& defines) {
    @autoreleasepool {
        string code;
        for (auto& define : defines)
            code += "#define "+define.first+" "+define.second+"\n";
        code += source;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.languageVersion = MTLLanguageVersion4_0;
        options.mathMode = MTLMathModeFast;
        options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsFast;
        MTL4LibraryDescriptor* descriptor = [MTL4LibraryDescriptor new];
        descriptor.source = [NSString stringWithUTF8String:code.c_str()];
        descriptor.options = options;
        // The synchronous Metal 4 API returns a dangling NSError with API validation
        // on macOS 26.6.2. Retain callback results and wait to preserve Common semantics.
        __block id<MTLLibrary> library = nil;
        __block NSError* error = nil;
        dispatch_semaphore_t ready = dispatch_semaphore_create(0);
        [impl->compiler newLibraryWithDescriptor:descriptor completionHandler:^(id<MTLLibrary> result, NSError* failure) {
            library = result;
            error = failure;
            dispatch_semaphore_signal(ready);
        }];
        dispatch_semaphore_wait(ready, DISPATCH_TIME_FOREVER);
        if (library == nil)
            throw OpenMMException("Error compiling Metal program: "+
                    (error == nil ? string("unknown error") : string(error.localizedDescription.UTF8String)));
        return ComputeProgram(new MetalProgram(*this, (__bridge void*) library));
    }
}

int MetalContext::getMaxThreadBlockSize() const {
    return impl->device.maxThreadsPerThreadgroup.width;
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

void MetalContext::clearBuffer(ArrayInterface& array) {
    MetalArray& metalArray = unwrap(array);
    if (metalArray.getSize() == 0)
        return;
    @autoreleasepool {
        MetalQueue& queue = getCurrentMetalQueue();
        auto command = queue.createCommand();
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>) metalArray.getBuffer();
        command->useBuffer(buffer);
        id<MTL4ComputeCommandEncoder> encoder = command->beginCompute();
        [encoder fillBuffer:buffer
                range:NSMakeRange(0, metalArray.getSize()*metalArray.getElementSize()) value:0];
        [encoder endEncoding];
        queue.submit(command);
    }
}

void MetalContext::addAutoclearBuffer(ArrayInterface& array) {
    unwrap(array);
    autoclearBuffers.push_back(&array);
}

void MetalContext::clearAutoclearBuffers() {
    for (auto* array : autoclearBuffers)
        clearBuffer(*array);
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

bool MetalContext::getBoxIsTriclinic() const {
    return periodicBoxVectors[0][1] != 0.0 || periodicBoxVectors[0][2] != 0.0 ||
            periodicBoxVectors[1][0] != 0.0 || periodicBoxVectors[1][2] != 0.0 ||
            periodicBoxVectors[2][0] != 0.0 || periodicBoxVectors[2][1] != 0.0;
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

ArrayInterface& MetalContext::getPosqCorrection() {
    throw OpenMMException("Metal does not use mixed precision");
}

ArrayInterface& MetalContext::getForceBuffers() {
    throw OpenMMException("Metal does not use floating point force buffers");
}

ArrayInterface& MetalContext::getFloatForceBuffer() {
    throw OpenMMException("Metal does not use floating point force buffers");
}

ArrayInterface& MetalContext::getEnergyParamDerivBuffer() {
    throw OpenMMException("Metal energy parameter derivatives are not implemented");
}

void MetalContext::addEnergyParameterDerivative(const string& param) {
    throw OpenMMException("Metal energy parameter derivatives are not implemented");
}

ComputeSort MetalContext::createSort(ComputeSortImpl::SortTrait* trait, unsigned int length, bool uniform) {
    delete trait;
    throw OpenMMException("Metal sorting is not implemented");
}

IntegrationUtilities& MetalContext::getIntegrationUtilities() {
    throw OpenMMException("Metal integration utilities are not implemented");
}

ExpressionUtilities& MetalContext::getExpressionUtilities() {
    throw OpenMMException("Metal expression utilities are not implemented");
}

BondedUtilities& MetalContext::getBondedUtilities() {
    throw OpenMMException("Metal bonded utilities are not implemented");
}

NonbondedUtilities& MetalContext::getNonbondedUtilities() {
    throw OpenMMException("Metal nonbonded utilities are not implemented");
}

NonbondedUtilities* MetalContext::createNonbondedUtilities() {
    throw OpenMMException("Metal nonbonded utilities are not implemented");
}

FFT3D MetalContext::createFFT(int xsize, int ysize, int zsize, bool realToComplex) {
    throw OpenMMException("Metal FFT is not implemented");
}

void MetalContext::setCharges(const vector<double>& charges) {
    throw OpenMMException("Metal nonbonded charges are not implemented");
}

bool MetalContext::requestPosqCharges() {
    throw OpenMMException("Metal nonbonded charges are not implemented");
}
