/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 *                                                                            *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Ported from the OpenMM CUDA Platform.                                      *
 * Source: platforms/cuda/src/CudaQueue.cpp                                   *
 *                                                                            *
 * Original CUDA Platform code:                                               *
 * Portions copyright (c) 2025 Stanford University and the Authors.           *
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

#include "MetalQueue.h"
#include "MetalCommand.h"
#include "openmm/OpenMMException.h"
#include <algorithm>
#include <deque>
#include <string>

using namespace OpenMM;
using namespace std;

MetalQueue::Command::Command(MetalQueue* owner, id<MTLDevice> device) : owner(owner) {
    allocator = [device newCommandAllocator];
    buffer = [device newCommandBuffer];
    NSError* error = nil;
    residency = [device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:&error];
    resources = [NSMutableArray new];
    if (allocator == nil || buffer == nil || residency == nil)
        throw OpenMMException("Error creating Metal 4 command resources");
    [buffer beginCommandBufferWithAllocator:allocator];
}

id<MTL4ComputeCommandEncoder> MetalQueue::Command::beginCompute() {
    id<MTL4ComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
    if (encoder == nil)
        throw OpenMMException("Error creating Metal 4 compute encoder");
    // Metal 4 has no implicit hazard tracking. Preserve CUDA/HIP stream ordering.
    MTLStages stages = MTLStageDispatch | MTLStageBlit;
    [encoder barrierAfterQueueStages:stages beforeStages:stages visibilityOptions:MTL4VisibilityOptionDevice];
    return encoder;
}

void MetalQueue::Command::useBuffer(id<MTLBuffer> buffer) {
    [resources addObject:buffer];
    [residency addAllocation:buffer];
}

void MetalQueue::Command::waitUntilCompleted() {
    unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [this] { return completed; });
}

bool MetalQueue::Command::isComplete() {
    lock_guard<std::mutex> lock(mutex);
    return completed;
}

class MetalQueue::Impl {
public:
    id<MTL4CommandQueue> queue;
    id<MTLBuffer> markerBuffer;
    deque<shared_ptr<Command>> pending;
};

MetalQueue::MetalQueue(void* device) : impl(new Impl()) {
    if (device == nullptr)
        throw OpenMMException("Cannot create a Metal queue without a device");
    id<MTLDevice> native = (__bridge id<MTLDevice>) device;
    impl->queue = [native newMTL4CommandQueue];
    impl->markerBuffer = [native newBufferWithLength:1 options:MTLResourceStorageModePrivate];
    if (impl->queue == nil || impl->markerBuffer == nil)
        throw OpenMMException("Error creating Metal 4 command queue");
}

MetalQueue::~MetalQueue() {
    // Finish outstanding device work; explicit waits report execution errors.
    for (const auto& command : impl->pending)
        command->waitUntilCompleted();
}

void* MetalQueue::getQueue() const {
    return (__bridge void*) impl->queue;
}

shared_ptr<MetalQueue::Command> MetalQueue::createCommand() {
    return make_shared<Command>(this, impl->queue.device);
}

shared_ptr<MetalQueue::Command> MetalQueue::createMarker() {
    auto command = createCommand();
    command->useBuffer(impl->markerBuffer);
    id<MTL4ComputeCommandEncoder> encoder = command->beginCompute();
    // An actual GPU operation also orders feedback after a queue-level event wait.
    [encoder fillBuffer:impl->markerBuffer range:NSMakeRange(0, 1) value:0];
    [encoder endEncoding];
    return command;
}

void MetalQueue::submit(const shared_ptr<Command>& command) {
    if (!command || command->owner != this)
        throw OpenMMException("Metal command buffer does not belong to this queue");
    if (command->submitted)
        throw OpenMMException("Metal command buffer has already been submitted");
    while (!impl->pending.empty() && impl->pending.front()->isComplete())
        wait(impl->pending.front());
    [command->residency commit];
    [command->buffer useResidencySet:command->residency];
    [command->buffer endCommandBuffer];
    MTL4CommitOptions* options = [MTL4CommitOptions new];
    // Capture ownership, not the queue: feedback can run on a different CPU thread.
    auto retained = command;
    [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        {
            lock_guard<mutex> lock(retained->mutex);
            retained->error = feedback.error;
            retained->completed = true;
        }
        retained->condition.notify_all();
    }];
    impl->pending.push_back(command);
    command->submitted = true;
    id<MTL4CommandBuffer> buffer = command->buffer;
    // Do not enqueue an event operation until all fallible preparation is complete.
    if (command->signalEvent != nil)
        [impl->queue signalEvent:command->signalEvent value:1];
    if (command->waitEvent != nil)
        [impl->queue waitForEvent:command->waitEvent value:1];
    [impl->queue commit:&buffer count:1 options:options];
}

void MetalQueue::finish() {
    if (!impl->pending.empty())
        wait(impl->pending.back());
}

void MetalQueue::wait(const shared_ptr<Command>& command) {
    if (!command || command->owner != this || !command->submitted)
        throw OpenMMException("Cannot wait for an unsubmitted or foreign Metal command buffer");
    // Copy the marker: callers may pass a reference to an entry we are about to erase.
    auto marker = command;
    auto markerPosition = find(impl->pending.begin(), impl->pending.end(), marker);
    NSError* error = nil;
    size_t count = (markerPosition == impl->pending.end() ? 0 : markerPosition-impl->pending.begin()+1);
    for (size_t i = 0; i < count; i++) {
        auto current = impl->pending.front();
        current->waitUntilCompleted();
        if (current->error != nil && error == nil)
            error = current->error;
        impl->pending.pop_front();
    }
    marker->waitUntilCompleted();
    if (marker->error != nil && error == nil)
        error = marker->error;
    if (error != nil) {
        const char* message = error.localizedDescription.UTF8String;
        throw OpenMMException("Error executing Metal 4 command buffer: "+
                string(message == nullptr ? "Unknown GPU error" : message));
    }
}
