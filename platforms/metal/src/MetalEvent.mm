#include "MetalEvent.h"
#include "MetalInternal.h"

#include <mutex>

using namespace OpenMM;
using namespace OpenMM::detail;
using namespace std;

struct MetalEvent::Impl {
    explicit Impl(const shared_ptr<MetalQueueState>& queue) : queue(queue) {
        event = [queue->device newSharedEvent];
        if (event == nil)
            throw OpenMMException("The selected Metal device could not create a shared event");
        event.label = @"OpenMM Metal event";
    }

    shared_ptr<MetalQueueState> queue;
    id<MTLSharedEvent> event = nil;
    id<MTLCommandBuffer> signalCommand = nil;
    uint64_t value = 0;
    mutex lock;
};

MetalEvent::MetalEvent(MetalQueue& queue) : impl(new Impl(queue.state)) {
}

MetalEvent::~MetalEvent() {
}

void MetalEvent::enqueue() {
    @autoreleasepool {
        lock_guard<mutex> eventLock(impl->lock);
        impl->queue->checkForErrors();
        lock_guard<mutex> queueLock(impl->queue->submissionMutex);
        id<MTLCommandBuffer> command = impl->queue->makeCommandBuffer("signal event");
        impl->value++;
        [command encodeSignalEvent:impl->event value:impl->value];
        impl->signalCommand = command;
        impl->queue->submit(command, false);
    }
}

void MetalEvent::wait() {
    @autoreleasepool {
        id<MTLCommandBuffer> command;
        {
            lock_guard<mutex> eventLock(impl->lock);
            command = impl->signalCommand;
        }
        if (command == nil)
            throw OpenMMException("Cannot wait for a MetalEvent before enqueue() has been called");
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted)
            throw OpenMMException("Metal event signal failed: "+describeError(command.error));
    }
}

void MetalEvent::queueWait(ComputeQueue queue) {
    @autoreleasepool {
        MetalQueue* metalQueue = dynamic_cast<MetalQueue*>(queue.get());
        if (metalQueue == NULL)
            throw OpenMMException("MetalEvent can only synchronize a MetalQueue");

        lock_guard<mutex> eventLock(impl->lock);
        if (impl->signalCommand == nil)
            throw OpenMMException("Cannot enqueue a wait for a MetalEvent before enqueue() has been called");
        if (metalQueue->state->device != impl->queue->device)
            throw OpenMMException("MetalEvent cannot synchronize queues from different devices");
        metalQueue->state->checkForErrors();
        lock_guard<mutex> queueLock(metalQueue->state->submissionMutex);
        id<MTLCommandBuffer> command = metalQueue->state->makeCommandBuffer("wait for event");
        [command encodeWaitForEvent:impl->event value:impl->value];
        metalQueue->state->submit(command, false);
    }
}
