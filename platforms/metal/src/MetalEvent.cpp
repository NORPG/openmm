#include "MetalEvent.h"
#include "MetalInternal.h"

#include <mutex>

using namespace OpenMM;
using namespace OpenMM::detail;
using namespace std;

struct MetalEvent::Impl {
    explicit Impl(const shared_ptr<MetalQueueState>& queue) : queue(queue) {
        OMMMetalErrorC error = {};
        int32_t status = OMMMetalEventCreate(queue->queue, &event, &error);
        checkMetalBridgeResult(status, error, "The selected Metal device could not create a shared event");
        if (event == nullptr)
            throw OpenMMException("The selected Metal device could not create a shared event: Swift bridge returned a null event");
    }

    ~Impl() {
        if (event != nullptr)
            OMMMetalEventRelease(event);
    }

    shared_ptr<MetalQueueState> queue;
    OMMMetalEventRef event = nullptr;
    bool enqueued = false;
    mutex lock;
};

MetalEvent::MetalEvent(MetalQueue& queue) : impl(new Impl(queue.state)) {
}

MetalEvent::~MetalEvent() {
}

void MetalEvent::enqueue() {
    lock_guard<mutex> eventLock(impl->lock);
    impl->queue->checkForErrors();

    OMMMetalErrorC error = {};
    lock_guard<mutex> queueLock(impl->queue->submissionMutex);
    int32_t status = OMMMetalEventEnqueue(impl->event, &error);
    checkMetalBridgeResult(status, error, "Metal failed to enqueue an event signal");
    impl->enqueued = true;
}

void MetalEvent::wait() {
    {
        lock_guard<mutex> eventLock(impl->lock);
        if (!impl->enqueued)
            throw OpenMMException("Cannot wait for a MetalEvent before enqueue() has been called");
    }

    OMMMetalErrorC error = {};
    int32_t status = OMMMetalEventWait(impl->event, &error);
    checkMetalBridgeResult(status, error, "Metal event signal failed");
}

void MetalEvent::queueWait(ComputeQueue queue) {
    MetalQueue* metalQueue = dynamic_cast<MetalQueue*>(queue.get());
    if (metalQueue == nullptr)
        throw OpenMMException("MetalEvent can only synchronize a MetalQueue");

    lock_guard<mutex> eventLock(impl->lock);
    if (!impl->enqueued)
        throw OpenMMException("Cannot enqueue a wait for a MetalEvent before enqueue() has been called");
    if (metalQueue->state->caps.getRegistryId() != impl->queue->caps.getRegistryId())
        throw OpenMMException("MetalEvent cannot synchronize queues from different devices");
    metalQueue->state->checkForErrors();

    OMMMetalErrorC error = {};
    lock_guard<mutex> queueLock(metalQueue->state->submissionMutex);
    int32_t status = OMMMetalEventQueueWait(impl->event, metalQueue->state->queue, &error);
    checkMetalBridgeResult(status, error, "Metal failed to enqueue an event wait");
}
