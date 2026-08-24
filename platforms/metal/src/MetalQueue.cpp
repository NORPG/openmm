#include "MetalQueue.h"
#include "MetalInternal.h"

#include <utility>

using namespace OpenMM;
using namespace OpenMM::detail;
using namespace std;

void OpenMM::detail::checkMetalBridgeResult(int32_t status, OMMMetalErrorC& error,
                                             const string& fallbackMessage) {
    if (status == OMM_METAL_SUCCESS) {
        OMMMetalErrorClear(&error);
        return;
    }

    string message;
    try {
        if (error.message != nullptr)
            message = error.message;
    }
    catch (...) {
        OMMMetalErrorClear(&error);
        throw;
    }
    OMMMetalErrorClear(&error);
    if (message.empty())
        message = fallbackMessage;
    throw OpenMMException(message);
}

MetalQueueState::MetalQueueState(size_t deviceIndex) {
    ScopedMetalDeviceCaps nativeCaps;
    OMMMetalErrorC error = {};
    OMMMetalQueueRef newQueue = nullptr;
    int32_t status = OMMMetalQueueCreate(deviceIndex, &newQueue, &nativeCaps.value, &error);
    checkMetalBridgeResult(status, error, "Cannot create a Metal command queue");
    if (newQueue == nullptr)
        throw OpenMMException("Cannot create a Metal command queue: Swift bridge returned a null queue");

    queue = newQueue;
    try {
        caps = MetalDeviceCaps::fromBridgeSnapshot(&nativeCaps.value);
    }
    catch (...) {
        OMMMetalQueueRelease(queue);
        queue = nullptr;
        throw;
    }
}

MetalQueueState::MetalQueueState(const shared_ptr<MetalQueueState>& parent) {
    if (parent == nullptr || parent->queue == nullptr)
        throw OpenMMException("Cannot create a sibling Metal command queue from a null parent");

    OMMMetalErrorC error = {};
    OMMMetalQueueRef newQueue = nullptr;
    int32_t status = OMMMetalQueueCreateSibling(parent->queue, &newQueue, &error);
    checkMetalBridgeResult(status, error, "Cannot create a sibling Metal command queue");
    if (newQueue == nullptr)
        throw OpenMMException("Cannot create a sibling Metal command queue: Swift bridge returned a null queue");

    queue = newQueue;
    try {
        caps = parent->caps;
    }
    catch (...) {
        OMMMetalQueueRelease(queue);
        queue = nullptr;
        throw;
    }
}

MetalQueueState::~MetalQueueState() {
    if (queue != nullptr)
        OMMMetalQueueRelease(queue);
}

void MetalQueueState::waitUntilIdle() {
    OMMMetalErrorC error = {};
    lock_guard<mutex> lock(submissionMutex);
    int32_t status = OMMMetalQueueWaitUntilIdle(queue, &error);
    checkMetalBridgeResult(status, error, "Metal failed while waiting for the command queue");
}

void MetalQueueState::checkForErrors() {
    OMMMetalErrorC error = {};
    int32_t status = OMMMetalQueueCheckForErrors(queue, &error);
    checkMetalBridgeResult(status, error, "Metal command queue reported an asynchronous error");
}

MetalQueue::MetalQueue(size_t deviceIndex) : state(make_shared<MetalQueueState>(deviceIndex)) {
}

MetalQueue::MetalQueue(const shared_ptr<MetalQueueState>& parent) : state(make_shared<MetalQueueState>(parent)) {
}

MetalQueue::~MetalQueue() {
    if (state != nullptr) {
        try {
            state->waitUntilIdle();
        }
        catch (...) {
            // Destructors cannot report asynchronous failures.  Explicitly
            // call waitUntilIdle() at API boundaries where errors matter.
        }
    }
}

shared_ptr<MetalQueue> MetalQueue::createSiblingQueue() const {
    return shared_ptr<MetalQueue>(new MetalQueue(state));
}

void MetalQueue::waitUntilIdle() {
    state->waitUntilIdle();
}

void MetalQueue::checkForErrors() {
    state->checkForErrors();
}

const MetalDeviceCaps& MetalQueue::getDeviceCaps() const {
    return state->caps;
}
