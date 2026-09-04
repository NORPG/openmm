#include "MetalQueue.h"
#include "MetalInternal.h"

#include <sstream>
#include <utility>

using namespace OpenMM;
using namespace OpenMM::detail;
using namespace std;

string OpenMM::detail::fromNSString(NSString* value) {
    if (value == nil)
        return string();
    const char* text = value.UTF8String;
    return (text == NULL ? string() : string(text));
}

string OpenMM::detail::describeError(NSError* error) {
    if (error == nil)
        return "unknown Metal error";
    string message = fromNSString(error.localizedDescription);
    NSString* reason = error.localizedFailureReason;
    if (reason != nil) {
        string detail = fromNSString(reason);
        if (!detail.empty() && detail != message)
            message += ": "+detail;
    }
    return message.empty() ? string("unknown Metal error") : message;
}

MetalQueueState::MetalQueueState(size_t deviceIndex) : device(nil), commandQueue(nil) {
    @autoreleasepool {
        NSArray<id<MTLDevice> >* devices = MTLCopyAllDevices();
        if (deviceIndex >= devices.count) {
            stringstream message;
            message << "Cannot create Metal queue: device index " << deviceIndex
                    << " is out of range (" << devices.count << " devices available)";
            throw OpenMMException(message.str());
        }
        device = devices[deviceIndex];
        commandQueue = [device newCommandQueue];
        if (commandQueue == nil)
            throw OpenMMException("Cannot create a Metal command queue for device "+fromNSString(device.name));
        caps = MetalDeviceCaps::fromNativeDevice((__bridge void*) device);
        commandQueue.label = @"OpenMM Metal queue";
    }
}

MetalQueueState::MetalQueueState(const shared_ptr<MetalQueueState>& parent) :
        device(parent->device), commandQueue(nil), caps(parent->caps) {
    @autoreleasepool {
        commandQueue = [device newCommandQueue];
        if (commandQueue == nil)
            throw OpenMMException("Cannot create a sibling Metal command queue for device "+caps.getName());
        commandQueue.label = @"OpenMM Metal sibling queue";
    }
}

MetalQueueState::~MetalQueueState() {
}

id<MTLCommandBuffer> MetalQueueState::makeCommandBuffer(const string& label) {
    id<MTLCommandBuffer> command = [commandQueue commandBuffer];
    if (command == nil)
        throw OpenMMException("Metal failed to allocate a command buffer for "+label);
    NSString* nativeLabel = [[NSString alloc] initWithBytes:label.data()
                                                     length:label.size()
                                                   encoding:NSUTF8StringEncoding];
    command.label = nativeLabel;
    return command;
}

void MetalQueueState::submit(id<MTLCommandBuffer> command, bool blocking,
                             const function<void(bool)>& completion) {
    if (command == nil)
        throw OpenMMException("Cannot submit a null Metal command buffer");
    if (blocking) {
        [command commit];
        [command waitUntilCompleted];
        bool succeeded = (command.status == MTLCommandBufferStatusCompleted);
        if (completion)
            completion(succeeded);
        if (!succeeded)
            throw OpenMMException("Metal command '"+fromNSString(command.label)+"' failed: "+describeError(command.error));
        return;
    }

    {
        lock_guard<mutex> lock(statusMutex);
        pendingAsyncCommands++;
    }
    shared_ptr<MetalQueueState> self = shared_from_this();
    function<void(bool)> callback = completion;
    [command addCompletedHandler:^(id<MTLCommandBuffer> completed) {
        bool succeeded = (completed.status == MTLCommandBufferStatusCompleted);
        if (callback) {
            try {
                callback(succeeded);
            }
            catch (const exception& e) {
                self->recordError("Metal completion handler failed: "+string(e.what()));
            }
            catch (...) {
                self->recordError("Metal completion handler failed with an unknown exception");
            }
        }
        if (!succeeded)
            self->recordError("Metal command '"+fromNSString(completed.label)+"' failed: "+describeError(completed.error));
        self->asyncFinished();
    }];
    [command commit];
}

void MetalQueueState::recordError(const string& error) {
    lock_guard<mutex> lock(statusMutex);
    asynchronousErrors.push_back(error);
}

void MetalQueueState::asyncFinished() {
    {
        lock_guard<mutex> lock(statusMutex);
        if (pendingAsyncCommands > 0)
            pendingAsyncCommands--;
    }
    statusChanged.notify_all();
}

void MetalQueueState::waitForAsyncCompletions() {
    unique_lock<mutex> lock(statusMutex);
    statusChanged.wait(lock, [&] { return pendingAsyncCommands == 0; });
}

void MetalQueueState::checkForErrors() {
    vector<string> errors;
    {
        lock_guard<mutex> lock(statusMutex);
        errors.swap(asynchronousErrors);
    }
    if (!errors.empty()) {
        string message = errors[0];
        for (size_t i = 1; i < errors.size(); i++)
            message += "\n"+errors[i];
        throw OpenMMException(message);
    }
}

void MetalQueueState::waitUntilIdle() {
    @autoreleasepool {
        string markerError;
        try {
            lock_guard<mutex> lock(submissionMutex);
            id<MTLCommandBuffer> marker = makeCommandBuffer("wait until idle");
            submit(marker, true);
        }
        catch (const exception& e) {
            markerError = e.what();
        }
        catch (...) {
            markerError = "Unknown error while submitting the Metal queue marker";
        }
        // Completion handlers run independently of command-buffer ordering.
        // Even if the marker failed, wait for them so nonblocking host work
        // cannot escape this synchronization boundary.
        waitForAsyncCompletions();
        string pendingError;
        try {
            checkForErrors();
        }
        catch (const exception& e) {
            pendingError = e.what();
        }
        catch (...) {
            pendingError = "Unknown asynchronous Metal queue error";
        }
        if (!markerError.empty() && !pendingError.empty())
            throw OpenMMException(markerError+"\n"+pendingError);
        if (!markerError.empty())
            throw OpenMMException(markerError);
        if (!pendingError.empty())
            throw OpenMMException(pendingError);
    }
}

MetalQueue::MetalQueue(size_t deviceIndex) : state(make_shared<MetalQueueState>(deviceIndex)) {
}

MetalQueue::MetalQueue(const shared_ptr<MetalQueueState>& parent) : state(make_shared<MetalQueueState>(parent)) {
}

MetalQueue::~MetalQueue() {
    if (state != NULL) {
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
