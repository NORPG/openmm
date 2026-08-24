#ifndef OPENMM_METALINTERNAL_H_
#define OPENMM_METALINTERNAL_H_

/* Private Objective-C++ implementation details for the Metal runtime. */

#ifndef __OBJC__
#error "MetalInternal.h may only be included from Objective-C++ sources"
#endif

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "MetalArray.h"
#include "MetalDeviceCaps.h"
#include "openmm/OpenMMException.h"
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace OpenMM {
namespace detail {

std::string fromNSString(NSString* value);
std::string describeError(NSError* error);

/** Shared native device and command queue state. */
class MetalQueueState : public std::enable_shared_from_this<MetalQueueState> {
public:
    explicit MetalQueueState(size_t deviceIndex);
    explicit MetalQueueState(const std::shared_ptr<MetalQueueState>& parent);
    ~MetalQueueState();

    id<MTLCommandBuffer> makeCommandBuffer(const std::string& label);

    /**
     * Commit a fully encoded command buffer.  Callers hold submissionMutex
     * across command-buffer creation, encoding, and this call.
     */
    void submit(id<MTLCommandBuffer> commandBuffer, bool blocking,
                const std::function<void(bool)>& completion = std::function<void(bool)>());
    void waitUntilIdle();
    void checkForErrors();
    void recordError(const std::string& error);

    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    MetalDeviceCaps caps;
    std::mutex submissionMutex;

private:
    void asyncFinished();
    void waitForAsyncCompletions();

    std::mutex statusMutex;
    std::condition_variable statusChanged;
    size_t pendingAsyncCommands = 0;
    std::vector<std::string> asynchronousErrors;
};

/**
 * Stable ownership identity for one resizable Metal buffer.
 *
 * MetalArray owns this state while it is alive.  Kernels retain the same state
 * when the array is bound, so destroying the public C++ wrapper cannot leave a
 * dangling kernel argument and resizing the array updates existing bindings.
 */
class MetalBufferState {
public:
    MetalBufferState(const std::shared_ptr<MetalQueueState>& queue, id<MTLBuffer> buffer) :
            queue(queue), buffer(buffer) {
    }

    id<MTLBuffer> getBuffer() const {
        std::lock_guard<std::mutex> lock(stateMutex);
        return buffer;
    }

    void setBuffer(id<MTLBuffer> replacement) {
        std::lock_guard<std::mutex> lock(stateMutex);
        buffer = replacement;
    }

    std::shared_ptr<MetalQueueState> queue;

private:
    mutable std::mutex stateMutex;
    id<MTLBuffer> buffer;
};

/** Narrow private bridge used by MetalKernel without exposing Metal types. */
class MetalArrayAccess {
public:
    static std::shared_ptr<MetalQueueState> getQueueState(const MetalArray& array);
    static std::shared_ptr<MetalBufferState> getBufferState(const MetalArray& array);
};

} // namespace detail
} // namespace OpenMM

#endif /*OPENMM_METALINTERNAL_H_*/
