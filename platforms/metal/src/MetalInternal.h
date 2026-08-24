#ifndef OPENMM_METALINTERNAL_H_
#define OPENMM_METALINTERNAL_H_

/* Private C++ implementation details for the Swift-backed Metal runtime. */

#include "MetalArray.h"
#include "MetalDeviceCaps.h"
#include "MetalSwiftBridge.h"
#include "openmm/OpenMMException.h"
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

namespace OpenMM {
namespace detail {

/**
 * Convert one C-ABI result into an OpenMM exception.  This always clears the
 * bridge-owned error message, including on success.
 */
void checkMetalBridgeResult(int32_t status, OMMMetalErrorC& error,
                            const std::string& fallbackMessage);

/** RAII storage for the bridge-owned device name in a capability snapshot. */
class ScopedMetalDeviceCaps {
public:
    ScopedMetalDeviceCaps() = default;
    ~ScopedMetalDeviceCaps() {
        OMMMetalDeviceCapsClear(&value);
    }

    ScopedMetalDeviceCaps(const ScopedMetalDeviceCaps&) = delete;
    ScopedMetalDeviceCaps& operator=(const ScopedMetalDeviceCaps&) = delete;

    OMMMetalDeviceCapsC value = {};
};

/** Shared Swift queue handle and its immutable device capability snapshot. */
class MetalQueueState {
public:
    explicit MetalQueueState(size_t deviceIndex);
    explicit MetalQueueState(const std::shared_ptr<MetalQueueState>& parent);
    ~MetalQueueState();

    void waitUntilIdle();
    void checkForErrors();

    OMMMetalQueueRef queue = nullptr;
    MetalDeviceCaps caps;
    std::mutex submissionMutex;
};

/** Narrow private bridge used by MetalKernel without exposing Swift types. */
class MetalArrayAccess {
public:
    static std::shared_ptr<MetalQueueState> getQueueState(const MetalArray& array);
    static OMMMetalBufferRef getBuffer(const MetalArray& array);
};

} // namespace detail
} // namespace OpenMM

#endif /*OPENMM_METALINTERNAL_H_*/
