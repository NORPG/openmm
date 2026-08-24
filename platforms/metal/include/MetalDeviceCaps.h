#ifndef OPENMM_METALDEVICECAPS_H_
#define OPENMM_METALDEVICECAPS_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2026 Stanford University and the Authors.           *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 * -------------------------------------------------------------------------- */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include "windowsExportMetal.h"

namespace OpenMM {

namespace detail {
class MetalQueueState;
}

/**
 * A snapshot of the public capabilities reported by one Metal device.
 *
 * This deliberately does not infer support for 64 bit atomics.  Native 64 bit
 * atomic operations needed by OpenMM are not part of the public MSL contract
 * on all devices that report the same GPU family.  Code that needs them must
 * use an explicit compile-and-dispatch probe or a portable fallback.
 */
class OPENMM_EXPORT_METAL MetalDeviceCaps {
public:
    /** Return a snapshot for every Metal device visible to this process. */
    static std::vector<MetalDeviceCaps> enumerate();

    const std::string& getName() const;
    uint64_t getRegistryId() const;
    bool isLowPower() const;
    bool isRemovable() const;
    bool hasUnifiedMemory() const;
    size_t getMaxBufferLength() const;
    size_t getRecommendedMaxWorkingSetSize() const;
    size_t getMaxThreadgroupMemoryLength() const;
    size_t getMaxThreadsPerThreadgroup() const;
    int getArgumentBufferTier() const;
    int getHighestAppleGpuFamily() const;
    bool supportsMac2Family() const;
    bool supportsMetal3Family() const;
    bool supportsMetal4Family() const;

private:
    static MetalDeviceCaps fromBridgeSnapshot(const void* snapshot);

    std::string name;
    uint64_t registryId = 0;
    bool lowPower = false;
    bool removable = false;
    bool unifiedMemory = false;
    size_t maxBufferLength = 0;
    size_t recommendedMaxWorkingSetSize = 0;
    size_t maxThreadgroupMemoryLength = 0;
    size_t maxThreadsPerThreadgroup = 0;
    int argumentBufferTier = 0;
    int highestAppleGpuFamily = 0;
    bool mac2Family = false;
    bool metal3Family = false;
    bool metal4Family = false;

    friend class detail::MetalQueueState;
};

} // namespace OpenMM

#endif /*OPENMM_METALDEVICECAPS_H_*/
