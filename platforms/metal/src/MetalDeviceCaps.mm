#include "MetalDeviceCaps.h"
#include "MetalInternal.h"

#include <algorithm>

using namespace OpenMM;
using namespace std;

MetalDeviceCaps MetalDeviceCaps::fromNativeDevice(void* nativeDevice) {
    id<MTLDevice> device = (__bridge id<MTLDevice>) nativeDevice;
    MetalDeviceCaps caps;
    caps.name = detail::fromNSString(device.name);
    caps.registryId = device.registryID;
    caps.lowPower = device.lowPower;
    caps.removable = device.removable;
    caps.unifiedMemory = device.hasUnifiedMemory;
    caps.maxBufferLength = device.maxBufferLength;
    caps.recommendedMaxWorkingSetSize = static_cast<size_t>(device.recommendedMaxWorkingSetSize);
    caps.maxThreadgroupMemoryLength = device.maxThreadgroupMemoryLength;
    caps.maxThreadsPerThreadgroup = device.maxThreadsPerThreadgroup.width;
    caps.argumentBufferTier = (device.argumentBuffersSupport == MTLArgumentBuffersTier2 ? 2 : 1);

    // Apple family values are contiguous (Apple1 == 1001).  Keeping the loop
    // bounded by the newest family in the build SDK avoids claiming support
    // for a family this source has never been validated against.
    for (int family = 1; family <= 10; family++) {
        if ([device supportsFamily:static_cast<MTLGPUFamily>(1000+family)])
            caps.highestAppleGpuFamily = family;
    }
    caps.mac2Family = [device supportsFamily:MTLGPUFamilyMac2];
    caps.metal3Family = [device supportsFamily:MTLGPUFamilyMetal3];
    if (@available(macOS 26.0, *))
        caps.metal4Family = [device supportsFamily:MTLGPUFamilyMetal4];
    return caps;
}

vector<MetalDeviceCaps> MetalDeviceCaps::enumerate() {
    @autoreleasepool {
        NSArray<id<MTLDevice> >* devices = MTLCopyAllDevices();
        vector<MetalDeviceCaps> result;
        result.reserve(devices.count);
        for (id<MTLDevice> device in devices)
            result.push_back(fromNativeDevice((__bridge void*) device));
        return result;
    }
}

const string& MetalDeviceCaps::getName() const {
    return name;
}

uint64_t MetalDeviceCaps::getRegistryId() const {
    return registryId;
}

bool MetalDeviceCaps::isLowPower() const {
    return lowPower;
}

bool MetalDeviceCaps::isRemovable() const {
    return removable;
}

bool MetalDeviceCaps::hasUnifiedMemory() const {
    return unifiedMemory;
}

size_t MetalDeviceCaps::getMaxBufferLength() const {
    return maxBufferLength;
}

size_t MetalDeviceCaps::getRecommendedMaxWorkingSetSize() const {
    return recommendedMaxWorkingSetSize;
}

size_t MetalDeviceCaps::getMaxThreadgroupMemoryLength() const {
    return maxThreadgroupMemoryLength;
}

size_t MetalDeviceCaps::getMaxThreadsPerThreadgroup() const {
    return maxThreadsPerThreadgroup;
}

int MetalDeviceCaps::getArgumentBufferTier() const {
    return argumentBufferTier;
}

int MetalDeviceCaps::getHighestAppleGpuFamily() const {
    return highestAppleGpuFamily;
}

bool MetalDeviceCaps::supportsMac2Family() const {
    return mac2Family;
}

bool MetalDeviceCaps::supportsMetal3Family() const {
    return metal3Family;
}

bool MetalDeviceCaps::supportsMetal4Family() const {
    return metal4Family;
}
