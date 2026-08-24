#include "MetalDeviceCaps.h"
#include "MetalInternal.h"

using namespace OpenMM;
using namespace OpenMM::detail;
using namespace std;

MetalDeviceCaps MetalDeviceCaps::fromBridgeSnapshot(const void* snapshot) {
    if (snapshot == nullptr)
        throw OpenMMException("Cannot read capabilities from a null Metal device snapshot");
    const OMMMetalDeviceCapsC& native = *static_cast<const OMMMetalDeviceCapsC*>(snapshot);

    MetalDeviceCaps caps;
    if (native.name != nullptr)
        caps.name = native.name;
    caps.registryId = native.registryId;
    caps.lowPower = (native.lowPower != 0);
    caps.removable = (native.removable != 0);
    caps.unifiedMemory = (native.unifiedMemory != 0);
    caps.maxBufferLength = static_cast<size_t>(native.maxBufferLength);
    caps.recommendedMaxWorkingSetSize = static_cast<size_t>(native.recommendedMaxWorkingSetSize);
    caps.maxThreadgroupMemoryLength = static_cast<size_t>(native.maxThreadgroupMemoryLength);
    caps.maxThreadsPerThreadgroup = static_cast<size_t>(native.maxThreadsPerThreadgroup);
    caps.argumentBufferTier = native.argumentBufferTier;
    caps.highestAppleGpuFamily = native.highestAppleGpuFamily;
    caps.mac2Family = (native.mac2 != 0);
    caps.metal3Family = (native.metal3 != 0);
    caps.metal4Family = (native.metal4 != 0);
    return caps;
}

vector<MetalDeviceCaps> MetalDeviceCaps::enumerate() {
    OMMMetalErrorC error = {};
    size_t count = 0;
    int32_t status = OMMMetalDeviceCount(&count, &error);
    checkMetalBridgeResult(status, error, "Cannot enumerate Metal devices");

    vector<MetalDeviceCaps> result;
    result.reserve(count);
    for (size_t index = 0; index < count; index++) {
        ScopedMetalDeviceCaps nativeCaps;
        OMMMetalErrorC capsError = {};
        status = OMMMetalDeviceCapsAt(index, &nativeCaps.value, &capsError);
        checkMetalBridgeResult(status, capsError, "Cannot query Metal device capabilities");
        result.push_back(fromBridgeSnapshot(&nativeCaps.value));
    }
    return result;
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
