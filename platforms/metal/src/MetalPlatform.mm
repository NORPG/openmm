#include "MetalPlatform.h"
#include "MetalContext.h"
#include "MetalKernelFactory.h"
#include "openmm/Context.h"
#include "openmm/HarmonicBondForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/System.h"
#include "openmm/kernels.h"
#include "openmm/internal/ContextImpl.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cctype>
#include <map>
#include <memory>
#include <string>

using namespace OpenMM;
using namespace std;

namespace {

string toLower(string value) {
    transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(tolower(c)); });
    return value;
}

string getDeviceName(id<MTLDevice> device) {
    const char* name = device.name.UTF8String;
    return name == NULL ? string() : string(name);
}

bool isSupportedDevice(id<MTLDevice> device) {
#if defined(__arm64__) || defined(__aarch64__)
    return device != nil && device.hasUnifiedMemory && !device.removable &&
            [device supportsFamily:MTLGPUFamilyApple1] &&
            [device supportsFamily:MTLGPUFamilyMetal3];
#else
    (void) device;
    return false;
#endif
}

id<MTLDevice> getSupportedDevice() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return isSupportedDevice(device) ? device : nil;
}

string propertyValue(const map<string, string>& properties, const string& name, const string& defaultValue) {
    map<string, string>::const_iterator value = properties.find(name);
    return value == properties.end() ? defaultValue : value->second;
}

} // namespace

class MetalPlatform::PlatformData {
public:
    PlatformData(const System& system, size_t deviceIndex, const string& deviceName, const string& precision) :
            context(new MetalContext(system, deviceIndex)) {
        propertyValues[MetalPlatform::MetalDeviceIndex()] = "0";
        propertyValues[MetalPlatform::MetalDeviceName()] = deviceName;
        propertyValues[MetalPlatform::MetalPrecision()] = precision;
    }

    unique_ptr<MetalContext> context;
    map<string, string> propertyValues;
};

MetalPlatform::MetalPlatform() {
    deprecatedPropertyReplacements["MetalDeviceIndex"] = MetalDeviceIndex();
    deprecatedPropertyReplacements["MetalDeviceName"] = MetalDeviceName();
    deprecatedPropertyReplacements["MetalPrecision"] = MetalPrecision();

    platformProperties.push_back(MetalDeviceIndex());
    platformProperties.push_back(MetalDeviceName());
    platformProperties.push_back(MetalPrecision());
    setPropertyDefaultValue(MetalDeviceIndex(), "0");
    setPropertyDefaultValue(MetalDeviceName(), "");
    setPropertyDefaultValue(MetalPrecision(), "single");

    MetalKernelFactory* factory = new MetalKernelFactory();
    registerKernelFactory(CalcForcesAndEnergyKernel::Name(), factory);
    registerKernelFactory(UpdateStateDataKernel::Name(), factory);
    registerKernelFactory(ApplyConstraintsKernel::Name(), factory);
    registerKernelFactory(VirtualSitesKernel::Name(), factory);
    registerKernelFactory(MinimizeKernel::Name(), factory);
    registerKernelFactory(CalcHarmonicBondForceKernel::Name(), factory);
    registerKernelFactory(IntegrateVerletStepKernel::Name(), factory);
}

double MetalPlatform::getSpeed() const {
    // Keep this experimental slice behind the Reference platform unless the
    // caller explicitly selects Metal.
    return 0.5;
}

bool MetalPlatform::supportsDoublePrecision() const {
    return false;
}

bool MetalPlatform::isPlatformSupported() {
    @autoreleasepool {
        return getSupportedDevice() != nil;
    }
}

MetalContext& MetalPlatform::getMetalContext(ContextImpl& context) {
    PlatformData* data = reinterpret_cast<PlatformData*>(context.getPlatformData());
    if (data == NULL || data->context.get() == NULL)
        throw OpenMMException("The Context does not contain native Metal platform data");
    return *data->context;
}

const MetalContext& MetalPlatform::getMetalContext(const ContextImpl& context) {
    const PlatformData* data = reinterpret_cast<const PlatformData*>(context.getPlatformData());
    if (data == NULL || data->context.get() == NULL)
        throw OpenMMException("The Context does not contain native Metal platform data");
    return *data->context;
}

const string& MetalPlatform::getPropertyValue(const Context& context, const string& property) const {
    const ContextImpl& impl = getContextImpl(context);
    const PlatformData* data = reinterpret_cast<const PlatformData*>(impl.getPlatformData());
    string propertyName = property;
    map<string, string>::const_iterator replacement = deprecatedPropertyReplacements.find(property);
    if (replacement != deprecatedPropertyReplacements.end())
        propertyName = replacement->second;
    if (data != NULL) {
        map<string, string>::const_iterator value = data->propertyValues.find(propertyName);
        if (value != data->propertyValues.end())
            return value->second;
    }
    return Platform::getPropertyValue(context, property);
}

vector<map<string, string> > MetalPlatform::getDevices(const map<string, string>& filters) const {
    @autoreleasepool {
        id<MTLDevice> device = getSupportedDevice();
        if (device == nil)
            return {};

        string requestedIndex = propertyValue(filters, MetalDeviceIndex(), "");
        if (!requestedIndex.empty() && requestedIndex != "0")
            return {};

        string deviceName = getDeviceName(device);
        string requestedName = propertyValue(filters, MetalDeviceName(), "");
        if (!requestedName.empty() && requestedName != deviceName)
            return {};

        string requestedPrecision = toLower(propertyValue(filters, MetalPrecision(), "single"));
        if (requestedPrecision.empty())
            requestedPrecision = "single";
        if (requestedPrecision != "single")
            return {};

        return {{{MetalDeviceIndex(), "0"},
                 {MetalDeviceName(), deviceName},
                 {MetalPrecision(), "single"}}};
    }
}

void MetalPlatform::contextCreated(ContextImpl& context, const map<string, string>& properties) const {
    @autoreleasepool {
        id<MTLDevice> device = getSupportedDevice();
        if (device == nil)
            throw OpenMMException("The Metal platform requires a supported built-in Apple Silicon GPU");

        string deviceIndex = propertyValue(properties, MetalDeviceIndex(), getPropertyDefaultValue(MetalDeviceIndex()));
        if (deviceIndex.empty())
            deviceIndex = "0";
        if (deviceIndex != "0")
            throw OpenMMException("The Metal platform currently supports only DeviceIndex 0");

        string requestedName = propertyValue(properties, MetalDeviceName(), getPropertyDefaultValue(MetalDeviceName()));
        string actualName = getDeviceName(device);
        if (!requestedName.empty() && requestedName != actualName)
            throw OpenMMException("The requested Metal DeviceName is not available");

        string precision = toLower(propertyValue(properties, MetalPrecision(), getPropertyDefaultValue(MetalPrecision())));
        if (precision.empty())
            precision = "single";
        if (precision != "single")
            throw OpenMMException("The Metal platform currently supports only single precision");

        const System& system = context.getSystem();
        if (system.getNumConstraints() != 0)
            throw OpenMMException("The first native Metal phase does not yet support constraints");
        for (int i = 0; i < system.getNumParticles(); i++)
            if (system.isVirtualSite(i))
                throw OpenMMException("The first native Metal phase does not yet support virtual sites");
        for (int i = 0; i < system.getNumForces(); i++) {
            const HarmonicBondForce* force = dynamic_cast<const HarmonicBondForce*>(&system.getForce(i));
            if (force == NULL)
                throw OpenMMException("The first native Metal phase supports only HarmonicBondForce");
            if (force->usesPeriodicBoundaryConditions())
                throw OpenMMException("The first native Metal phase does not yet support periodic harmonic bonds");
        }

        context.setPlatformData(new PlatformData(system, 0, actualName, precision));
    }
}

void MetalPlatform::contextDestroyed(ContextImpl& context) const {
    delete reinterpret_cast<PlatformData*>(context.getPlatformData());
}
