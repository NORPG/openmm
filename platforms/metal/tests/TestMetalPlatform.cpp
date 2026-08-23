#include "MetalPlatform.h"
#include "MetalPlugin.h"
#include "openmm/Platform.h"
#include "openmm/kernels.h"
#include "openmm/internal/AssertionUtilities.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace OpenMM;
using namespace std;

#ifndef OPENMM_METAL_USE_STATIC_LIBRARIES
extern "C" void registerPlatforms();
#endif

int main() {
    try {
        if (!MetalPlatform::isPlatformSupported()) {
            cout << "Test skipped: no supported Metal device is visible" << endl;
            return 0;
        }

        MetalPlatform platform;
        ASSERT_EQUAL(string("Metal"), platform.getName());
        ASSERT(!platform.supportsDoublePrecision());

        const vector<string>& names = platform.getPropertyNames();
        ASSERT_EQUAL(3, names.size());
        ASSERT_EQUAL(string("DeviceIndex"), names[0]);
        ASSERT_EQUAL(string("DeviceName"), names[1]);
        ASSERT_EQUAL(string("Precision"), names[2]);
        ASSERT_EQUAL(string("0"), platform.getPropertyDefaultValue("DeviceIndex"));
        ASSERT_EQUAL(string("single"), platform.getPropertyDefaultValue("Precision"));

        vector<map<string, string> > devices = platform.getDevices();
        ASSERT_EQUAL(1, devices.size());
        ASSERT_EQUAL(string("0"), devices[0]["DeviceIndex"]);
        ASSERT(!devices[0]["DeviceName"].empty());
        ASSERT_EQUAL(string("single"), devices[0]["Precision"]);
        ASSERT_EQUAL(0, platform.getDevices({{"DeviceIndex", "1"}}).size());
        ASSERT_EQUAL(0, platform.getDevices({{"Precision", "mixed"}}).size());
        ASSERT_EQUAL(0, platform.getDevices({{"Precision", "double"}}).size());

        const vector<string> phaseOneKernels = {
            CalcForcesAndEnergyKernel::Name(),
            UpdateStateDataKernel::Name(),
            ApplyConstraintsKernel::Name(),
            VirtualSitesKernel::Name(),
            MinimizeKernel::Name(),
            CalcHarmonicBondForceKernel::Name(),
            IntegrateVerletStepKernel::Name()
        };
        ASSERT(platform.supportsKernels(phaseOneKernels));
        ASSERT(!platform.supportsKernels({"UnimplementedMetalKernel"}));

#ifdef OPENMM_METAL_USE_STATIC_LIBRARIES
        registerMetalPlatform();
#else
        registerPlatforms();
#endif
        ASSERT_EQUAL(string("Metal"), Platform::getPlatformByName("Metal").getName());
    }
    catch (const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
