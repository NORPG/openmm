#include "MetalPlatform.h"
#include "MetalPlugin.h"

using namespace OpenMM;

extern "C" OPENMM_EXPORT_METAL void registerMetalPlatform() {
    if (MetalPlatform::isPlatformSupported())
        Platform::registerPlatform(new MetalPlatform());
}

#ifndef OPENMM_METAL_BUILDING_STATIC_LIBRARY
extern "C" OPENMM_EXPORT_METAL void registerPlatforms() {
    registerMetalPlatform();
}
#endif
