#ifndef OPENMM_METALPLUGIN_H_
#define OPENMM_METALPLUGIN_H_

#include "windowsExportMetal.h"

/**
 * Register the statically linked Metal platform, if the current machine is
 * supported.  Shared plugins are registered through OpenMM's conventional
 * registerPlatforms() entry point instead.
 */
extern "C" OPENMM_EXPORT_METAL void registerMetalPlatform();

#endif // OPENMM_METALPLUGIN_H_
