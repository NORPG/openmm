#ifndef OPENMM_WINDOWSEXPORTMETAL_H_
#define OPENMM_WINDOWSEXPORTMETAL_H_

/*
 * Keep the Metal plugin's public ABI annotation separate from the core
 * library.  The backend is Apple-only today, but using the same convention as
 * the other OpenMM plugins makes the ownership of exported symbols explicit.
 */
#ifdef _MSC_VER
    #if defined(OPENMM_METAL_BUILDING_SHARED_LIBRARY)
        #define OPENMM_EXPORT_METAL __declspec(dllexport)
    #elif defined(OPENMM_METAL_BUILDING_STATIC_LIBRARY) || defined(OPENMM_METAL_USE_STATIC_LIBRARIES)
        #define OPENMM_EXPORT_METAL
    #else
        #define OPENMM_EXPORT_METAL __declspec(dllimport)
    #endif
#else
    #define OPENMM_EXPORT_METAL
#endif

#endif // OPENMM_WINDOWSEXPORTMETAL_H_
