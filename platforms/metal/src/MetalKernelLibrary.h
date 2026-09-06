#ifndef OPENMM_METALKERNELLIBRARY_H_
#define OPENMM_METALKERNELLIBRARY_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * Portions copyright (c) 2026 Stanford University and the Authors.           *
 * -------------------------------------------------------------------------- */

#ifdef OPENMM_METAL_USE_EMBEDDED_METALLIB
#include "MetalProgram.h"
#include "openmm/OpenMMException.h"
#include <cstdint>
#include <memory>

extern "C" const unsigned char openmmMetalMetallibStart[];
extern "C" const unsigned char openmmMetalMetallibEnd[];

namespace OpenMM {

/** Load the shared offline library on the queue owning the production buffers. */
inline std::unique_ptr<MetalProgram> loadProductionMetalProgram(MetalQueue& queue) {
    const uintptr_t start = reinterpret_cast<uintptr_t>(openmmMetalMetallibStart);
    const uintptr_t end = reinterpret_cast<uintptr_t>(openmmMetalMetallibEnd);
    if (end <= start)
        throw OpenMMException("The embedded OpenMM Metal library is empty");
    return std::unique_ptr<MetalProgram>(new MetalProgram(
            queue, openmmMetalMetallibStart, static_cast<size_t>(end-start)));
}

} // namespace OpenMM
#endif

#endif /*OPENMM_METALKERNELLIBRARY_H_*/
