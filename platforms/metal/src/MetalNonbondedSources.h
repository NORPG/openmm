/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * Ported from OpenMM's OpenCL/CUDA/HIP nonbonded kernels and utilities.
 * Original code: Portions copyright (c) 2009-2025 Stanford University and
 * the Authors. Authors: Peter Eastman and the OpenMM contributors.
 * Metal code: Portions copyright (c) 2026 Chun-Chi Hung.
 * Author: Chun-Chi Hung.
 * This program is free software under the GNU Lesser General Public License,
 * version 3 or (at your option) any later version. It is distributed without
 * any warranty; see <http://www.gnu.org/licenses/> for the license.
 * -------------------------------------------------------------------------- */

#ifndef OPENMM_METALNONBONDEDSOURCES_H_
#define OPENMM_METALNONBONDEDSOURCES_H_

#include "MetalKernelSources.h"
#include <string>

namespace OpenMM {
namespace MetalNonbondedSources {

/** @brief Return Metal's native nonbonded implementation. */
inline std::string nonbondedSource() {
    return MetalKernelSources::nonbonded;
}

/** @brief Supply only definitions for introduced native neighbor features. */
inline std::string neighbors(const std::string& source, bool sparsePairs) {
    return source;
}

} // namespace MetalNonbondedSources
} // namespace OpenMM
#endif
