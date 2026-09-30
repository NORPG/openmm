/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Adapted from Common pairwise kernels used by CUDA, HIP, and OpenCL.        *
 * Original OpenMM code:                                                      *
 * Portions copyright (c) 2008-2026 Stanford University and the Authors.       *
 * Authors: Peter Eastman                                                     *
 * Metal Platform code:                                                       *
 * Portions copyright (c) 2026 Chun-Chi Hung.                                 *
 * Authors: Chun-Chi Hung                                                     *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 * This program is distributed WITHOUT ANY WARRANTY; without even the        *
 * implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. *
 * See the GNU Lesser General Public License for more details.                *
 * You should have received a copy of the GNU Lesser General Public License  *
 * along with this program. If not, see <http://www.gnu.org/licenses/>.       *
 * -------------------------------------------------------------------------- */

#ifndef OPENMM_METALPAIRWISEOPTIMIZATIONS_H_
#define OPENMM_METALPAIRWISEOPTIMIZATIONS_H_

#include <set>
#include <string>

namespace OpenMM {

/**
 * @brief Define-only selection of optional Common pairwise transport paths.
 *
 * Register paths change lane communication, not fixed-point representation,
 * exclusions, or pair enumeration. Every path is written in the shader source;
 * this class only prepends its independent selection defines and reads the
 * marked CustomGB declarations to describe dynamically generated lane fields.
 * Common's generator, formulas, and kernel arguments remain unchanged.
 * Register paths require 32 active lanes per SIMD group, including padded
 * lanes of the last tile. An explicit source-owned metadata declaration names
 * read-only GBSA Born-force parameters without inspecting the expression.
 */
class MetalPairwiseOptimizations {
public:
    /** Independently selectable shader branches; defaults are the Common path. */
    struct Settings {
        bool customGBValue = false;
        bool customGBEnergy = false;
        bool gbsaBorn = false;
        bool gbsaForce = false;
        bool dpdParticles = false;
        bool dpdTile = false;
        bool customHbond = false;
    };

    /** @return Settings selected by the independent build-time switches. */
    static Settings getBuildSettings();
    /** @brief Prepend build-time selection defines, leaving the kernel body intact. */
    static std::string apply(const std::string& source);
    /** @brief Explicit settings support focused tests of individual paths. */
    static std::string apply(const std::string& source, const Settings& settings);
    /**
     * @brief Read the source-owned GBSA Born-force parameter metadata declaration.
     *
     * Only the OPENMM_METAL_GBSA_BORN_FORCE_PARAMETERS declaration is parsed.
     * Missing, duplicate, or invalid declarations return an empty set. A modified
     * interaction must maintain its explicit read-only real-arithmetic contract.
     * The returned names include the particle suffix (for example, bornForce1).
     */
    static std::set<std::string> getGBSAChainRuleBornForceParameters(const std::string& source);

};

} // namespace OpenMM
#endif
