/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 *                                                                            *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Ported from the OpenMM CUDA Platform.                                      *
 * Source: platforms/cuda/include/CudaQueue.h                                 *
 *                                                                            *
 * Original CUDA Platform code:                                               *
 * Portions copyright (c) 2025 Stanford University and the Authors.           *
 * Authors: Peter Eastman                                                     *
 *                                                                            *
 * Metal Platform code:                                                       *
 * Portions copyright (c) 2026 Chun-Chi Hung.                                 *
 * Authors: Chun-Chi Hung                                                     *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the               *
 * GNU Lesser General Public License for more details.                        *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public License   *
 * along with this program. If not, see <http://www.gnu.org/licenses/>.       *
 * -------------------------------------------------------------------------- */

#ifndef OPENMM_METALQUEUE_H_
#define OPENMM_METALQUEUE_H_

#include "openmm/common/ComputeQueue.h"
#include <memory>

namespace OpenMM {

/**
 * @brief Metal 4 queue with explicit ordering, resource lifetime, and error reporting.
 *
 * Adapted from the CUDA/HIP queue interface. Native types remain private so this
 * header can be included from ordinary C++11 code.
 * @warning Host encoding/submission calls must be serialized by the caller.
 */
class MetalQueue : public ComputeQueueImpl {
public:
    /** @brief Opaque ownership handle for one Metal 4 submission, defined privately. */
    struct Command;
    /**
     * @brief Create a native Metal 4 queue and its marker buffer.
     * @param device Borrowed, non-null native MTLDevice handle.
     * @throws OpenMMException If the device is null or resource creation fails.
     */
    explicit MetalQueue(void* device);
    /**
     * @brief Wait for tracked submissions before releasing native resources.
     * @note Destruction does not report GPU errors; use wait() or finish() explicitly.
     */
    ~MetalQueue();
    /** @return A borrowed native MTL4CommandQueue handle; do not release it. */
    void* getQueue() const;
    /**
     * @return A new, recording command with its own allocator and residency set.
     * @throws OpenMMException If resource creation fails.
     * @note The caller ends its encoder before passing this handle to submit().
     */
    std::shared_ptr<Command> createCommand();
    /**
     * @return An unsubmitted one-byte GPU fill marker, ready for submit().
     * @note A real GPU operation makes completion observable after queue-level event waits.
     * @throws OpenMMException If command creation or encoding fails.
     */
    std::shared_ptr<Command> createMarker();
    /**
     * @brief End and commit a command, retaining its resources through completion.
     * @param command A recording command created by this queue, with no open encoder.
     * @throws OpenMMException If null, foreign, already submitted, or a reaped command failed.
     * @note Does not wait for execution. Reaping an earlier error prevents this submission.
     */
    void submit(const std::shared_ptr<Command>& command);
    /**
     * @brief Wait for all currently tracked submissions, including event-wait markers.
     * @throws OpenMMException If GPU execution fails.
     */
    void finish();
    /**
     * @brief Wait through a submitted marker and report preceding tracked execution errors.
     * @param command A submitted command from this queue; it may already have been reaped.
     * @throws OpenMMException If null, foreign, unsubmitted, or GPU execution failed.
     * @note Drains through the marker before reporting the first error. Later work is not
     *       waited for; previously reaped errors are not retained, except in the marker itself.
     */
    void wait(const std::shared_ptr<Command>& command);
private:
    class Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace OpenMM

#endif /*OPENMM_METALQUEUE_H_*/
