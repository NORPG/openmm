/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 *                                                                            *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
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

#ifndef OPENMM_METALCOMMAND_H_
#define OPENMM_METALCOMMAND_H_

#include "MetalQueue.h"
#import <Metal/Metal.h>
#include <condition_variable>
#include <mutex>

namespace OpenMM {

/**
 * @brief Private, per-submission Metal 4 storage and completion state.
 *
 * Metal 4 does not retain encoded resources. Keep the allocator, residency set,
 * buffers, and other referenced objects alive until commit feedback arrives.
 * Host encoding is serialized; only completion fields are shared with the callback.
 */
struct MetalQueue::Command {
    /**
     * @brief Allocate and begin one independently owned command buffer.
     * @param owner Queue that will submit and track this command.
     * @param device Native device used to allocate command storage and residency.
     */
    Command(MetalQueue* owner, id<MTLDevice> device);
    /** @return A compute encoder ordered after preceding dispatches and copies on the queue. */
    id<MTL4ComputeCommandEncoder> beginCompute();
    /**
     * @brief Retain a buffer and include it in this submission's residency set.
     * @param buffer Native buffer referenced by an encoded operation.
     */
    void useBuffer(id<MTLBuffer> buffer);
    /** @brief Block until this submission's feedback callback reports completion. */
    void waitUntilCompleted();
    /** @return Whether the feedback callback has completed. */
    bool isComplete();

    MetalQueue* owner;
    id<MTL4CommandBuffer> buffer = nil;
    id<MTL4CommandAllocator> allocator = nil;
    id<MTLResidencySet> residency = nil;
    NSMutableArray* resources = nil;
    // Queue-level event operations are submitted together with their tracked marker.
    id<MTLEvent> signalEvent = nil;
    id<MTLEvent> waitEvent = nil;
    bool submitted = false;
    std::mutex mutex;
    std::condition_variable condition;
    bool completed = false;
    NSError* error = nil;
};

} // namespace OpenMM

#endif
