#ifndef OPENMM_METALQUEUE_H_
#define OPENMM_METALQUEUE_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2026 Stanford University and the Authors.           *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 * -------------------------------------------------------------------------- */

#include "MetalDeviceCaps.h"
#include "openmm/common/ComputeQueue.h"
#include <cstddef>
#include <memory>

namespace OpenMM {

namespace detail {
class MetalQueueState;
}

/**
 * A serial Metal command queue.  Objects created from it retain the underlying
 * queue and device state, so they remain valid if this wrapper is destroyed.
 */
class OPENMM_EXPORT_METAL MetalQueue : public ComputeQueueImpl {
public:
    /** Create a queue for the device at the given MTLCopyAllDevices() index. */
    explicit MetalQueue(size_t deviceIndex = 0);
    ~MetalQueue();

    MetalQueue(const MetalQueue&) = delete;
    MetalQueue& operator=(const MetalQueue&) = delete;

    /** Create another command queue on the same Metal device. */
    std::shared_ptr<MetalQueue> createSiblingQueue() const;

    /** Wait for all previously submitted GPU and asynchronous host work. */
    void waitUntilIdle();

    /** Throw any asynchronous command error observed since the last check. */
    void checkForErrors();

    const MetalDeviceCaps& getDeviceCaps() const;

private:
    explicit MetalQueue(const std::shared_ptr<detail::MetalQueueState>& parent);
    std::shared_ptr<detail::MetalQueueState> state;

    friend class MetalArray;
    friend class MetalEvent;
    friend class MetalKernel;
    friend class MetalProgram;
};

} // namespace OpenMM

#endif /*OPENMM_METALQUEUE_H_*/
