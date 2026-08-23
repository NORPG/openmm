#ifndef OPENMM_METALEVENT_H_
#define OPENMM_METALEVENT_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2026 Stanford University and the Authors.           *
 * -------------------------------------------------------------------------- */

#include "MetalQueue.h"
#include "openmm/common/ComputeEvent.h"
#include <memory>

namespace OpenMM {

/** A Metal shared-event based synchronization point. */
class OPENMM_EXPORT_METAL MetalEvent : public ComputeEventImpl {
public:
    explicit MetalEvent(MetalQueue& queue);
    ~MetalEvent();

    void enqueue() override;
    void wait() override;
    void queueWait(ComputeQueue queue) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace OpenMM

#endif /*OPENMM_METALEVENT_H_*/
