#ifndef OPENMM_METALCAPABILITYPROBEINTERNAL_H_
#define OPENMM_METALCAPABILITYPROBEINTERNAL_H_

#include "MetalCapabilityProbe.h"
#include <mutex>
#include <string>

namespace OpenMM {
class MetalQueue;
namespace detail {

/** Shared by an owning queue and its siblings, but not independent queues. */
struct MetalCapabilityProbeCache {
    std::once_flag splitFixedPointOnce;
    MetalCapabilityProbeResult splitFixedPoint;
};

std::string getSplitFixedPointProbeSource();

// Internal runner also permits deterministic compile/pipeline/dispatch/result
// failure tests.  Call only with a dedicated queue; never with simulation data.
MetalCapabilityProbeResult runSplitFixedPointProbe(MetalQueue& queue,
        const std::string& source, int blockSize = -1);

} // namespace detail
} // namespace OpenMM

#endif /*OPENMM_METALCAPABILITYPROBEINTERNAL_H_*/
