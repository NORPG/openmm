#ifndef OPENMM_METALKERNELFACTORY_H_
#define OPENMM_METALKERNELFACTORY_H_

#include "openmm/KernelFactory.h"

namespace OpenMM {

/**
 * Factory entry point for native Metal kernels.
 *
 * Phase one registers only the seven kernels needed for the documented
 * HarmonicBondForce plus VerletIntegrator vertical slice.
 */
class MetalKernelFactory : public KernelFactory {
public:
    KernelImpl* createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const;
};

} // namespace OpenMM

#endif // OPENMM_METALKERNELFACTORY_H_
