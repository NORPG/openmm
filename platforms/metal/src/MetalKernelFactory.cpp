#include "MetalKernelFactory.h"
#include "MetalKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/kernels.h"

using namespace OpenMM;
using namespace std;

KernelImpl* MetalKernelFactory::createKernelImpl(string name, const Platform& platform, ContextImpl& context) const {
    if (name == CalcForcesAndEnergyKernel::Name())
        return new MetalCalcForcesAndEnergyKernel(name, platform);
    if (name == UpdateStateDataKernel::Name())
        return new MetalUpdateStateDataKernel(name, platform);
    if (name == ApplyConstraintsKernel::Name())
        return new MetalApplyConstraintsKernel(name, platform);
    if (name == VirtualSitesKernel::Name())
        return new MetalVirtualSitesKernel(name, platform);
    if (name == MinimizeKernel::Name())
        return new MetalMinimizeKernel(name, platform);
    if (name == CalcHarmonicBondForceKernel::Name())
        return new MetalCalcHarmonicBondForceKernel(name, platform, context);
    if (name == CalcNonbondedForceKernel::Name())
        return new MetalCalcNonbondedForceKernel(name, platform, context);
    if (name == IntegrateVerletStepKernel::Name())
        return new MetalIntegrateVerletStepKernel(name, platform, context);
    throw OpenMMException("The Metal kernel '"+name+"' has not been implemented");
}
