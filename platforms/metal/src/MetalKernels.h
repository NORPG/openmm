#ifndef OPENMM_METALKERNELS_H_
#define OPENMM_METALKERNELS_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2026 Stanford University and the Authors.           *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the       *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "openmm/kernels.h"
#include <iosfwd>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace OpenMM {

/** Clear and finalize the force buffers used by the first native Metal slice. */
class MetalCalcForcesAndEnergyKernel : public CalcForcesAndEnergyKernel {
public:
    MetalCalcForcesAndEnergyKernel(std::string name, const Platform& platform);
    void initialize(const System& system) override;
    void beginComputation(ContextImpl& context, bool includeForce, bool includeEnergy, int groups) override;
    double finishComputation(ContextImpl& context, bool includeForce, bool includeEnergy, int groups, bool& valid) override;
};

/** Read and write the explicit state buffers owned by MetalContext. */
class MetalUpdateStateDataKernel : public UpdateStateDataKernel {
public:
    MetalUpdateStateDataKernel(std::string name, const Platform& platform);
    void initialize(const System& system) override;
    double getTime(const ContextImpl& context) const override;
    void setTime(ContextImpl& context, double time) override;
    long long getStepCount(const ContextImpl& context) const override;
    void setStepCount(const ContextImpl& context, long long count) override;
    void getPositions(ContextImpl& context, std::vector<Vec3>& positions, bool allowPeriodic = false) override;
    void setPositions(ContextImpl& context, const std::vector<Vec3>& positions) override;
    void getVelocities(ContextImpl& context, std::vector<Vec3>& velocities) override;
    void setVelocities(ContextImpl& context, const std::vector<Vec3>& velocities) override;
    void computeShiftedVelocities(ContextImpl& context, double timeShift, std::vector<Vec3>& velocities) override;
    void getForces(ContextImpl& context, std::vector<Vec3>& forces) override;
    void getEnergyParameterDerivatives(ContextImpl& context, std::map<std::string, double>& derivs) override;
    void getPeriodicBoxVectors(ContextImpl& context, Vec3& a, Vec3& b, Vec3& c) const override;
    void setPeriodicBoxVectors(ContextImpl& context, const Vec3& a, const Vec3& b, const Vec3& c) override;
    void createCheckpoint(ContextImpl& context, std::ostream& stream) override;
    void loadCheckpoint(ContextImpl& context, std::istream& stream) override;

private:
    std::vector<double> masses;
};

/** Constraints are deliberately unsupported in phase one; this is a checked no-op. */
class MetalApplyConstraintsKernel : public ApplyConstraintsKernel {
public:
    MetalApplyConstraintsKernel(std::string name, const Platform& platform);
    void initialize(const System& system) override;
    void apply(ContextImpl& context, double tol) override;
    void applyToVelocities(ContextImpl& context, double tol) override;
};

/** Virtual sites are deliberately unsupported in phase one; this is a checked no-op. */
class MetalVirtualSitesKernel : public VirtualSitesKernel {
public:
    MetalVirtualSitesKernel(std::string name, const Platform& platform);
    void initialize(const System& system) override;
    void computePositions(ContextImpl& context) override;
};

/** Use OpenMM's platform-independent minimizer with Metal force evaluations. */
class MetalMinimizeKernel : public MinimizeKernel {
public:
    MetalMinimizeKernel(std::string name, const Platform& platform);
    void initialize(const System& system) override;
    void execute(ContextImpl& context, double tolerance, int maxIterations, MinimizationReporter* reporter) override;
};

/** Native MSL implementation of nonperiodic HarmonicBondForce. */
class MetalCalcHarmonicBondForceKernel : public CalcHarmonicBondForceKernel {
public:
    MetalCalcHarmonicBondForceKernel(std::string name, const Platform& platform, ContextImpl& context);
    ~MetalCalcHarmonicBondForceKernel() override;
    void initialize(const System& system, const HarmonicBondForce& force) override;
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy) override;
    void copyParametersToContext(ContextImpl& context, const HarmonicBondForce& force,
                                 int firstBond, int lastBond) override;

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};

/** Swift-owned native MSL implementation of NoCutoff Coulomb and Lennard-Jones interactions. */
class MetalCalcNonbondedForceKernel : public CalcNonbondedForceKernel {
public:
    MetalCalcNonbondedForceKernel(std::string name, const Platform& platform, ContextImpl& context);
    ~MetalCalcNonbondedForceKernel() override;
    void initialize(const System& system, const NonbondedForce& force) override;
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy,
                   bool includeDirect, bool includeReciprocal) override;
    void copyParametersToContext(ContextImpl& context, const NonbondedForce& force,
                                 int firstParticle, int lastParticle,
                                 int firstException, int lastException) override;
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const override;
    void getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const override;

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};

/** Native MSL implementation of unconstrained leapfrog Verlet integration. */
class MetalIntegrateVerletStepKernel : public IntegrateVerletStepKernel {
public:
    MetalIntegrateVerletStepKernel(std::string name, const Platform& platform, ContextImpl& context);
    ~MetalIntegrateVerletStepKernel() override;
    void initialize(const System& system, const VerletIntegrator& integrator) override;
    void execute(ContextImpl& context, const VerletIntegrator& integrator) override;
    double computeKineticEnergy(ContextImpl& context, const VerletIntegrator& integrator) override;

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace OpenMM

#endif /*OPENMM_METALKERNELS_H_*/
