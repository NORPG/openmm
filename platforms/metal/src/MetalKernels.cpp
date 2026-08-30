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

#include "MetalKernels.h"
#include "MetalArray.h"
#include "MetalContext.h"
#ifndef OPENMM_METAL_USE_EMBEDDED_METALLIB
#include "MetalKernelSources.h"
#endif
#include "MetalPlatform.h"
#include "MetalProgram.h"
#include "ReferenceMinimize.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace OpenMM;
using namespace std;

#ifdef OPENMM_METAL_USE_EMBEDDED_METALLIB
extern "C" const unsigned char openmmMetalMetallibStart[];
extern "C" const unsigned char openmmMetalMetallibEnd[];
#endif

namespace {

struct MetalBondIndex {
    uint32_t particle1;
    uint32_t particle2;
};

struct MetalBondParameter {
    float length;
    float k;
};

struct MetalNonbondedParameter {
    float charge;
    float sigma;
    float epsilon;
    float padding;
};

struct MetalExceptionEntry {
    uint32_t particle;
    uint32_t exception;
};

MetalContext& getMetalContext(ContextImpl& context) {
    return MetalPlatform::getMetalContext(context);
}

MetalContext& getMetalContext(const ContextImpl& context) {
    return MetalPlatform::getMetalContext(const_cast<ContextImpl&>(context));
}

void validateNoVirtualSites(const System& system) {
    for (int i = 0; i < system.getNumParticles(); i++) {
        if (system.isVirtualSite(i))
            throw OpenMMException("The first native Metal backend phase does not support virtual sites");
    }
}

#ifdef OPENMM_METAL_USE_EMBEDDED_METALLIB
unique_ptr<MetalProgram> loadProductionMetalProgram(MetalQueue& queue) {
    const uintptr_t start = reinterpret_cast<uintptr_t>(openmmMetalMetallibStart);
    const uintptr_t end = reinterpret_cast<uintptr_t>(openmmMetalMetallibEnd);
    if (end <= start)
        throw OpenMMException("The embedded OpenMM Metal library is empty");
    return unique_ptr<MetalProgram>(new MetalProgram(
            queue, openmmMetalMetallibStart,
            static_cast<size_t>(end-start)));
}
#endif

} // namespace

MetalCalcForcesAndEnergyKernel::MetalCalcForcesAndEnergyKernel(string name, const Platform& platform) :
        CalcForcesAndEnergyKernel(name, platform) {
}

void MetalCalcForcesAndEnergyKernel::initialize(const System& system) {
    (void) system;
}

void MetalCalcForcesAndEnergyKernel::beginComputation(ContextImpl& context, bool includeForce,
                                                       bool includeEnergy, int groups) {
    (void) includeEnergy;
    (void) groups;
    getMetalContext(context).clearAutoclearBuffers();
    if (includeForce)
        getMetalContext(context).clearForces();
}

double MetalCalcForcesAndEnergyKernel::finishComputation(ContextImpl& context, bool includeForce,
                                                          bool includeEnergy, int groups, bool& valid) {
    (void) context;
    (void) includeForce;
    (void) includeEnergy;
    (void) groups;
    valid = true;
    return 0.0;
}

MetalUpdateStateDataKernel::MetalUpdateStateDataKernel(string name, const Platform& platform) :
        UpdateStateDataKernel(name, platform) {
}

void MetalUpdateStateDataKernel::initialize(const System& system) {
    masses.resize(system.getNumParticles());
    for (int i = 0; i < system.getNumParticles(); i++)
        masses[i] = system.getParticleMass(i);
}

double MetalUpdateStateDataKernel::getTime(const ContextImpl& context) const {
    return getMetalContext(context).getTime();
}

void MetalUpdateStateDataKernel::setTime(ContextImpl& context, double time) {
    getMetalContext(context).setTime(time);
}

long long MetalUpdateStateDataKernel::getStepCount(const ContextImpl& context) const {
    return getMetalContext(context).getStepCount();
}

void MetalUpdateStateDataKernel::setStepCount(const ContextImpl& context, long long count) {
    getMetalContext(context).setStepCount(count);
}

void MetalUpdateStateDataKernel::getPositions(ContextImpl& context, vector<Vec3>& positions, bool allowPeriodic) {
    (void) allowPeriodic;
    getMetalContext(context).getPositions(positions);
}

void MetalUpdateStateDataKernel::setPositions(ContextImpl& context, const vector<Vec3>& positions) {
    getMetalContext(context).setPositions(positions);
}

void MetalUpdateStateDataKernel::getVelocities(ContextImpl& context, vector<Vec3>& velocities) {
    getMetalContext(context).getVelocities(velocities);
}

void MetalUpdateStateDataKernel::setVelocities(ContextImpl& context, const vector<Vec3>& velocities) {
    getMetalContext(context).setVelocities(velocities);
}

void MetalUpdateStateDataKernel::computeShiftedVelocities(ContextImpl& context, double timeShift,
                                                           vector<Vec3>& velocities) {
    vector<Vec3> forces;
    MetalContext& metal = getMetalContext(context);
    metal.getVelocities(velocities);
    metal.getForces(forces);
    if (velocities.size() != masses.size() || forces.size() != masses.size())
        throw OpenMMException("Metal state buffers do not match the System particle count");
    for (size_t i = 0; i < masses.size(); i++) {
        if (masses[i] != 0.0)
            velocities[i] += forces[i]*(timeShift/masses[i]);
    }
}

void MetalUpdateStateDataKernel::getForces(ContextImpl& context, vector<Vec3>& forces) {
    getMetalContext(context).getForces(forces);
}

void MetalUpdateStateDataKernel::getEnergyParameterDerivatives(ContextImpl& context,
                                                                map<string, double>& derivs) {
    derivs.clear();
    for (const auto& parameter : context.getParameters())
        derivs[parameter.first] = 0.0;
}

void MetalUpdateStateDataKernel::getPeriodicBoxVectors(ContextImpl& context, Vec3& a, Vec3& b, Vec3& c) const {
    getMetalContext(context).getPeriodicBoxVectors(a, b, c);
}

void MetalUpdateStateDataKernel::setPeriodicBoxVectors(ContextImpl& context, const Vec3& a,
                                                        const Vec3& b, const Vec3& c) {
    getMetalContext(context).setPeriodicBoxVectors(a, b, c);
}

void MetalUpdateStateDataKernel::createCheckpoint(ContextImpl& context, ostream& stream) {
    getMetalContext(context).createCheckpoint(stream);
}

void MetalUpdateStateDataKernel::loadCheckpoint(ContextImpl& context, istream& stream) {
    getMetalContext(context).loadCheckpoint(stream);
}

MetalApplyConstraintsKernel::MetalApplyConstraintsKernel(string name, const Platform& platform) :
        ApplyConstraintsKernel(name, platform) {
}

void MetalApplyConstraintsKernel::initialize(const System& system) {
    if (system.getNumConstraints() != 0)
        throw OpenMMException("The first native Metal backend phase does not support constraints");
}

void MetalApplyConstraintsKernel::apply(ContextImpl& context, double tol) {
    (void) context;
    (void) tol;
}

void MetalApplyConstraintsKernel::applyToVelocities(ContextImpl& context, double tol) {
    (void) context;
    (void) tol;
}

MetalVirtualSitesKernel::MetalVirtualSitesKernel(string name, const Platform& platform) :
        VirtualSitesKernel(name, platform) {
}

void MetalVirtualSitesKernel::initialize(const System& system) {
    validateNoVirtualSites(system);
}

void MetalVirtualSitesKernel::computePositions(ContextImpl& context) {
    (void) context;
}

MetalMinimizeKernel::MetalMinimizeKernel(string name, const Platform& platform) :
        MinimizeKernel(name, platform) {
}

void MetalMinimizeKernel::initialize(const System& system) {
    (void) system;
}

void MetalMinimizeKernel::execute(ContextImpl& context, double tolerance, int maxIterations,
                                   MinimizationReporter* reporter) {
    ReferenceMinimize::minimize(context, tolerance, maxIterations, reporter);
}

class MetalCalcHarmonicBondForceKernel::Impl {
public:
    explicit Impl(MetalContext& context) : context(&context), numParticles(context.getNumParticles()),
            numBonds(0), initialized(false) {
    }

    MetalContext* context;
    uint32_t numParticles;
    uint32_t numBonds;
    bool initialized;
    vector<MetalBondIndex> hostBonds;
    vector<MetalBondParameter> hostParameters;
    unique_ptr<MetalArray> bonds;
    unique_ptr<MetalArray> parameters;
    unique_ptr<MetalArray> energyByParticle;
    unique_ptr<MetalProgram> program;
    shared_ptr<MetalKernel> kernel;
};

MetalCalcHarmonicBondForceKernel::MetalCalcHarmonicBondForceKernel(string name, const Platform& platform,
                                                                   ContextImpl& context) :
        CalcHarmonicBondForceKernel(name, platform), impl(new Impl(getMetalContext(context))) {
}

MetalCalcHarmonicBondForceKernel::~MetalCalcHarmonicBondForceKernel() {
}

void MetalCalcHarmonicBondForceKernel::initialize(const System& system, const HarmonicBondForce& force) {
    if (impl->initialized)
        throw OpenMMException("The Metal harmonic bond kernel has already been initialized");
    if (force.usesPeriodicBoundaryConditions())
        throw OpenMMException("The first native Metal backend phase does not support periodic harmonic bonds");
    if (system.getNumParticles() != static_cast<int>(impl->numParticles))
        throw OpenMMException("The Metal harmonic bond kernel was created for a different System");

    impl->numBonds = static_cast<uint32_t>(force.getNumBonds());
    impl->hostBonds.resize(impl->numBonds);
    impl->hostParameters.resize(impl->numBonds);
    for (uint32_t i = 0; i < impl->numBonds; i++) {
        int particle1, particle2;
        double length, k;
        force.getBondParameters(i, particle1, particle2, length, k);
        impl->hostBonds[i] = {static_cast<uint32_t>(particle1), static_cast<uint32_t>(particle2)};
        impl->hostParameters[i] = {static_cast<float>(length), static_cast<float>(k)};
    }

    MetalQueue& queue = impl->context->getQueue();
    impl->bonds.reset(new MetalArray(queue, impl->numBonds, sizeof(MetalBondIndex), "Metal harmonic bond indices"));
    impl->parameters.reset(new MetalArray(queue, impl->numBonds, sizeof(MetalBondParameter), "Metal harmonic bond parameters"));
    impl->energyByParticle.reset(new MetalArray(queue, impl->numParticles, sizeof(float), "Metal harmonic bond energy"));
    if (impl->numBonds != 0) {
        impl->bonds->upload(static_cast<const void*>(impl->hostBonds.data()), true);
        impl->parameters->upload(static_cast<const void*>(impl->hostParameters.data()), true);
    }

#ifdef OPENMM_METAL_USE_EMBEDDED_METALLIB
    impl->program = loadProductionMetalProgram(queue);
#else
    impl->program.reset(new MetalProgram(queue, MetalKernelSources::harmonicBond));
#endif
    impl->kernel = impl->program->createMetalKernel("computeHarmonicBonds");
    impl->kernel->addArg(impl->context->getPositions());
    impl->kernel->addArg(impl->context->getForces());
    impl->kernel->addArg(*impl->bonds);
    impl->kernel->addArg(*impl->parameters);
    impl->kernel->addArg(*impl->energyByParticle);
    impl->kernel->addArg(impl->numParticles);
    impl->kernel->addArg(impl->numBonds);
    impl->kernel->addArg(uint32_t(0));
    impl->kernel->addArg(uint32_t(0));
    impl->initialized = true;
}

double MetalCalcHarmonicBondForceKernel::execute(ContextImpl& context, bool includeForces,
                                                  bool includeEnergy) {
    if (!impl->initialized)
        throw OpenMMException("The Metal harmonic bond kernel has not been initialized");
    if (&getMetalContext(context) != impl->context)
        throw OpenMMException("The Metal harmonic bond kernel cannot be used with a different Context");
    if (impl->numParticles == 0)
        return 0.0;

    impl->kernel->setArg(7, static_cast<uint32_t>(includeForces));
    impl->kernel->setArg(8, static_cast<uint32_t>(includeEnergy));
    impl->kernel->execute(static_cast<int>(impl->numParticles));
    if (!includeEnergy)
        return 0.0;

    vector<float> energy(impl->numParticles);
    impl->energyByParticle->download(static_cast<void*>(energy.data()), true);
    double total = 0.0;
    for (float value : energy)
        total += value;
    return total;
}

void MetalCalcHarmonicBondForceKernel::copyParametersToContext(ContextImpl& context,
                                                                const HarmonicBondForce& force,
                                                                int firstBond, int lastBond) {
    if (!impl->initialized)
        throw OpenMMException("The Metal harmonic bond kernel has not been initialized");
    if (&getMetalContext(context) != impl->context)
        throw OpenMMException("The Metal harmonic bond kernel cannot be used with a different Context");
    if (force.usesPeriodicBoundaryConditions())
        throw OpenMMException("The first native Metal backend phase does not support periodic harmonic bonds");
    if (force.getNumBonds() != static_cast<int>(impl->numBonds))
        throw OpenMMException("updateParametersInContext: The number of bonds has changed");
    if (firstBond > lastBond)
        return;
    if (firstBond < 0 || lastBond >= static_cast<int>(impl->numBonds))
        throw OpenMMException("updateParametersInContext: The changed harmonic bond range is invalid");

    for (int i = firstBond; i <= lastBond; i++) {
        int particle1, particle2;
        double length, k;
        force.getBondParameters(i, particle1, particle2, length, k);
        if (particle1 != static_cast<int>(impl->hostBonds[i].particle1) ||
                particle2 != static_cast<int>(impl->hostBonds[i].particle2))
            throw OpenMMException("updateParametersInContext: The set of particles in a bond has changed");
        impl->hostParameters[i] = {static_cast<float>(length), static_cast<float>(k)};
    }
    impl->parameters->uploadSubArray(impl->hostParameters.data()+firstBond, firstBond,
                                     lastBond-firstBond+1, true);
}

class MetalCalcNonbondedForceKernel::Impl {
public:
    explicit Impl(MetalContext& context) : context(&context), numParticles(context.getNumParticles()),
            numExceptions(0), initialized(false) {
    }

    MetalContext* context;
    uint32_t numParticles;
    uint32_t numExceptions;
    bool initialized;
    vector<MetalNonbondedParameter> hostParticleParameters;
    vector<MetalBondIndex> hostExceptionParticles;
    vector<MetalNonbondedParameter> hostExceptionParameters;
    vector<uint32_t> hostExceptionOffsets;
    vector<MetalExceptionEntry> hostExceptionEntries;
    unique_ptr<MetalArray> particleParameters;
    unique_ptr<MetalArray> exceptionOffsets;
    unique_ptr<MetalArray> exceptionEntries;
    unique_ptr<MetalArray> exceptionParameters;
    unique_ptr<MetalArray> energyByParticle;
    unique_ptr<MetalProgram> program;
    shared_ptr<MetalKernel> kernel;
};

MetalCalcNonbondedForceKernel::MetalCalcNonbondedForceKernel(string name, const Platform& platform,
                                                             ContextImpl& context) :
        CalcNonbondedForceKernel(name, platform), impl(new Impl(getMetalContext(context))) {
}

MetalCalcNonbondedForceKernel::~MetalCalcNonbondedForceKernel() {
}

void MetalCalcNonbondedForceKernel::initialize(const System& system, const NonbondedForce& force) {
    if (impl->initialized)
        throw OpenMMException("The Metal nonbonded kernel has already been initialized");
    if (force.getNonbondedMethod() != NonbondedForce::NoCutoff)
        throw OpenMMException("The native Metal backend currently supports only NoCutoff NonbondedForce");
    if (force.getNumParticleParameterOffsets() != 0 || force.getNumExceptionParameterOffsets() != 0)
        throw OpenMMException("The native Metal backend does not yet support NonbondedForce parameter offsets");
    if (system.getNumParticles() != static_cast<int>(impl->numParticles) ||
            force.getNumParticles() != static_cast<int>(impl->numParticles))
        throw OpenMMException("The Metal nonbonded kernel was created for a different System");

    impl->numExceptions = static_cast<uint32_t>(force.getNumExceptions());
    impl->hostParticleParameters.resize(impl->numParticles);
    for (uint32_t i = 0; i < impl->numParticles; i++) {
        double charge, sigma, epsilon;
        force.getParticleParameters(i, charge, sigma, epsilon);
        impl->hostParticleParameters[i] = {static_cast<float>(charge), static_cast<float>(sigma),
                                           static_cast<float>(epsilon), 0.0f};
    }

    impl->hostExceptionParticles.resize(impl->numExceptions);
    impl->hostExceptionParameters.resize(impl->numExceptions);
    vector<vector<MetalExceptionEntry> > adjacency(impl->numParticles);
    for (uint32_t i = 0; i < impl->numExceptions; i++) {
        int particle1, particle2;
        double chargeProd, sigma, epsilon;
        force.getExceptionParameters(i, particle1, particle2, chargeProd, sigma, epsilon);
        const uint32_t first = static_cast<uint32_t>(min(particle1, particle2));
        const uint32_t second = static_cast<uint32_t>(max(particle1, particle2));
        impl->hostExceptionParticles[i] = {first, second};
        impl->hostExceptionParameters[i] = {static_cast<float>(chargeProd), static_cast<float>(sigma),
                                            static_cast<float>(epsilon), 0.0f};
        adjacency[first].push_back({second, i});
        adjacency[second].push_back({first, i});
    }

    impl->hostExceptionOffsets.resize(static_cast<size_t>(impl->numParticles)+1);
    for (uint32_t particle = 0; particle < impl->numParticles; particle++) {
        vector<MetalExceptionEntry>& entries = adjacency[particle];
        sort(entries.begin(), entries.end(), [](const MetalExceptionEntry& first,
                                                const MetalExceptionEntry& second) {
            return first.particle < second.particle;
        });
        impl->hostExceptionOffsets[particle] = static_cast<uint32_t>(impl->hostExceptionEntries.size());
        impl->hostExceptionEntries.insert(impl->hostExceptionEntries.end(), entries.begin(), entries.end());
    }
    impl->hostExceptionOffsets[impl->numParticles] = static_cast<uint32_t>(impl->hostExceptionEntries.size());

    MetalQueue& queue = impl->context->getQueue();
    impl->particleParameters.reset(new MetalArray(queue, impl->numParticles,
            sizeof(MetalNonbondedParameter), "Metal nonbonded particle parameters"));
    impl->exceptionOffsets.reset(new MetalArray(queue, impl->hostExceptionOffsets.size(),
            sizeof(uint32_t), "Metal nonbonded exception offsets"));
    impl->exceptionEntries.reset(new MetalArray(queue, impl->hostExceptionEntries.size(),
            sizeof(MetalExceptionEntry), "Metal nonbonded exception entries"));
    impl->exceptionParameters.reset(new MetalArray(queue, impl->numExceptions,
            sizeof(MetalNonbondedParameter), "Metal nonbonded exception parameters"));
    impl->energyByParticle.reset(new MetalArray(queue, impl->numParticles,
            sizeof(float), "Metal nonbonded energy"));
    if (impl->numParticles != 0)
        impl->particleParameters->upload(impl->hostParticleParameters.data(), true);
    impl->exceptionOffsets->upload(impl->hostExceptionOffsets.data(), true);
    if (impl->numExceptions != 0) {
        impl->exceptionEntries->upload(impl->hostExceptionEntries.data(), true);
        impl->exceptionParameters->upload(impl->hostExceptionParameters.data(), true);
    }

#ifdef OPENMM_METAL_USE_EMBEDDED_METALLIB
    impl->program = loadProductionMetalProgram(queue);
#else
    impl->program.reset(new MetalProgram(queue, MetalKernelSources::nonbonded));
#endif
    impl->kernel = impl->program->createMetalKernel("computeNoCutoffNonbonded");
    impl->kernel->addArg(impl->context->getPositions());
    impl->kernel->addArg(impl->context->getForces());
    impl->kernel->addArg(*impl->particleParameters);
    impl->kernel->addArg(*impl->exceptionOffsets);
    impl->kernel->addArg(*impl->exceptionEntries);
    impl->kernel->addArg(*impl->exceptionParameters);
    impl->kernel->addArg(*impl->energyByParticle);
    impl->kernel->addArg(impl->numParticles);
    impl->kernel->addArg(uint32_t(0));
    impl->kernel->addArg(uint32_t(0));
    impl->initialized = true;
}

double MetalCalcNonbondedForceKernel::execute(ContextImpl& context, bool includeForces,
                                               bool includeEnergy, bool includeDirect,
                                               bool includeReciprocal) {
    (void) includeReciprocal;
    if (!impl->initialized)
        throw OpenMMException("The Metal nonbonded kernel has not been initialized");
    if (&getMetalContext(context) != impl->context)
        throw OpenMMException("The Metal nonbonded kernel cannot be used with a different Context");
    if (!includeDirect || (!includeForces && !includeEnergy) || impl->numParticles == 0)
        return 0.0;

    impl->kernel->setArg(8, static_cast<uint32_t>(includeForces));
    impl->kernel->setArg(9, static_cast<uint32_t>(includeEnergy));
    impl->kernel->execute(static_cast<int>(impl->numParticles));
    if (!includeEnergy)
        return 0.0;

    vector<float> energy(impl->numParticles);
    impl->energyByParticle->download(energy.data(), true);
    double total = 0.0;
    for (float value : energy)
        total += value;
    return total;
}

void MetalCalcNonbondedForceKernel::copyParametersToContext(ContextImpl& context,
                                                             const NonbondedForce& force,
                                                             int firstParticle, int lastParticle,
                                                             int firstException, int lastException) {
    if (!impl->initialized)
        throw OpenMMException("The Metal nonbonded kernel has not been initialized");
    if (&getMetalContext(context) != impl->context)
        throw OpenMMException("The Metal nonbonded kernel cannot be used with a different Context");
    if (force.getNumParticles() != static_cast<int>(impl->numParticles))
        throw OpenMMException("updateParametersInContext: The number of particles has changed");
    if (force.getNumExceptions() != static_cast<int>(impl->numExceptions))
        throw OpenMMException("updateParametersInContext: The number of exceptions has changed");
    if (force.getNumParticleParameterOffsets() != 0 || force.getNumExceptionParameterOffsets() != 0)
        throw OpenMMException("updateParametersInContext: Metal does not support NonbondedForce parameter offsets");

    for (uint32_t i = 0; i < impl->numExceptions; i++) {
        int particle1, particle2;
        double chargeProd, sigma, epsilon;
        force.getExceptionParameters(i, particle1, particle2, chargeProd, sigma, epsilon);
        const uint32_t first = static_cast<uint32_t>(min(particle1, particle2));
        const uint32_t second = static_cast<uint32_t>(max(particle1, particle2));
        if (first != impl->hostExceptionParticles[i].particle1 ||
                second != impl->hostExceptionParticles[i].particle2)
            throw OpenMMException("updateParametersInContext: The set of particles in an exception has changed");
    }

    if (firstParticle <= lastParticle) {
        if (firstParticle < 0 || lastParticle >= static_cast<int>(impl->numParticles))
            throw OpenMMException("updateParametersInContext: The changed nonbonded particle range is invalid");
        for (int i = firstParticle; i <= lastParticle; i++) {
            double charge, sigma, epsilon;
            force.getParticleParameters(i, charge, sigma, epsilon);
            impl->hostParticleParameters[i] = {static_cast<float>(charge), static_cast<float>(sigma),
                                               static_cast<float>(epsilon), 0.0f};
        }
        impl->particleParameters->uploadSubArray(impl->hostParticleParameters.data()+firstParticle,
                firstParticle, lastParticle-firstParticle+1, true);
    }
    if (firstException <= lastException) {
        if (firstException < 0 || lastException >= static_cast<int>(impl->numExceptions))
            throw OpenMMException("updateParametersInContext: The changed nonbonded exception range is invalid");
        for (int i = firstException; i <= lastException; i++) {
            int particle1, particle2;
            double chargeProd, sigma, epsilon;
            force.getExceptionParameters(i, particle1, particle2, chargeProd, sigma, epsilon);
            impl->hostExceptionParameters[i] = {static_cast<float>(chargeProd), static_cast<float>(sigma),
                                                static_cast<float>(epsilon), 0.0f};
        }
        impl->exceptionParameters->uploadSubArray(impl->hostExceptionParameters.data()+firstException,
                firstException, lastException-firstException+1, true);
    }
}

void MetalCalcNonbondedForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    (void) alpha;
    (void) nx;
    (void) ny;
    (void) nz;
    throw OpenMMException("getPMEParametersInContext: This Metal Context is not using PME or LJPME");
}

void MetalCalcNonbondedForceKernel::getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    (void) alpha;
    (void) nx;
    (void) ny;
    (void) nz;
    throw OpenMMException("getLJPMEParametersInContext: This Metal Context is not using LJPME");
}

class MetalIntegrateVerletStepKernel::Impl {
public:
    explicit Impl(MetalContext& context) : context(&context), numParticles(context.getNumParticles()),
            initialized(false) {
    }

    MetalContext* context;
    uint32_t numParticles;
    bool initialized;
    vector<double> masses;
    unique_ptr<MetalProgram> program;
    shared_ptr<MetalKernel> kernel;
};

MetalIntegrateVerletStepKernel::MetalIntegrateVerletStepKernel(string name, const Platform& platform,
                                                               ContextImpl& context) :
        IntegrateVerletStepKernel(name, platform), impl(new Impl(getMetalContext(context))) {
}

MetalIntegrateVerletStepKernel::~MetalIntegrateVerletStepKernel() {
}

void MetalIntegrateVerletStepKernel::initialize(const System& system, const VerletIntegrator& integrator) {
    (void) integrator;
    if (impl->initialized)
        throw OpenMMException("The Metal Verlet kernel has already been initialized");
    if (system.getNumConstraints() != 0)
        throw OpenMMException("The first native Metal backend phase does not support constrained Verlet integration");
    validateNoVirtualSites(system);
    if (system.getNumParticles() != static_cast<int>(impl->numParticles))
        throw OpenMMException("The Metal Verlet kernel was created for a different System");

    impl->masses.resize(impl->numParticles);
    for (uint32_t i = 0; i < impl->numParticles; i++)
        impl->masses[i] = system.getParticleMass(i);

    MetalQueue& queue = impl->context->getQueue();
#ifdef OPENMM_METAL_USE_EMBEDDED_METALLIB
    impl->program = loadProductionMetalProgram(queue);
#else
    impl->program.reset(new MetalProgram(queue, MetalKernelSources::verlet));
#endif
    impl->kernel = impl->program->createMetalKernel("integrateVerlet");
    impl->kernel->addArg(impl->context->getPositions());
    impl->kernel->addArg(impl->context->getVelocities());
    impl->kernel->addArg(impl->context->getForces());
    impl->kernel->addArg(impl->context->getInverseMasses());
    impl->kernel->addArg(float(0.0f));
    impl->kernel->addArg(impl->numParticles);
    impl->initialized = true;
}

void MetalIntegrateVerletStepKernel::execute(ContextImpl& context, const VerletIntegrator& integrator) {
    if (!impl->initialized)
        throw OpenMMException("The Metal Verlet kernel has not been initialized");
    if (&getMetalContext(context) != impl->context)
        throw OpenMMException("The Metal Verlet kernel cannot be used with a different Context");
    const double stepSize = integrator.getStepSize();
    const float metalStepSize = static_cast<float>(stepSize);
    impl->kernel->setArg(4, metalStepSize);
    impl->kernel->execute(static_cast<int>(impl->numParticles));
    impl->context->advanceTime(stepSize);
}

double MetalIntegrateVerletStepKernel::computeKineticEnergy(ContextImpl& context,
                                                             const VerletIntegrator& integrator) {
    if (!impl->initialized)
        throw OpenMMException("The Metal Verlet kernel has not been initialized");
    if (&getMetalContext(context) != impl->context)
        throw OpenMMException("The Metal Verlet kernel cannot be used with a different Context");

    vector<Vec3> velocities;
    vector<Vec3> forces;
    impl->context->getVelocities(velocities);
    impl->context->getForces(forces);
    if (velocities.size() != impl->masses.size() || forces.size() != impl->masses.size())
        throw OpenMMException("Metal state buffers do not match the System particle count");

    const double timeShift = 0.5*integrator.getStepSize();
    double energy = 0.0;
    for (size_t i = 0; i < impl->masses.size(); i++) {
        if (impl->masses[i] != 0.0) {
            Vec3 shifted = velocities[i]+forces[i]*(timeShift/impl->masses[i]);
            energy += 0.5*impl->masses[i]*shifted.dot(shifted);
        }
    }
    return energy;
}
