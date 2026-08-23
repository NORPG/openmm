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
#include "MetalPlatform.h"
#include "MetalProgram.h"
#include "ReferenceMinimize.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"

#include <cstdint>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace OpenMM;
using namespace std;

namespace {

struct MetalBondIndex {
    uint32_t particle1;
    uint32_t particle2;
};

struct MetalBondParameter {
    float length;
    float k;
};

const char* harmonicBondSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void computeHarmonicBonds(device const float4* positions [[buffer(0)]],
                                 device float4* forces [[buffer(1)]],
                                 device const uint2* bonds [[buffer(2)]],
                                 device const float2* parameters [[buffer(3)]],
                                 device float* energyByParticle [[buffer(4)]],
                                 constant uint& numParticles [[buffer(5)]],
                                 constant uint& numBonds [[buffer(6)]],
                                 constant uint& includeForces [[buffer(7)]],
                                 constant uint& includeEnergy [[buffer(8)]],
                                 uint particle [[thread_position_in_grid]]) {
    if (particle >= numParticles)
        return;

    float3 particleForce = float3(0.0f);
    float particleEnergy = 0.0f;
    for (uint bond = 0; bond < numBonds; bond++) {
        const uint2 atoms = bonds[bond];
        if (particle != atoms.x && particle != atoms.y)
            continue;

        const float3 delta = positions[atoms.y].xyz-positions[atoms.x].xyz;
        const float distance = length(delta);
        const float2 parameter = parameters[bond];
        const float displacement = distance-parameter.x;
        if (includeForces != 0 && distance > 0.0f) {
            const float3 forceOnFirst = (parameter.y*displacement/distance)*delta;
            if (particle == atoms.x)
                particleForce += forceOnFirst;
            if (particle == atoms.y)
                particleForce -= forceOnFirst;
        }
        // Assign each bond's energy to its first endpoint so it is counted once.
        if (includeEnergy != 0 && particle == atoms.x)
            particleEnergy += 0.5f*parameter.y*displacement*displacement;
    }
    if (includeForces != 0)
        forces[particle] += float4(particleForce, 0.0f);
    energyByParticle[particle] = includeEnergy == 0 ? 0.0f : particleEnergy;
}
)METAL";

const char* verletSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void integrateVerlet(device float4* positions [[buffer(0)]],
                            device float4* velocities [[buffer(1)]],
                            device const float4* forces [[buffer(2)]],
                            device const float* inverseMasses [[buffer(3)]],
                            constant float& stepSize [[buffer(4)]],
                            constant uint& numParticles [[buffer(5)]],
                            uint particle [[thread_position_in_grid]]) {
    if (particle >= numParticles)
        return;
    const float inverseMass = inverseMasses[particle];
    if (inverseMass == 0.0f)
        return;
    float3 velocity = velocities[particle].xyz;
    velocity += stepSize*inverseMass*forces[particle].xyz;
    const float3 position = positions[particle].xyz+stepSize*velocity;
    velocities[particle] = float4(velocity, 0.0f);
    positions[particle] = float4(position, 0.0f);
}
)METAL";

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

    impl->program.reset(new MetalProgram(queue, harmonicBondSource));
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
    impl->program.reset(new MetalProgram(queue, verletSource));
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
