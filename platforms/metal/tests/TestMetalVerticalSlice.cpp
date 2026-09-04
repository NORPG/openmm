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
 * and/or sell copies of the Software, and to permit persons to whom the      *
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

#include "MetalPlatform.h"
#include "ReferencePlatform.h"
#include "openmm/Context.h"
#include "openmm/HarmonicAngleForce.h"
#include "openmm/HarmonicBondForce.h"
#include "openmm/LangevinMiddleIntegrator.h"
#include "openmm/LocalEnergyMinimizer.h"
#include "openmm/State.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/VirtualSite.h"
#include "openmm/internal/AssertionUtilities.h"

#include <cmath>
#include <exception>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace OpenMM;
using namespace std;

namespace {

const double FORCE_TOL = 2e-5;
const double STATE_TOL = 2e-6;
const double TRAJECTORY_TOL = 2e-4;

/**
 * Verlet normally requests forces while querying energy because its kinetic
 * energy is evaluated at a half-step.  This test-only subclass lets us
 * exercise the backend's genuine energy-only (includeForces=false) path after
 * first populating a valid force buffer.
 */
class EnergyOnlyVerletIntegrator : public VerletIntegrator {
public:
    explicit EnergyOnlyVerletIntegrator(double stepSize) : VerletIntegrator(stepSize) {
    }
    bool kineticEnergyRequiresForce() const override {
        return false;
    }
};

HarmonicBondForce* addBond(System& system, int particle1, int particle2,
                           double length=1.0, double k=100.0) {
    HarmonicBondForce* force = new HarmonicBondForce();
    force->addBond(particle1, particle2, length, k);
    system.addForce(force);
    return force;
}

void assertContextRejected(System& system, MetalPlatform& platform,
                           const map<string, string>& properties={}) {
    VerletIntegrator integrator(0.001);
    bool rejected = false;
    try {
        Context context(system, integrator, platform, properties);
    }
    catch (const exception&) {
        rejected = true;
    }
    ASSERT(rejected);
}

void testHarmonicBondAndVerlet(MetalPlatform& platform) {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    addBond(system, 0, 1);

    const double stepSize = 0.001;
    VerletIntegrator integrator(stepSize);
    Context context(system, integrator, platform);
    vector<Vec3> positions = {Vec3(0, 0, 0), Vec3(1.1, 0, 0)};
    vector<Vec3> velocities(2, Vec3(0, 0, 0));
    context.setPositions(positions);
    context.setVelocities(velocities);

    // The bond is stretched by 0.1 nm.  For k=100 this gives E=0.5 and
    // equal and opposite forces with magnitude 10.
    State initial = context.getState(State::Forces | State::Energy);
    ASSERT_EQUAL_TOL(0.5, initial.getPotentialEnergy(), FORCE_TOL);
    ASSERT_EQUAL_VEC(Vec3(10, 0, 0), initial.getForces()[0], FORCE_TOL);
    ASSERT_EQUAL_VEC(Vec3(-10, 0, 0), initial.getForces()[1], FORCE_TOL);

    integrator.step(1);
    State afterStep = context.getState(State::Positions | State::Velocities);

    // This is the same kick-then-drift update used by Reference Verlet for an
    // unconstrained system: v <- v+dt*f/m, x <- x+dt*v.
    ASSERT_EQUAL_VEC(Vec3(0.00001, 0, 0), afterStep.getPositions()[0], STATE_TOL);
    ASSERT_EQUAL_VEC(Vec3(1.09999, 0, 0), afterStep.getPositions()[1], STATE_TOL);
    ASSERT_EQUAL_VEC(Vec3(0.01, 0, 0), afterStep.getVelocities()[0], STATE_TOL);
    ASSERT_EQUAL_VEC(Vec3(-0.01, 0, 0), afterStep.getVelocities()[1], STATE_TOL);
    ASSERT_EQUAL_TOL(stepSize, afterStep.getTime(), STATE_TOL);
    ASSERT_EQUAL(1, afterStep.getStepCount());
}

void testMultipleBondsWithSharedParticle(MetalPlatform& platform) {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    system.addParticle(1.0);
    HarmonicBondForce* force = new HarmonicBondForce();
    force->addBond(0, 1, 1.0, 100.0);
    force->addBond(1, 2, 1.0, 40.0);
    system.addForce(force);

    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions({Vec3(0, 0, 0), Vec3(1.1, 0, 0), Vec3(2.3, 0, 0)});

    State state = context.getState(State::Forces | State::Energy);
    ASSERT_EQUAL_TOL(1.3, state.getPotentialEnergy(), FORCE_TOL);
    ASSERT_EQUAL_VEC(Vec3(10, 0, 0), state.getForces()[0], FORCE_TOL);
    ASSERT_EQUAL_VEC(Vec3(-2, 0, 0), state.getForces()[1], FORCE_TOL);
    ASSERT_EQUAL_VEC(Vec3(-8, 0, 0), state.getForces()[2], FORCE_TOL);
}

void testMultipleForceObjectsAndThreadgroups(MetalPlatform& platform) {
    const int numParticles = 257;
    System system;
    for (int i = 0; i < numParticles; i++)
        system.addParticle(1.0);
    HarmonicBondForce* chain = new HarmonicBondForce();
    for (int i = 0; i+1 < numParticles; i++)
        chain->addBond(i, i+1, 1.0, 100.0);
    system.addForce(chain);

    // A second Force object exercises ordered accumulation by separately
    // created Metal force kernels.
    HarmonicBondForce* extra = new HarmonicBondForce();
    extra->addBond(0, 1, 1.0, 50.0);
    system.addForce(extra);

    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    vector<Vec3> positions(numParticles);
    for (int i = 0; i < numParticles; i++)
        positions[i] = Vec3(1.01*i, 0, 0);
    context.setPositions(positions);

    State state = context.getState(State::Forces | State::Energy);
    const Vec3 zero;
    ASSERT_EQUAL_TOL(1.2825, state.getPotentialEnergy(), 2.0e-3);
    ASSERT_EQUAL_VEC(Vec3(1.5, 0, 0), state.getForces()[0], 2.0e-3);
    ASSERT_EQUAL_VEC(Vec3(-0.5, 0, 0), state.getForces()[1], 2.0e-3);
    ASSERT_EQUAL_VEC(zero, state.getForces()[128], 2.0e-3);
    ASSERT_EQUAL_VEC(Vec3(-1.0, 0, 0), state.getForces()[256], 2.0e-3);
}

void testTrajectoryAgainstReference(MetalPlatform& platform) {
    System system;
    system.addParticle(1.0);
    system.addParticle(2.0);
    addBond(system, 0, 1, 1.0, 80.0);

    const double stepSize = 0.0005;
    ReferencePlatform referencePlatform;
    VerletIntegrator referenceIntegrator(stepSize);
    VerletIntegrator metalIntegrator(stepSize);
    Context referenceContext(system, referenceIntegrator, referencePlatform);
    Context metalContext(system, metalIntegrator, platform);
    vector<Vec3> positions = {Vec3(0.05, -0.02, 0.01), Vec3(1.12, 0.06, -0.03)};
    vector<Vec3> velocities = {Vec3(0.03, -0.02, 0.01), Vec3(-0.01, 0.04, -0.02)};
    referenceContext.setPositions(positions);
    referenceContext.setVelocities(velocities);
    metalContext.setPositions(positions);
    metalContext.setVelocities(velocities);

    for (int step = 1; step <= 20; step++) {
        referenceIntegrator.step(1);
        metalIntegrator.step(1);
        State expected = referenceContext.getState(State::Positions | State::Velocities | State::Energy);
        State found = metalContext.getState(State::Positions | State::Velocities | State::Energy);
        ASSERT_EQUAL(expected.getPositions().size(), found.getPositions().size());
        ASSERT_EQUAL(expected.getVelocities().size(), found.getVelocities().size());
        for (int i = 0; i < static_cast<int>(expected.getPositions().size()); i++) {
            ASSERT_EQUAL_VEC(expected.getPositions()[i], found.getPositions()[i], TRAJECTORY_TOL);
            ASSERT_EQUAL_VEC(expected.getVelocities()[i], found.getVelocities()[i], TRAJECTORY_TOL);
        }
        ASSERT_EQUAL_TOL(expected.getPotentialEnergy(), found.getPotentialEnergy(), TRAJECTORY_TOL);
        ASSERT_EQUAL_TOL(expected.getKineticEnergy(), found.getKineticEnergy(), TRAJECTORY_TOL);
        ASSERT_EQUAL_TOL(expected.getTime(), found.getTime(), TRAJECTORY_TOL);
        ASSERT_EQUAL(expected.getStepCount(), found.getStepCount());
        ASSERT_EQUAL(step, found.getStepCount());
    }
}

void testUpdateParametersInContext(MetalPlatform& platform) {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    HarmonicBondForce* force = addBond(system, 0, 1, 1.0, 100.0);

    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions({Vec3(0, 0, 0), Vec3(1.1, 0, 0)});
    State before = context.getState(State::Forces | State::Energy);
    ASSERT_EQUAL_TOL(0.5, before.getPotentialEnergy(), FORCE_TOL);
    ASSERT_EQUAL_VEC(Vec3(10, 0, 0), before.getForces()[0], FORCE_TOL);

    force->setBondParameters(0, 0, 1, 0.95, 40.0);
    force->updateParametersInContext(context);
    State after = context.getState(State::Forces | State::Energy);
    ASSERT_EQUAL_TOL(0.45, after.getPotentialEnergy(), FORCE_TOL);
    ASSERT_EQUAL_VEC(Vec3(6, 0, 0), after.getForces()[0], FORCE_TOL);
    ASSERT_EQUAL_VEC(Vec3(-6, 0, 0), after.getForces()[1], FORCE_TOL);
}

void testEnergyOnlyPreservesForces(MetalPlatform& platform) {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    addBond(system, 0, 1);

    const double stepSize = 0.01;
    EnergyOnlyVerletIntegrator integrator(stepSize);
    Context context(system, integrator, platform);
    context.setPositions({Vec3(0, 0, 0), Vec3(1.1, 0, 0)});
    context.setVelocities({Vec3(0, 0, 0), Vec3(0, 0, 0)});

    State forcesBefore = context.getState(State::Forces);
    State energyOnly = context.getState(State::Energy);
    ASSERT_EQUAL_TOL(0.5, energyOnly.getPotentialEnergy(), FORCE_TOL);
    // Each particle has shifted speed 0.5*dt*10 = 0.05, so the total
    // half-step kinetic energy is 2*(m*v^2/2) = 0.0025.
    ASSERT_EQUAL_TOL(0.0025, energyOnly.getKineticEnergy(), FORCE_TOL);

    State forcesAfter = context.getState(State::Forces);
    for (int i = 0; i < 2; i++)
        ASSERT_EQUAL_VEC(forcesBefore.getForces()[i], forcesAfter.getForces()[i], FORCE_TOL);
}

void testCheckpointRoundTrip(MetalPlatform& platform) {
    System system;
    system.addParticle(1.0);
    system.addParticle(2.0);
    addBond(system, 0, 1, 1.0, 75.0);

    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions({Vec3(0.1, 0.2, -0.1), Vec3(1.2, -0.1, 0.05)});
    context.setVelocities({Vec3(0.02, -0.01, 0.03), Vec3(-0.04, 0.01, -0.02)});
    context.setPeriodicBoxVectors(Vec3(3, 0, 0), Vec3(0, 3, 0), Vec3(0, 0, 3));
    integrator.step(5);
    State saved = context.getState(State::Positions | State::Velocities);

    stringstream checkpoint(ios_base::in | ios_base::out | ios_base::binary);
    context.createCheckpoint(checkpoint);
    ASSERT(!checkpoint.str().empty());

    integrator.step(4);
    State continued = context.getState(State::Positions | State::Velocities);
    context.setPositions({Vec3(9, 8, 7), Vec3(6, 5, 4)});
    context.setVelocities({Vec3(-3, -2, -1), Vec3(1, 2, 3)});
    context.setTime(12.0);
    context.setStepCount(99);
    context.setPeriodicBoxVectors(Vec3(4, 0, 0), Vec3(0, 4, 0), Vec3(0, 0, 4));

    checkpoint.clear();
    checkpoint.seekg(0);
    context.loadCheckpoint(checkpoint);
    State restored = context.getState(State::Positions | State::Velocities);
    for (int i = 0; i < 2; i++) {
        ASSERT_EQUAL_VEC(saved.getPositions()[i], restored.getPositions()[i], STATE_TOL);
        ASSERT_EQUAL_VEC(saved.getVelocities()[i], restored.getVelocities()[i], STATE_TOL);
    }
    ASSERT_EQUAL_TOL(saved.getTime(), restored.getTime(), STATE_TOL);
    ASSERT_EQUAL(saved.getStepCount(), restored.getStepCount());
    Vec3 a, b, c;
    restored.getPeriodicBoxVectors(a, b, c);
    ASSERT_EQUAL_VEC(Vec3(3, 0, 0), a, STATE_TOL);
    ASSERT_EQUAL_VEC(Vec3(0, 3, 0), b, STATE_TOL);
    ASSERT_EQUAL_VEC(Vec3(0, 0, 3), c, STATE_TOL);

    integrator.step(4);
    State replayed = context.getState(State::Positions | State::Velocities);
    for (int i = 0; i < 2; i++) {
        ASSERT_EQUAL_VEC(continued.getPositions()[i], replayed.getPositions()[i], STATE_TOL);
        ASSERT_EQUAL_VEC(continued.getVelocities()[i], replayed.getVelocities()[i], STATE_TOL);
    }
    ASSERT_EQUAL_TOL(continued.getTime(), replayed.getTime(), STATE_TOL);
    ASSERT_EQUAL(continued.getStepCount(), replayed.getStepCount());
}

void testZeroBondForce(MetalPlatform& platform) {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    system.addForce(new HarmonicBondForce());

    const double stepSize = 0.002;
    VerletIntegrator integrator(stepSize);
    Context context(system, integrator, platform);
    context.setPositions({Vec3(0, 0, 0), Vec3(1, 0, 0)});
    context.setVelocities({Vec3(0.1, 0, 0), Vec3(-0.2, 0, 0)});
    State initial = context.getState(State::Forces | State::Energy);
    ASSERT_EQUAL_TOL(0.0, initial.getPotentialEnergy(), FORCE_TOL);
    ASSERT_EQUAL_VEC(Vec3(0, 0, 0), initial.getForces()[0], FORCE_TOL);
    ASSERT_EQUAL_VEC(Vec3(0, 0, 0), initial.getForces()[1], FORCE_TOL);

    integrator.step(3);
    State state = context.getState(State::Positions | State::Velocities | State::Energy);
    ASSERT_EQUAL_VEC(Vec3(0.0006, 0, 0), state.getPositions()[0], STATE_TOL);
    ASSERT_EQUAL_VEC(Vec3(0.9988, 0, 0), state.getPositions()[1], STATE_TOL);
    ASSERT_EQUAL_VEC(Vec3(0.1, 0, 0), state.getVelocities()[0], STATE_TOL);
    ASSERT_EQUAL_VEC(Vec3(-0.2, 0, 0), state.getVelocities()[1], STATE_TOL);
    ASSERT_EQUAL_TOL(0.025, state.getKineticEnergy(), FORCE_TOL);
    ASSERT_EQUAL_TOL(3*stepSize, state.getTime(), STATE_TOL);
    ASSERT_EQUAL(3, state.getStepCount());
}

void testMinimization(MetalPlatform& platform) {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    addBond(system, 0, 1, 1.0, 100.0);

    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions({Vec3(0, 0, 0), Vec3(1.2, 0, 0)});
    LocalEnergyMinimizer::minimize(context, 1.0e-5, 100);
    State state = context.getState(State::Positions | State::Forces | State::Energy);
    Vec3 delta = state.getPositions()[1]-state.getPositions()[0];
    const Vec3 zero;
    ASSERT_EQUAL_TOL(1.0, sqrt(delta.dot(delta)), 2.0e-5);
    ASSERT_EQUAL_TOL(0.0, state.getPotentialEnergy(), 2.0e-7);
    ASSERT_EQUAL_VEC(zero, state.getForces()[0], 2.0e-5);
    ASSERT_EQUAL_VEC(zero, state.getForces()[1], 2.0e-5);
}

void testUnsupportedFeaturesAreRejected(MetalPlatform& platform) {
    {
        System system;
        system.addParticle(1.0);
        system.addParticle(1.0);
        system.addConstraint(0, 1, 1.0);
        addBond(system, 0, 1);
        assertContextRejected(system, platform);
    }
    {
        System system;
        system.addParticle(1.0);
        system.addParticle(1.0);
        system.addParticle(0.0);
        system.setVirtualSite(2, new TwoParticleAverageSite(0, 1, 0.5, 0.5));
        addBond(system, 0, 1);
        assertContextRejected(system, platform);
    }
    {
        System system;
        system.addParticle(1.0);
        system.addParticle(1.0);
        system.setDefaultPeriodicBoxVectors(Vec3(3, 0, 0), Vec3(0, 3, 0), Vec3(0, 0, 3));
        HarmonicBondForce* force = addBond(system, 0, 1);
        force->setUsesPeriodicBoundaryConditions(true);
        assertContextRejected(system, platform);
    }
    {
        System system;
        system.addParticle(1.0);
        system.addParticle(1.0);
        addBond(system, 0, 1);
        map<string, string> properties;
        properties[MetalPlatform::MetalPrecision()] = "mixed";
        assertContextRejected(system, platform, properties);
    }
    {
        System system;
        system.addParticle(1.0);
        system.addParticle(1.0);
        addBond(system, 0, 1);
        assertContextRejected(system, platform, {{MetalPlatform::MetalDeviceIndex(), "1"}});
        assertContextRejected(system, platform, {{MetalPlatform::MetalDeviceName(), "not a Metal device"}});
    }
    {
        System system;
        system.addParticle(1.0);
        system.addParticle(1.0);
        system.addParticle(1.0);
        HarmonicAngleForce* force = new HarmonicAngleForce();
        force->addAngle(0, 1, 2, 1.0, 100.0);
        system.addForce(force);
        assertContextRejected(system, platform);
    }
    {
        System system;
        system.addParticle(1.0);
        system.addParticle(1.0);
        addBond(system, 0, 1);
        LangevinMiddleIntegrator integrator(300.0, 1.0, 0.001);
        bool rejected = false;
        try {
            Context context(system, integrator, platform);
        }
        catch (const exception&) {
            rejected = true;
        }
        ASSERT(rejected);
    }
}

} // namespace

int main() {
    try {
        if (!MetalPlatform::isPlatformSupported()) {
            cout << "Test skipped: no supported Metal device is visible" << endl;
            return 0;
        }
        MetalPlatform platform;
        testHarmonicBondAndVerlet(platform);
        testMultipleBondsWithSharedParticle(platform);
        testMultipleForceObjectsAndThreadgroups(platform);
        testTrajectoryAgainstReference(platform);
        testUpdateParametersInContext(platform);
        testEnergyOnlyPreservesForces(platform);
        testCheckpointRoundTrip(platform);
        testZeroBondForce(platform);
        testMinimization(platform);
        testUnsupportedFeaturesAreRejected(platform);
    }
    catch (const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
