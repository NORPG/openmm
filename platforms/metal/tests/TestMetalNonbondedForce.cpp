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

#include "MetalPlatform.h"
#include "ReferencePlatform.h"
#include "openmm/Context.h"
#include "openmm/NonbondedForce.h"
#include "openmm/State.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/internal/AssertionUtilities.h"

#include <cmath>
#include <exception>
#include <iostream>
#include <vector>

using namespace OpenMM;
using namespace std;

namespace {

const double COULOMB = 138.935456;
const double ENERGY_TOL = 3e-4;
const double FORCE_TOL = 5e-4;
const double TRAJECTORY_TOL = 3e-4;

struct PairResult {
    double energy;
    Vec3 forceOnFirst;
};

PairResult calculatePair(const Vec3& first, const Vec3& second, double chargeProduct,
                         double sigma, double epsilon) {
    const Vec3 delta = first-second;
    const double r2 = delta.dot(delta);
    const double inverseR = 1.0/sqrt(r2);
    const double sigmaOverR2 = sigma*sigma/r2;
    const double sigmaOverR6 = sigmaOverR2*sigmaOverR2*sigmaOverR2;
    const double energy = COULOMB*chargeProduct*inverseR+
            4.0*epsilon*(sigmaOverR6-1.0)*sigmaOverR6;
    const double forceScale = (COULOMB*chargeProduct*inverseR+
            24.0*epsilon*(2.0*sigmaOverR6-1.0)*sigmaOverR6)/r2;
    return {energy, delta*forceScale};
}

class EnergyOnlyVerletIntegrator : public VerletIntegrator {
public:
    explicit EnergyOnlyVerletIntegrator(double stepSize) : VerletIntegrator(stepSize) {
    }
    bool kineticEnergyRequiresForce() const override {
        return false;
    }
};

void assertZeroForces(const vector<Vec3>& forces, double tolerance=FORCE_TOL) {
    const Vec3 zero;
    for (const Vec3& force : forces)
        ASSERT_EQUAL_VEC(zero, force, tolerance);
}

void assertContextRejected(System& system, MetalPlatform& platform) {
    VerletIntegrator integrator(0.001);
    bool rejected = false;
    try {
        Context context(system, integrator, platform);
    }
    catch (const exception&) {
        rejected = true;
    }
    ASSERT(rejected);
}

void testAnalyticCoulombAndLennardJones(MetalPlatform& platform) {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    NonbondedForce* force = new NonbondedForce();
    force->addParticle(0.5, 0.8, 0.25);
    force->addParticle(-0.75, 1.0, 1.0);
    system.addForce(force);

    const vector<Vec3> positions = {Vec3(0.1, -0.2, 0.3), Vec3(0.7, 0.6, 0.3)};
    const PairResult expected = calculatePair(positions[0], positions[1], -0.375, 0.9, 0.5);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    State state = context.getState(State::Forces | State::Energy);
    ASSERT_EQUAL_TOL(expected.energy, state.getPotentialEnergy(), ENERGY_TOL);
    ASSERT_EQUAL_VEC(expected.forceOnFirst, state.getForces()[0], FORCE_TOL);
    ASSERT_EQUAL_VEC(-expected.forceOnFirst, state.getForces()[1], FORCE_TOL);
}

void testExceptionsAndParameterUpdates(MetalPlatform& platform) {
    System system;
    for (int i = 0; i < 3; i++)
        system.addParticle(1.0);
    NonbondedForce* force = new NonbondedForce();
    force->addParticle(1.0, 1.0, 0.0);
    force->addParticle(-1.0, 0.8, 0.25);
    force->addParticle(0.5, 1.2, 1.0);
    const int exception02 = force->addException(0, 2, 0.3, 1.2, 0.4);
    const int exception01 = force->addException(0, 1, 0.0, 1.0, 0.0);
    system.addForce(force);
    const vector<Vec3> positions = {Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(2, 0, 0)};

    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(positions);

    PairResult exceptionPair = calculatePair(positions[0], positions[2], 0.3, 1.2, 0.4);
    PairResult regularPair = calculatePair(positions[1], positions[2], -0.5, 1.0, 0.5);
    State state = context.getState(State::Forces | State::Energy);
    ASSERT_EQUAL_TOL(exceptionPair.energy+regularPair.energy, state.getPotentialEnergy(), ENERGY_TOL);
    ASSERT_EQUAL_VEC(exceptionPair.forceOnFirst, state.getForces()[0], FORCE_TOL);
    ASSERT_EQUAL_VEC(regularPair.forceOnFirst, state.getForces()[1], FORCE_TOL);
    ASSERT_EQUAL_VEC(-exceptionPair.forceOnFirst-regularPair.forceOnFirst,
                     state.getForces()[2], FORCE_TOL);

    // A zero exception must be skipped before evaluating distance, so excluded
    // particles can overlap without contaminating other interactions with NaN.
    vector<Vec3> overlapped = positions;
    overlapped[1] = overlapped[0];
    context.setPositions(overlapped);
    exceptionPair = calculatePair(overlapped[0], overlapped[2], 0.3, 1.2, 0.4);
    regularPair = calculatePair(overlapped[1], overlapped[2], -0.5, 1.0, 0.5);
    state = context.getState(State::Forces | State::Energy);
    ASSERT_EQUAL_TOL(exceptionPair.energy+regularPair.energy, state.getPotentialEnergy(), ENERGY_TOL);
    ASSERT_EQUAL_VEC(exceptionPair.forceOnFirst, state.getForces()[0], FORCE_TOL);
    context.setPositions(positions);

    force->setParticleParameters(1, -0.8, 0.6, 0.09);
    force->setExceptionParameters(exception02, 0, 2, -0.2, 0.9, 0.15);
    force->updateParametersInContext(context);
    exceptionPair = calculatePair(positions[0], positions[2], -0.2, 0.9, 0.15);
    regularPair = calculatePair(positions[1], positions[2], -0.4, 0.9, 0.3);
    state = context.getState(State::Forces | State::Energy);
    ASSERT_EQUAL_TOL(exceptionPair.energy+regularPair.energy, state.getPotentialEnergy(), ENERGY_TOL);
    ASSERT_EQUAL_VEC(exceptionPair.forceOnFirst, state.getForces()[0], FORCE_TOL);
    ASSERT_EQUAL_VEC(regularPair.forceOnFirst, state.getForces()[1], FORCE_TOL);

    // Changing an exclusion into an interacting exception must not require a
    // topology rebuild: every exception is present in the Metal CSR.
    force->setExceptionParameters(exception01, 0, 1, 0.1, 0.7, 0.05);
    force->updateParametersInContext(context);
    const PairResult restoredPair = calculatePair(positions[0], positions[1], 0.1, 0.7, 0.05);
    state = context.getState(State::Forces | State::Energy);
    ASSERT_EQUAL_TOL(restoredPair.energy+exceptionPair.energy+regularPair.energy,
                     state.getPotentialEnergy(), ENERGY_TOL);
    ASSERT_EQUAL_VEC(restoredPair.forceOnFirst+exceptionPair.forceOnFirst,
                     state.getForces()[0], FORCE_TOL);

    // Validate the complete candidate model before uploading either changed
    // range.  A rejected topology update must not partially apply the particle
    // update that was submitted in the same call.
    force->setParticleParameters(1, -0.6, 0.6, 0.09);
    force->setExceptionParameters(exception02, 1, 2, -0.2, 0.9, 0.15);
    bool rejected = false;
    try {
        force->updateParametersInContext(context);
    }
    catch (const exception&) {
        rejected = true;
    }
    ASSERT(rejected);
    state = context.getState(State::Forces | State::Energy);
    ASSERT_EQUAL_TOL(restoredPair.energy+exceptionPair.energy+regularPair.energy,
                     state.getPotentialEnergy(), ENERGY_TOL);
    ASSERT_EQUAL_VEC(restoredPair.forceOnFirst+exceptionPair.forceOnFirst,
                     state.getForces()[0], FORCE_TOL);
}

void testMultipleForcesAndGroups(MetalPlatform& platform) {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    NonbondedForce* first = new NonbondedForce();
    first->addParticle(-1.5, 1.0, 1.2);
    first->addParticle(0.5, 1.0, 1.0);
    system.addForce(first);
    NonbondedForce* second = new NonbondedForce();
    second->addParticle(0.4, 1.4, 0.5);
    second->addParticle(0.3, 1.8, 1.0);
    second->setForceGroup(1);
    system.addForce(second);

    const vector<Vec3> positions = {Vec3(0, 0, 0), Vec3(1.5, 0, 0)};
    const PairResult expectedFirst = calculatePair(positions[0], positions[1], -0.75, 1.0,
                                                    sqrt(1.2));
    const PairResult expectedSecond = calculatePair(positions[0], positions[1], 0.12, 1.6,
                                                     sqrt(0.5));
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(positions);

    State firstState = context.getState(State::Forces | State::Energy, false, 1 << 0);
    State secondState = context.getState(State::Forces | State::Energy, false, 1 << 1);
    State allState = context.getState(State::Forces | State::Energy);
    State unusedState = context.getState(State::Forces | State::Energy, false, 1 << 2);
    ASSERT_EQUAL_TOL(expectedFirst.energy, firstState.getPotentialEnergy(), ENERGY_TOL);
    ASSERT_EQUAL_TOL(expectedSecond.energy, secondState.getPotentialEnergy(), ENERGY_TOL);
    ASSERT_EQUAL_VEC(expectedFirst.forceOnFirst, firstState.getForces()[0], FORCE_TOL);
    ASSERT_EQUAL_VEC(expectedSecond.forceOnFirst, secondState.getForces()[0], FORCE_TOL);
    ASSERT_EQUAL_TOL(expectedFirst.energy+expectedSecond.energy, allState.getPotentialEnergy(), ENERGY_TOL);
    ASSERT_EQUAL_VEC(expectedFirst.forceOnFirst+expectedSecond.forceOnFirst,
                     allState.getForces()[0], FORCE_TOL);
    ASSERT_EQUAL_TOL(0.0, unusedState.getPotentialEnergy(), ENERGY_TOL);
    assertZeroForces(unusedState.getForces());
}

void testIncludeFlags(MetalPlatform& platform) {
    {
        System system;
        system.addParticle(1.0);
        system.addParticle(1.0);
        NonbondedForce* force = new NonbondedForce();
        force->addParticle(0.5, 1.0, 0.0);
        force->addParticle(-0.5, 1.0, 0.0);
        system.addForce(force);
        const double stepSize = 0.01;
        EnergyOnlyVerletIntegrator integrator(stepSize);
        Context context(system, integrator, platform);
        context.setPositions({Vec3(0, 0, 0), Vec3(1, 0, 0)});
        context.setVelocities({Vec3(), Vec3()});
        State forceState = context.getState(State::Forces);
        const double forceMagnitude = COULOMB*0.25;
        ASSERT_EQUAL_VEC(Vec3(forceMagnitude, 0, 0), forceState.getForces()[0], FORCE_TOL);
        State energyState = context.getState(State::Energy);
        ASSERT_EQUAL_TOL(-forceMagnitude, energyState.getPotentialEnergy(), ENERGY_TOL);
        ASSERT_EQUAL_TOL(0.25*stepSize*stepSize*forceMagnitude*forceMagnitude,
                         energyState.getKineticEnergy(), ENERGY_TOL);
    }
    {
        System system;
        system.addParticle(1.0);
        system.addParticle(1.0);
        NonbondedForce* force = new NonbondedForce();
        force->addParticle(1.0, 0.5, 0.2);
        force->addParticle(-1.0, 0.7, 0.3);
        force->setIncludeDirectSpace(false);
        system.addForce(force);
        VerletIntegrator integrator(0.001);
        Context context(system, integrator, platform);
        context.setPositions({Vec3(0, 0, 0), Vec3(1, 0, 0)});
        State state = context.getState(State::Forces | State::Energy);
        ASSERT_EQUAL_TOL(0.0, state.getPotentialEnergy(), ENERGY_TOL);
        assertZeroForces(state.getForces());
    }
}

void testThreadgroups(MetalPlatform& platform) {
    const int numParticles = 257;
    System system;
    NonbondedForce* force = new NonbondedForce();
    for (int i = 0; i < numParticles; i++) {
        system.addParticle(1.0);
        force->addParticle(0.0, 1.0, 0.0);
    }
    force->addException(0, numParticles-1, 0.25, 1.0, 0.0);
    system.addForce(force);
    vector<Vec3> positions(numParticles);
    for (int i = 0; i < numParticles; i++)
        positions[i] = Vec3(0.01*i, 0.02*(i%7), -0.01*(i%5));
    positions[0] = Vec3(0, 0, 0);
    positions[numParticles-1] = Vec3(2, 0, 0);
    const PairResult expected = calculatePair(positions[0], positions[numParticles-1],
                                               0.25, 1.0, 0.0);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    State state = context.getState(State::Forces | State::Energy);
    ASSERT_EQUAL_TOL(expected.energy, state.getPotentialEnergy(), ENERGY_TOL);
    ASSERT_EQUAL_VEC(expected.forceOnFirst, state.getForces()[0], FORCE_TOL);
    ASSERT_EQUAL_VEC(-expected.forceOnFirst, state.getForces()[numParticles-1], FORCE_TOL);
    const Vec3 zero;
    ASSERT_EQUAL_VEC(zero, state.getForces()[128], FORCE_TOL);
}

void testTrajectoryAgainstReference(MetalPlatform& platform) {
    System system;
    const vector<double> masses = {12.0, 16.0, 14.0, 10.0};
    for (double mass : masses)
        system.addParticle(mass);
    NonbondedForce* force = new NonbondedForce();
    force->addParticle(0.20, 0.30, 0.20);
    force->addParticle(-0.25, 0.32, 0.15);
    force->addParticle(0.15, 0.28, 0.10);
    force->addParticle(-0.10, 0.35, 0.25);
    force->addException(0, 3, 0.0, 1.0, 0.0);
    system.addForce(force);
    const vector<Vec3> positions = {
        Vec3(0.05, -0.02, 0.01), Vec3(1.12, 0.06, -0.03),
        Vec3(-0.31, 1.08, 0.19), Vec3(0.42, -0.57, 1.26)
    };
    const vector<Vec3> velocities = {
        Vec3(0.03, -0.02, 0.01), Vec3(-0.01, 0.04, -0.02),
        Vec3(0.02, 0.01, -0.03), Vec3(-0.03, 0.02, 0.04)
    };

    const double stepSize = 0.0005;
    ReferencePlatform referencePlatform;
    VerletIntegrator referenceIntegrator(stepSize);
    VerletIntegrator metalIntegrator(stepSize);
    Context referenceContext(system, referenceIntegrator, referencePlatform);
    Context metalContext(system, metalIntegrator, platform);
    referenceContext.setPositions(positions);
    referenceContext.setVelocities(velocities);
    metalContext.setPositions(positions);
    metalContext.setVelocities(velocities);

    State expected = referenceContext.getState(State::Forces | State::Energy);
    State found = metalContext.getState(State::Forces | State::Energy);
    ASSERT_EQUAL_TOL(expected.getPotentialEnergy(), found.getPotentialEnergy(), FORCE_TOL);
    for (int i = 0; i < 4; i++)
        ASSERT_EQUAL_VEC(expected.getForces()[i], found.getForces()[i], FORCE_TOL);

    for (int step = 0; step < 20; step++) {
        referenceIntegrator.step(1);
        metalIntegrator.step(1);
        expected = referenceContext.getState(State::Positions | State::Velocities | State::Energy);
        found = metalContext.getState(State::Positions | State::Velocities | State::Energy);
        for (int i = 0; i < 4; i++) {
            ASSERT_EQUAL_VEC(expected.getPositions()[i], found.getPositions()[i], TRAJECTORY_TOL);
            ASSERT_EQUAL_VEC(expected.getVelocities()[i], found.getVelocities()[i], TRAJECTORY_TOL);
        }
        ASSERT_EQUAL_TOL(expected.getPotentialEnergy(), found.getPotentialEnergy(), TRAJECTORY_TOL);
        ASSERT_EQUAL_TOL(expected.getKineticEnergy(), found.getKineticEnergy(), TRAJECTORY_TOL);
        ASSERT_EQUAL_TOL(expected.getTime(), found.getTime(), TRAJECTORY_TOL);
        ASSERT_EQUAL(expected.getStepCount(), found.getStepCount());
    }
}

void testUnsupportedModesAndOffsets(MetalPlatform& platform) {
    const vector<NonbondedForce::NonbondedMethod> methods = {
        NonbondedForce::CutoffNonPeriodic, NonbondedForce::CutoffPeriodic,
        NonbondedForce::Ewald, NonbondedForce::PME, NonbondedForce::LJPME
    };
    for (NonbondedForce::NonbondedMethod method : methods) {
        System system;
        system.addParticle(1.0);
        system.addParticle(1.0);
        system.setDefaultPeriodicBoxVectors(Vec3(4, 0, 0), Vec3(0, 4, 0), Vec3(0, 0, 4));
        NonbondedForce* force = new NonbondedForce();
        force->addParticle(1.0, 0.5, 0.2);
        force->addParticle(-1.0, 0.5, 0.2);
        force->setNonbondedMethod(method);
        force->setCutoffDistance(1.0);
        system.addForce(force);
        assertContextRejected(system, platform);
    }
    {
        System system;
        system.addParticle(1.0);
        system.addParticle(1.0);
        NonbondedForce* force = new NonbondedForce();
        force->addParticle(1.0, 0.5, 0.2);
        force->addParticle(-1.0, 0.5, 0.2);
        force->addGlobalParameter("lambda", 0.0);
        force->addParticleParameterOffset("lambda", 0, 1.0, 0.0, 0.0);
        system.addForce(force);
        assertContextRejected(system, platform);
    }
    {
        System system;
        system.addParticle(1.0);
        system.addParticle(1.0);
        NonbondedForce* force = new NonbondedForce();
        force->addParticle(1.0, 0.5, 0.2);
        force->addParticle(-1.0, 0.5, 0.2);
        force->addException(0, 1, 0.0, 1.0, 0.0);
        system.addForce(force);
        VerletIntegrator integrator(0.001);
        Context context(system, integrator, platform);
        context.setPositions({Vec3(0, 0, 0), Vec3(1, 0, 0)});
        double alpha;
        int nx, ny, nz;
        bool pmeRejected = false;
        bool ljpmeRejected = false;
        try {
            force->getPMEParametersInContext(context, alpha, nx, ny, nz);
        }
        catch (const exception&) {
            pmeRejected = true;
        }
        try {
            force->getLJPMEParametersInContext(context, alpha, nx, ny, nz);
        }
        catch (const exception&) {
            ljpmeRejected = true;
        }
        ASSERT(pmeRejected);
        ASSERT(ljpmeRejected);
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
        testAnalyticCoulombAndLennardJones(platform);
        testExceptionsAndParameterUpdates(platform);
        testMultipleForcesAndGroups(platform);
        testIncludeFlags(platform);
        testThreadgroups(platform);
        testTrajectoryAgainstReference(platform);
        testUnsupportedModesAndOffsets(platform);
    }
    catch (const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
