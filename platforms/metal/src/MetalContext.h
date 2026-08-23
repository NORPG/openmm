#ifndef OPENMM_METALCONTEXT_H_
#define OPENMM_METALCONTEXT_H_

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

#include "MetalArray.h"
#include "MetalQueue.h"
#include "openmm/System.h"
#include "openmm/Vec3.h"
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

namespace OpenMM {

/** Four single precision values with the layout used by MSL float4. */
struct MetalFloat4 {
    float x, y, z, w;
};

/**
 * Device state for the first native Metal vertical slice.
 *
 * This is intentionally not yet a ComputeContext implementation.  It owns the
 * small, explicit set of buffers needed by HarmonicBondForce and
 * VerletIntegrator while the general Common-source-to-MSL lowering layer is
 * still being developed.
 */
class MetalContext {
public:
    MetalContext(const System& system, size_t deviceIndex);
    ~MetalContext();

    int getNumParticles() const;
    MetalQueue& getQueue();
    MetalArray& getPositions();
    MetalArray& getVelocities();
    MetalArray& getForces();
    MetalArray& getInverseMasses();

    void setPositions(const std::vector<Vec3>& values);
    void getPositions(std::vector<Vec3>& values) const;
    void setVelocities(const std::vector<Vec3>& values);
    void getVelocities(std::vector<Vec3>& values) const;
    void getForces(std::vector<Vec3>& values) const;
    void clearForces();

    double getTime() const;
    void setTime(double value);
    long long getStepCount() const;
    void setStepCount(long long value);
    void advanceTime(double stepSize);

    void getPeriodicBoxVectors(Vec3& a, Vec3& b, Vec3& c) const;
    void setPeriodicBoxVectors(const Vec3& a, const Vec3& b, const Vec3& c);

    void createCheckpoint(std::ostream& stream) const;
    void loadCheckpoint(std::istream& stream);

private:
    std::shared_ptr<MetalQueue> queue;
    std::unique_ptr<MetalArray> positions;
    std::unique_ptr<MetalArray> velocities;
    std::unique_ptr<MetalArray> forces;
    std::unique_ptr<MetalArray> inverseMasses;
    int numParticles;
    double time;
    long long stepCount;
    Vec3 boxVectors[3];
};

} // namespace OpenMM

#endif /*OPENMM_METALCONTEXT_H_*/
