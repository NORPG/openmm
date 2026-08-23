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

#include "MetalContext.h"
#include "openmm/OpenMMException.h"
#include <cstdint>
#include <istream>
#include <ostream>
#include <sstream>

using namespace OpenMM;
using namespace std;

namespace {

const uint32_t checkpointMagic = 0x4d544c31; // "MTL1"
const uint32_t checkpointVersion = 1;

template <class T>
void writeValue(ostream& stream, const T& value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!stream)
        throw OpenMMException("Error writing a Metal checkpoint");
}

template <class T>
void readValue(istream& stream, T& value) {
    stream.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!stream)
        throw OpenMMException("Error reading a Metal checkpoint");
}

vector<MetalFloat4> toFloat4(const vector<Vec3>& values, int expectedSize, const string& label) {
    if (values.size() != static_cast<size_t>(expectedSize)) {
        stringstream message;
        message << "Error setting Metal " << label << ": expected " << expectedSize
                << " values but received " << values.size();
        throw OpenMMException(message.str());
    }
    vector<MetalFloat4> result(expectedSize);
    for (int i = 0; i < expectedSize; i++) {
        result[i].x = static_cast<float>(values[i][0]);
        result[i].y = static_cast<float>(values[i][1]);
        result[i].z = static_cast<float>(values[i][2]);
        result[i].w = 0.0f;
    }
    return result;
}

void fromFloat4(const vector<MetalFloat4>& values, vector<Vec3>& result) {
    result.resize(values.size());
    for (int i = 0; i < static_cast<int>(values.size()); i++)
        result[i] = Vec3(values[i].x, values[i].y, values[i].z);
}

} // namespace

MetalContext::MetalContext(const System& system, size_t deviceIndex) :
        queue(new MetalQueue(deviceIndex)), numParticles(system.getNumParticles()), time(0.0), stepCount(0) {
    positions.reset(new MetalArray(*queue, numParticles, sizeof(MetalFloat4), "Metal positions"));
    velocities.reset(new MetalArray(*queue, numParticles, sizeof(MetalFloat4), "Metal velocities"));
    forces.reset(new MetalArray(*queue, numParticles, sizeof(MetalFloat4), "Metal forces"));
    inverseMasses.reset(new MetalArray(*queue, numParticles, sizeof(float), "Metal inverse masses"));

    vector<MetalFloat4> zeroVectors(numParticles, {0.0f, 0.0f, 0.0f, 0.0f});
    vector<float> inverseMassValues(numParticles);
    for (int i = 0; i < numParticles; i++) {
        double mass = system.getParticleMass(i);
        inverseMassValues[i] = (mass == 0.0 ? 0.0f : static_cast<float>(1.0/mass));
    }
    positions->upload(zeroVectors);
    velocities->upload(zeroVectors);
    forces->upload(zeroVectors);
    inverseMasses->upload(inverseMassValues);
    system.getDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
}

MetalContext::~MetalContext() {
    if (queue != NULL) {
        try {
            queue->waitUntilIdle();
        }
        catch (...) {
            // Destructors must not throw.  Synchronous API boundaries report
            // command errors before platform data is destroyed.
        }
    }
}

int MetalContext::getNumParticles() const {
    return numParticles;
}

MetalQueue& MetalContext::getQueue() {
    return *queue;
}

MetalArray& MetalContext::getPositions() {
    return *positions;
}

MetalArray& MetalContext::getVelocities() {
    return *velocities;
}

MetalArray& MetalContext::getForces() {
    return *forces;
}

MetalArray& MetalContext::getInverseMasses() {
    return *inverseMasses;
}

void MetalContext::setPositions(const vector<Vec3>& values) {
    positions->upload(toFloat4(values, numParticles, "positions"));
}

void MetalContext::getPositions(vector<Vec3>& values) const {
    vector<MetalFloat4> data;
    positions->download(data);
    fromFloat4(data, values);
}

void MetalContext::setVelocities(const vector<Vec3>& values) {
    velocities->upload(toFloat4(values, numParticles, "velocities"));
}

void MetalContext::getVelocities(vector<Vec3>& values) const {
    vector<MetalFloat4> data;
    velocities->download(data);
    fromFloat4(data, values);
}

void MetalContext::getForces(vector<Vec3>& values) const {
    vector<MetalFloat4> data;
    forces->download(data);
    fromFloat4(data, values);
}

void MetalContext::clearForces() {
    vector<MetalFloat4> zero(numParticles, {0.0f, 0.0f, 0.0f, 0.0f});
    forces->upload(static_cast<const void*>(zero.data()), false);
}

double MetalContext::getTime() const {
    return time;
}

void MetalContext::setTime(double value) {
    time = value;
}

long long MetalContext::getStepCount() const {
    return stepCount;
}

void MetalContext::setStepCount(long long value) {
    stepCount = value;
}

void MetalContext::advanceTime(double stepSize) {
    time += stepSize;
    stepCount++;
}

void MetalContext::getPeriodicBoxVectors(Vec3& a, Vec3& b, Vec3& c) const {
    a = boxVectors[0];
    b = boxVectors[1];
    c = boxVectors[2];
}

void MetalContext::setPeriodicBoxVectors(const Vec3& a, const Vec3& b, const Vec3& c) {
    boxVectors[0] = a;
    boxVectors[1] = b;
    boxVectors[2] = c;
}

void MetalContext::createCheckpoint(ostream& stream) const {
    vector<MetalFloat4> positionData;
    vector<MetalFloat4> velocityData;
    vector<MetalFloat4> forceData;
    positions->download(positionData);
    velocities->download(velocityData);
    forces->download(forceData);
    writeValue(stream, checkpointMagic);
    writeValue(stream, checkpointVersion);
    writeValue(stream, numParticles);
    writeValue(stream, time);
    writeValue(stream, stepCount);
    stream.write(reinterpret_cast<const char*>(boxVectors), sizeof(boxVectors));
    stream.write(reinterpret_cast<const char*>(positionData.data()), positionData.size()*sizeof(MetalFloat4));
    stream.write(reinterpret_cast<const char*>(velocityData.data()), velocityData.size()*sizeof(MetalFloat4));
    stream.write(reinterpret_cast<const char*>(forceData.data()), forceData.size()*sizeof(MetalFloat4));
    if (!stream)
        throw OpenMMException("Error writing a Metal checkpoint");
}

void MetalContext::loadCheckpoint(istream& stream) {
    uint32_t magic, version;
    int particleCount;
    readValue(stream, magic);
    readValue(stream, version);
    readValue(stream, particleCount);
    if (magic != checkpointMagic || version != checkpointVersion)
        throw OpenMMException("This is not a supported Metal checkpoint");
    if (particleCount != numParticles)
        throw OpenMMException("The Metal checkpoint contains a different number of particles");
    readValue(stream, time);
    readValue(stream, stepCount);
    stream.read(reinterpret_cast<char*>(boxVectors), sizeof(boxVectors));
    vector<MetalFloat4> positionData(numParticles);
    vector<MetalFloat4> velocityData(numParticles);
    vector<MetalFloat4> forceData(numParticles);
    stream.read(reinterpret_cast<char*>(positionData.data()), positionData.size()*sizeof(MetalFloat4));
    stream.read(reinterpret_cast<char*>(velocityData.data()), velocityData.size()*sizeof(MetalFloat4));
    stream.read(reinterpret_cast<char*>(forceData.data()), forceData.size()*sizeof(MetalFloat4));
    if (!stream)
        throw OpenMMException("Error reading a Metal checkpoint");
    positions->upload(positionData);
    velocities->upload(velocityData);
    forces->upload(forceData);
}
