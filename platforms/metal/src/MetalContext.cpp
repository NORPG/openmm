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
#include "MetalFixedPoint.h"
#include "MetalEvent.h"
#include "MetalProgram.h"
#include "openmm/OpenMMException.h"
#include "openmm/common/ComputeArray.h"
#include "openmm/internal/ThreadPool.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <istream>
#include <limits>
#include <memory>
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

void validateVectorSize(size_t actual, int expected, const string& label) {
    if (actual != static_cast<size_t>(expected)) {
        stringstream message;
        message << "Error setting Metal " << label << ": expected " << expected
                << " values but received " << actual;
        throw OpenMMException(message.str());
    }
}

void fromFloat4(const vector<MetalFloat4>& values, int count, vector<Vec3>& result) {
    result.resize(count);
    for (int i = 0; i < count; i++)
        result[i] = Vec3(values[i].x, values[i].y, values[i].z);
}

string addDefines(const string& source, const map<string, string>& defines) {
    string result;
    for (const auto& define : defines) {
        result += "#define "+define.first;
        if (!define.second.empty())
            result += " "+define.second;
        result += "\n";
    }
    return result+source;
}

[[noreturn]] void throwUnsupported(const string& feature) {
    throw OpenMMException("The minimal Metal ComputeContext does not yet support "+feature);
}

} // namespace

MetalContext::MetalContext(const System& system, ContextImpl* owner, size_t deviceIndex) :
        ComputeContext(system), queue(new MetalQueue(deviceIndex)), threadPool(new ThreadPool(1)),
        contextImpl(owner), numAtomBlocks(0), numThreadBlocks(1), energyWorkspace(0.0),
        hasAssignedPosqCharges(false) {
    defaultQueue = queue;
    currentQueue = queue;

    numAtoms = system.getNumParticles();
    paddedNumAtoms = TileSize*((numAtoms+TileSize-1)/TileSize);
    numAtomBlocks = paddedNumAtoms/TileSize;
    numThreadBlocks = max(1, numAtomBlocks);

    positions.reset(new MetalArray());
    velocities.reset(new MetalArray());
    forces.reset(new MetalArray());
    inverseMasses.reset(new MetalArray());
    energyBuffer.reset(new MetalArray());
    energyParamDerivBuffer.reset(new MetalArray());
    atomIndexDevice.reset(new MetalArray());
    positions->initialize(*this, *queue, paddedNumAtoms, sizeof(MetalFloat4), "Metal posq");
    velocities->initialize(*this, *queue, paddedNumAtoms, sizeof(MetalFloat4), "Metal velm");
    forces->initialize(*this, *queue, paddedNumAtoms, sizeof(MetalFloat4), "Metal force buffer");
    inverseMasses->initialize(*this, *queue, paddedNumAtoms, sizeof(float), "Metal inverse masses");
    energyBuffer->initialize(*this, *queue, numThreadBlocks*ThreadBlockSize, sizeof(float), "Metal energy buffer");
    energyParamDerivBuffer->initialize(*this, *queue, 1, sizeof(float), "Metal energy parameter derivative buffer");
    atomIndexDevice->initialize(*this, *queue, paddedNumAtoms, sizeof(int), "Metal atom index");

    charges.assign(paddedNumAtoms, 0.0f);
    inverseMassValues.assign(paddedNumAtoms, 0.0f);
    vector<MetalFloat4> positionValues(paddedNumAtoms, {0.0f, 0.0f, 0.0f, 0.0f});
    vector<MetalFloat4> velocityValues(paddedNumAtoms, {0.0f, 0.0f, 0.0f, 0.0f});
    vector<MetalFloat4> forceValues(paddedNumAtoms, {0.0f, 0.0f, 0.0f, 0.0f});
    atomIndex.resize(paddedNumAtoms);
    posCellOffsets.resize(paddedNumAtoms, mm_int4(0, 0, 0, 0));
    for (int i = 0; i < paddedNumAtoms; i++) {
        atomIndex[i] = i;
        if (i < numAtoms) {
            double mass = system.getParticleMass(i);
            inverseMassValues[i] = (mass == 0.0 ? 0.0f : static_cast<float>(1.0/mass));
            velocityValues[i].w = inverseMassValues[i];
        }
    }
    positions->upload(positionValues);
    velocities->upload(velocityValues);
    forces->upload(forceValues);
    inverseMasses->upload(inverseMassValues);
    atomIndexDevice->upload(atomIndex);
    clearBuffer(*energyBuffer);
    clearBuffer(*energyParamDerivBuffer);
    updatePinnedBufferSize();
    system.getDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
}

MetalContext::~MetalContext() {
    if (workThread != NULL) {
        try {
            workThread->flush();
        }
        catch (...) {
        }
    }
    try {
        MetalQueue& current = getCurrentMetalQueue();
        current.waitUntilIdle();
        if (&current != queue.get())
            queue->waitUntilIdle();
    }
    catch (...) {
        // Destructors must not throw.  Synchronous API boundaries report
        // command errors before platform data is destroyed.
    }
    for (auto force : ComputeContext::forces)
        delete force;
    for (auto listener : reorderListeners)
        delete listener;
    for (auto computation : preComputations)
        delete computation;
    for (auto computation : postComputations)
        delete computation;
    delete workThread;
    workThread = NULL;
}

int MetalContext::getNumParticles() const {
    return numAtoms;
}

MetalQueue& MetalContext::getQueue() {
    return *queue;
}

MetalQueue& MetalContext::getCurrentMetalQueue() {
    ComputeQueue current = getCurrentQueue();
    MetalQueue* metal = dynamic_cast<MetalQueue*>(current.get());
    if (metal == NULL)
        throw OpenMMException("The current queue is not a MetalQueue");
    return *metal;
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
    validateVectorSize(values.size(), numAtoms, "positions");
    vector<MetalFloat4> data(paddedNumAtoms, {0.0f, 0.0f, 0.0f, 0.0f});
    for (int i = 0; i < numAtoms; i++) {
        data[i].x = static_cast<float>(values[i][0]);
        data[i].y = static_cast<float>(values[i][1]);
        data[i].z = static_cast<float>(values[i][2]);
        data[i].w = charges[i];
    }
    positions->upload(data);
}

void MetalContext::getPositions(vector<Vec3>& values) const {
    vector<MetalFloat4> data;
    positions->download(data);
    fromFloat4(data, numAtoms, values);
}

void MetalContext::setVelocities(const vector<Vec3>& values) {
    validateVectorSize(values.size(), numAtoms, "velocities");
    vector<MetalFloat4> data(paddedNumAtoms, {0.0f, 0.0f, 0.0f, 0.0f});
    for (int i = 0; i < numAtoms; i++) {
        data[i].x = static_cast<float>(values[i][0]);
        data[i].y = static_cast<float>(values[i][1]);
        data[i].z = static_cast<float>(values[i][2]);
        data[i].w = inverseMassValues[i];
    }
    velocities->upload(data);
}

void MetalContext::getVelocities(vector<Vec3>& values) const {
    vector<MetalFloat4> data;
    velocities->download(data);
    fromFloat4(data, numAtoms, values);
}

void MetalContext::getForces(vector<Vec3>& values) const {
    vector<MetalFloat4> data;
    forces->download(data);
    fromFloat4(data, numAtoms, values);
}

void MetalContext::clearForces() {
    clearBuffer(*forces);
}

void MetalContext::clearAutoclearBuffers() {
    for (ArrayInterface* array : autoclearBuffers)
        clearBuffer(*array);
}

void MetalContext::advanceTime(double stepSize) {
    time += stepSize;
    stepCount++;
}

int MetalContext::getNumContexts() const {
    return 1;
}

int MetalContext::getContextIndex() const {
    return 0;
}

vector<ComputeContext*> MetalContext::getAllContexts() {
    return {this};
}

ContextImpl* MetalContext::getContextImpl() {
    return contextImpl;
}

double& MetalContext::getEnergyWorkspace() {
    return energyWorkspace;
}

ComputeQueue MetalContext::createQueue() {
    return queue->createSiblingQueue();
}

MetalArray* MetalContext::createArray() {
    return new MetalArray();
}

ComputeEvent MetalContext::createEvent() {
    return ComputeEvent(new MetalEvent(getCurrentMetalQueue()));
}

ComputeSort MetalContext::createSort(ComputeSortImpl::SortTrait* trait, unsigned int length, bool uniform) {
    unique_ptr<ComputeSortImpl::SortTrait> ownedTrait(trait);
    (void) length;
    (void) uniform;
    throwUnsupported("sorting");
}

ComputeProgram MetalContext::compileProgram(const string source, const map<string, string>& defines) {
    return ComputeProgram(new MetalProgram(getCurrentMetalQueue(), addDefines(source, defines)));
}

int MetalContext::computeThreadBlockSize(double memory) const {
    if (!isfinite(memory) || memory < 0.0)
        throw OpenMMException("Metal per-thread memory must be a finite nonnegative value");
    int maxThreads = getMaxThreadBlockSize();
    if (memory == 0.0)
        return maxThreads;
    double memoryLimit = static_cast<double>(queue->getDeviceCaps().getMaxThreadgroupMemoryLength())/memory;
    int limit = static_cast<int>(min<double>(maxThreads, max(1.0, floor(memoryLimit))));
    int simdWidth = getSIMDWidth();
    if (limit < simdWidth)
        return min(simdWidth, maxThreads);
    return max(simdWidth, (limit/simdWidth)*simdWidth);
}

MetalArray& MetalContext::unwrap(ArrayInterface& array) const {
    ArrayInterface* candidate = &array;
    ComputeArray* wrapper = dynamic_cast<ComputeArray*>(&array);
    if (wrapper != NULL)
        candidate = &wrapper->getArray();
    MetalArray* metal = dynamic_cast<MetalArray*>(candidate);
    if (metal == NULL)
        throw OpenMMException("Array argument is not a MetalArray");
    return *metal;
}

void MetalContext::clearBuffer(ArrayInterface& array) {
    unwrap(array).clear(false);
}

void MetalContext::addAutoclearBuffer(ArrayInterface& array) {
    MetalArray* metal = &unwrap(array);
    if (find(autoclearBuffers.begin(), autoclearBuffers.end(), metal) == autoclearBuffers.end())
        autoclearBuffers.push_back(metal);
}

bool MetalContext::getIsCPU() const {
    return false;
}

int MetalContext::getSIMDWidth() const {
    return 32;
}

bool MetalContext::getSupports64BitGlobalAtomics() const {
    return false;
}

bool MetalContext::getSupportsDoublePrecision() const {
    return false;
}

bool MetalContext::getUseDoublePrecision() const {
    return false;
}

bool MetalContext::getUseMixedPrecision() const {
    return false;
}

int MetalContext::getNumAtomBlocks() const {
    return numAtomBlocks;
}

int MetalContext::getNumThreadBlocks() const {
    return numThreadBlocks;
}

int MetalContext::getMaxThreadBlockSize() const {
    size_t value = queue->getDeviceCaps().getMaxThreadsPerThreadgroup();
    return static_cast<int>(min(value, static_cast<size_t>(numeric_limits<int>::max())));
}

MetalArray& MetalContext::getPosq() {
    return *positions;
}

ArrayInterface& MetalContext::getPosqCorrection() {
    throwUnsupported("mixed-precision position corrections");
}

MetalArray& MetalContext::getVelm() {
    return *velocities;
}

MetalArray& MetalContext::getForceBuffers() {
    return *forces;
}

MetalArray& MetalContext::getFloatForceBuffer() {
    return *forces;
}

ArrayInterface& MetalContext::getLongForceBuffer() {
    throwUnsupported("64-bit fixed-point force buffers");
}

MetalArray& MetalContext::getEnergyBuffer() {
    return *energyBuffer;
}

MetalArray& MetalContext::getEnergyParamDerivBuffer() {
    return *energyParamDerivBuffer;
}

void* MetalContext::getPinnedBuffer() {
    return pinnedBuffer.data();
}

ThreadPool& MetalContext::getThreadPool() {
    return *threadPool;
}

MetalArray& MetalContext::getAtomIndexArray() {
    return *atomIndexDevice;
}

bool MetalContext::getBoxIsTriclinic() const {
    return boxVectors[0][1] != 0.0 || boxVectors[0][2] != 0.0 ||
           boxVectors[1][0] != 0.0 || boxVectors[1][2] != 0.0 ||
           boxVectors[2][0] != 0.0 || boxVectors[2][1] != 0.0;
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

IntegrationUtilities& MetalContext::getIntegrationUtilities() {
    throwUnsupported("Common integration utilities");
}

ExpressionUtilities& MetalContext::getExpressionUtilities() {
    throwUnsupported("Common expression utilities");
}

BondedUtilities& MetalContext::getBondedUtilities() {
    throwUnsupported("Common bonded utilities");
}

NonbondedUtilities& MetalContext::getNonbondedUtilities() {
    throwUnsupported("Common nonbonded utilities");
}

NonbondedUtilities* MetalContext::createNonbondedUtilities() {
    throwUnsupported("Common nonbonded utilities");
}

FFT3D MetalContext::createFFT(int xsize, int ysize, int zsize, bool realToComplex) {
    (void) xsize;
    (void) ysize;
    (void) zsize;
    (void) realToComplex;
    throwUnsupported("FFT");
}

void MetalContext::initializeContexts() {
}

void MetalContext::setCharges(const vector<double>& values) {
    validateVectorSize(values.size(), numAtoms, "charges");
    vector<MetalFloat4> data;
    positions->download(data);
    for (int i = 0; i < numAtoms; i++) {
        charges[i] = static_cast<float>(values[i]);
        data[i].w = charges[i];
    }
    positions->upload(data);
}

bool MetalContext::requestPosqCharges() {
    bool result = !hasAssignedPosqCharges;
    hasAssignedPosqCharges = true;
    return result;
}

const vector<string>& MetalContext::getEnergyParamDerivNames() const {
    return energyParamDerivNames;
}

map<string, double>& MetalContext::getEnergyParamDerivWorkspace() {
    return energyParamDerivWorkspace;
}

void MetalContext::addEnergyParameterDerivative(const string& param) {
    if (find(energyParamDerivNames.begin(), energyParamDerivNames.end(), param) != energyParamDerivNames.end())
        return;
    energyParamDerivNames.push_back(param);
    size_t size = energyParamDerivNames.size()*energyBuffer->getSize();
    energyParamDerivBuffer->resize(max<size_t>(1, size));
    clearBuffer(*energyParamDerivBuffer);
    updatePinnedBufferSize();
}

void MetalContext::flushQueue() {
    // Every Metal operation in this backend commits its command buffer before
    // returning.  Checking the queue is therefore sufficient for flush
    // semantics; completion remains asynchronous.
    getCurrentMetalQueue().checkForErrors();
}

void MetalContext::updatePinnedBufferSize() {
    size_t bytes = 1;
    const MetalArray* arrays[] = {positions.get(), velocities.get(), forces.get(), inverseMasses.get(),
                                  energyBuffer.get(), energyParamDerivBuffer.get(), atomIndexDevice.get()};
    for (const MetalArray* array : arrays) {
        if (array != NULL && array->isInitialized())
            bytes = max(bytes, array->getSize()*static_cast<size_t>(array->getElementSize()));
    }
    pinnedBuffer.resize(bytes);
}

void MetalContext::createCheckpoint(ostream& stream) const {
    vector<MetalFloat4> positionData;
    vector<MetalFloat4> velocityData;
    vector<MetalFloat4> forceData;
    positions->download(positionData);
    velocities->download(velocityData);
    forces->download(forceData);
    positionData.resize(numAtoms);
    velocityData.resize(numAtoms);
    forceData.resize(numAtoms);
    for (int i = 0; i < numAtoms; i++) {
        positionData[i].w = 0.0f;
        velocityData[i].w = 0.0f;
    }
    writeValue(stream, checkpointMagic);
    writeValue(stream, checkpointVersion);
    writeValue(stream, numAtoms);
    writeValue(stream, time);
    writeValue(stream, stepCount);
    stream.write(reinterpret_cast<const char*>(boxVectors), sizeof(boxVectors));
    if (numAtoms > 0) {
        stream.write(reinterpret_cast<const char*>(positionData.data()), positionData.size()*sizeof(MetalFloat4));
        stream.write(reinterpret_cast<const char*>(velocityData.data()), velocityData.size()*sizeof(MetalFloat4));
        stream.write(reinterpret_cast<const char*>(forceData.data()), forceData.size()*sizeof(MetalFloat4));
    }
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
    if (particleCount != numAtoms)
        throw OpenMMException("The Metal checkpoint contains a different number of particles");
    readValue(stream, time);
    readValue(stream, stepCount);
    stream.read(reinterpret_cast<char*>(boxVectors), sizeof(boxVectors));
    vector<MetalFloat4> positionData(paddedNumAtoms, {0.0f, 0.0f, 0.0f, 0.0f});
    vector<MetalFloat4> velocityData(paddedNumAtoms, {0.0f, 0.0f, 0.0f, 0.0f});
    vector<MetalFloat4> forceData(paddedNumAtoms, {0.0f, 0.0f, 0.0f, 0.0f});
    if (numAtoms > 0) {
        stream.read(reinterpret_cast<char*>(positionData.data()), numAtoms*sizeof(MetalFloat4));
        stream.read(reinterpret_cast<char*>(velocityData.data()), numAtoms*sizeof(MetalFloat4));
        stream.read(reinterpret_cast<char*>(forceData.data()), numAtoms*sizeof(MetalFloat4));
    }
    if (!stream)
        throw OpenMMException("Error reading a Metal checkpoint");
    for (int i = 0; i < numAtoms; i++) {
        // Charges and inverse masses are System metadata, not checkpointed
        // state.  This also keeps version-1 checkpoints written by the old
        // vertical slice (whose w components were zero) compatible.
        positionData[i].w = charges[i];
        velocityData[i].w = inverseMassValues[i];
    }
    positions->upload(positionData);
    velocities->upload(velocityData);
    forces->upload(forceData);
}
