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
#include "openmm/common/ComputeContext.h"
#include <iosfwd>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace OpenMM {

/** Four single precision values with the layout used by MSL float4. */
struct MetalFloat4 {
    float x, y, z, w;
};

/**
 * Minimal ComputeContext implementation for the native Metal backend.
 *
 * The initial implementation provides the core runtime objects and standard
 * single-precision state buffers needed by the existing Metal vertical slice.
 * Common sorting, FFT, and generated-kernel utilities remain explicit future
 * work and report a clear error when requested.
 */
class MetalContext : public ComputeContext {
public:
    MetalContext(const System& system, ContextImpl* contextImpl, size_t deviceIndex);
    ~MetalContext() override;

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
    void clearAutoclearBuffers();

    void advanceTime(double stepSize);

    int getNumContexts() const override;
    int getContextIndex() const override;
    std::vector<ComputeContext*> getAllContexts() override;
    ContextImpl* getContextImpl() override;
    double& getEnergyWorkspace() override;
    ComputeQueue createQueue() override;
    MetalArray* createArray() override;
    ComputeEvent createEvent() override;
    ComputeSort createSort(ComputeSortImpl::SortTrait* trait, unsigned int length, bool uniform=true) override;
    ComputeProgram compileProgram(const std::string source,
            const std::map<std::string, std::string>& defines=std::map<std::string, std::string>()) override;
    int computeThreadBlockSize(double memory) const override;
    void clearBuffer(ArrayInterface& array) override;
    void addAutoclearBuffer(ArrayInterface& array) override;
    bool getIsCPU() const override;
    int getSIMDWidth() const override;
    bool getSupports64BitGlobalAtomics() const override;
    bool getSupportsDoublePrecision() const override;
    bool getUseDoublePrecision() const override;
    bool getUseMixedPrecision() const override;
    int getNumAtomBlocks() const override;
    int getNumThreadBlocks() const override;
    int getMaxThreadBlockSize() const override;
    MetalArray& getPosq() override;
    ArrayInterface& getPosqCorrection() override;
    MetalArray& getVelm() override;
    MetalArray& getForceBuffers() override;
    MetalArray& getFloatForceBuffer() override;
    ArrayInterface& getLongForceBuffer() override;
    MetalArray& getEnergyBuffer() override;
    MetalArray& getEnergyParamDerivBuffer() override;
    void* getPinnedBuffer() override;
    ThreadPool& getThreadPool() override;
    MetalArray& getAtomIndexArray() override;
    bool getBoxIsTriclinic() const override;
    void getPeriodicBoxVectors(Vec3& a, Vec3& b, Vec3& c) const override;
    void setPeriodicBoxVectors(const Vec3& a, const Vec3& b, const Vec3& c) override;
    IntegrationUtilities& getIntegrationUtilities() override;
    ExpressionUtilities& getExpressionUtilities() override;
    BondedUtilities& getBondedUtilities() override;
    NonbondedUtilities& getNonbondedUtilities() override;
    NonbondedUtilities* createNonbondedUtilities() override;
    FFT3D createFFT(int xsize, int ysize, int zsize, bool realToComplex=false) override;
    void initializeContexts() override;
    void setCharges(const std::vector<double>& charges) override;
    bool requestPosqCharges() override;
    const std::vector<std::string>& getEnergyParamDerivNames() const override;
    std::map<std::string, double>& getEnergyParamDerivWorkspace() override;
    void addEnergyParameterDerivative(const std::string& param) override;
    void flushQueue() override;

    void createCheckpoint(std::ostream& stream) const;
    void loadCheckpoint(std::istream& stream);

private:
    MetalQueue& getCurrentMetalQueue();
    MetalArray& unwrap(ArrayInterface& array) const;
    void updatePinnedBufferSize();

    std::shared_ptr<MetalQueue> queue;
    std::unique_ptr<MetalArray> positions;
    std::unique_ptr<MetalArray> velocities;
    std::unique_ptr<MetalArray> forces;
    std::unique_ptr<MetalArray> inverseMasses;
    std::unique_ptr<MetalArray> energyBuffer;
    std::unique_ptr<MetalArray> energyParamDerivBuffer;
    std::unique_ptr<MetalArray> atomIndexDevice;
    std::unique_ptr<ThreadPool> threadPool;
    ContextImpl* contextImpl;
    int numAtomBlocks;
    int numThreadBlocks;
    double energyWorkspace;
    bool hasAssignedPosqCharges;
    Vec3 boxVectors[3];
    std::vector<float> charges;
    std::vector<float> inverseMassValues;
    std::vector<unsigned char> pinnedBuffer;
    std::vector<ArrayInterface*> autoclearBuffers;
    std::vector<std::string> energyParamDerivNames;
    std::map<std::string, double> energyParamDerivWorkspace;
};

} // namespace OpenMM

#endif /*OPENMM_METALCONTEXT_H_*/
