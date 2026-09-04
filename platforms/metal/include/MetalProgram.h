#ifndef OPENMM_METALPROGRAM_H_
#define OPENMM_METALPROGRAM_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2026 Stanford University and the Authors.           *
 * -------------------------------------------------------------------------- */

#include "MetalKernel.h"
#include "MetalQueue.h"
#include "openmm/common/ComputeProgram.h"
#include <memory>
#include <string>

namespace OpenMM {

struct MetalProgramOptions {
    bool fastMathEnabled = true;
    /** MSL 3.0 is the portable baseline for the initial macOS 13 backend. */
    int languageVersionMajor = 3;
    int languageVersionMinor = 0;
};

/** A Metal library compiled from MSL source at runtime. */
class OPENMM_EXPORT_METAL MetalProgram : public ComputeProgramImpl {
public:
    MetalProgram(MetalQueue& queue, const std::string& source, const MetalProgramOptions& options = MetalProgramOptions());
    ~MetalProgram();

    MetalProgram(const MetalProgram&) = delete;
    MetalProgram& operator=(const MetalProgram&) = delete;

    /** ComputeProgram interface; creates a direct-slot kernel. */
    ComputeKernel createKernel(const std::string& name) override;

    /** Create a native kernel, optionally using one MSL argument buffer. */
    std::shared_ptr<MetalKernel> createMetalKernel(const std::string& name,
                                                   MetalBindingMode bindingMode = MetalBindingMode::Direct,
                                                   int argumentBufferIndex = 0);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace OpenMM

#endif /*OPENMM_METALPROGRAM_H_*/
