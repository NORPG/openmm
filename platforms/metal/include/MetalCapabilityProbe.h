#ifndef OPENMM_METALCAPABILITYPROBE_H_
#define OPENMM_METALCAPABILITYPROBE_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * Portions copyright (c) 2026 Stanford University and the Authors.           *
 * -------------------------------------------------------------------------- */

#include "windowsExportMetal.h"
#include <string>

namespace OpenMM {

/** Result of an operation-specific runtime compilation and GPU execution probe. */
struct OPENMM_EXPORT_METAL MetalCapabilityProbeResult {
    enum class Status {
        NotRun,
        Supported,
        InitializationFailed,
        CompilationFailed,
        PipelineCreationFailed,
        ExecutionFailed,
        ValidationFailed
    };

    Status status = Status::NotRun;
    std::string diagnostic;

    bool isSupported() const {
        return status == Status::Supported;
    }
    const char* getStatusName() const;
};

} // namespace OpenMM

#endif /*OPENMM_METALCAPABILITYPROBE_H_*/
