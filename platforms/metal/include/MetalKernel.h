#ifndef OPENMM_METALKERNEL_H_
#define OPENMM_METALKERNEL_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2026 Stanford University and the Authors.           *
 * -------------------------------------------------------------------------- */

#include "openmm/common/ComputeKernel.h"
#include "windowsExportMetal.h"
#include <memory>
#include <string>

namespace OpenMM {

class MetalProgram;

/** Selects how sequential ComputeKernel arguments are bound to an MSL kernel. */
enum class MetalBindingMode {
    Direct,
    ArgumentBuffer
};

/** Native Metal implementation of ComputeKernelImpl. */
class OPENMM_EXPORT_METAL MetalKernel : public ComputeKernelImpl {
public:
    ~MetalKernel();

    std::string getName() const override;
    int getMaxBlockSize() const override;
    void execute(int threads, int blockSize = -1) override;

protected:
    void addArrayArg(ArrayInterface& value) override;
    void addPrimitiveArg(const void* value, int size) override;
    void addEmptyArg() override;
    void setArrayArg(int index, ArrayInterface& value) override;
    void setPrimitiveArg(int index, const void* value, int size) override;

private:
    struct Impl;
    explicit MetalKernel(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl;

    friend class MetalProgram;
};

} // namespace OpenMM

#endif /*OPENMM_METALKERNEL_H_*/
