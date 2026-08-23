#ifndef OPENMM_METALARRAY_H_
#define OPENMM_METALARRAY_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2026 Stanford University and the Authors.           *
 * -------------------------------------------------------------------------- */

#include "MetalQueue.h"
#include "openmm/common/ArrayInterface.h"
#include <memory>
#include <string>
#include <vector>

namespace OpenMM {

namespace detail {
class MetalArrayAccess;
}

/** A private-storage Metal buffer with staged host transfers. */
class OPENMM_EXPORT_METAL MetalArray : public ArrayInterface {
public:
    template <class T>
    static MetalArray* create(MetalQueue& queue, size_t size, const std::string& name) {
        return new MetalArray(queue, size, sizeof(T), name);
    }

    MetalArray();
    MetalArray(MetalQueue& queue, size_t size, int elementSize, const std::string& name);
    ~MetalArray();

    MetalArray(const MetalArray&) = delete;
    MetalArray& operator=(const MetalArray&) = delete;

    /**
     * ArrayInterface hook.  Standalone arrays do not know how to obtain a
     * MetalQueue from an arbitrary ComputeContext, so this overload throws.
     * A Metal ComputeContext should call the overload accepting both objects.
     */
    void initialize(ComputeContext& context, size_t size, int elementSize, const std::string& name) override;
    void initialize(MetalQueue& queue, size_t size, int elementSize, const std::string& name);
    void initialize(ComputeContext& context, MetalQueue& queue, size_t size, int elementSize, const std::string& name);

    void resize(size_t size) override;
    bool isInitialized() const override;
    size_t getSize() const override;
    int getElementSize() const override;
    const std::string& getName() const override;
    ComputeContext& getContext() override;

    using ArrayInterface::download;
    using ArrayInterface::upload;

    template <class T>
    void upload(const std::vector<T>& data, bool convert = false) {
        ArrayInterface::upload(data, convert);
    }
    template <class T>
    void download(std::vector<T>& data, bool convert = false) const {
        ArrayInterface::download(data, convert);
    }

    void upload(const void* data, bool blocking = true) override;
    void uploadSubArray(const void* data, int offset, int elements, bool blocking = true) override;
    void download(void* data, bool blocking = true) const override;
    void downloadSubArray(void* data, int offset, int elements, bool blocking = true) const;
    void copyTo(ArrayInterface& dest) const override;
    void copySubArrayTo(MetalArray& dest, int sourceOffset, int destOffset, int elements) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;

    void initializeImpl(ComputeContext* context, MetalQueue& queue, size_t size, int elementSize, const std::string& name);

    friend class detail::MetalArrayAccess;
    friend class MetalKernel;
};

} // namespace OpenMM

#endif /*OPENMM_METALARRAY_H_*/
