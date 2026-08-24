#include "MetalArray.h"
#include "MetalInternal.h"

#include <limits>
#include <utility>

using namespace OpenMM;
using namespace OpenMM::detail;
using namespace std;

namespace {

size_t checkedByteCount(size_t elements, int elementSize, const string& operation) {
    if (elementSize <= 0)
        throw OpenMMException(operation+": element size must be positive");
    if (elements > numeric_limits<size_t>::max()/static_cast<size_t>(elementSize))
        throw OpenMMException(operation+": buffer size overflow");
    return elements*static_cast<size_t>(elementSize);
}

void validateRange(size_t size, int offset, int elements, const string& operation) {
    if (offset < 0 || elements < 0 || static_cast<size_t>(offset) > size ||
            static_cast<size_t>(elements) > size-static_cast<size_t>(offset))
        throw OpenMMException(operation+": requested range exceeds the array");
}

} // namespace

struct MetalArray::Impl {
    ~Impl() {
        if (buffer != nullptr)
            OMMMetalBufferRelease(buffer);
    }

    shared_ptr<MetalQueueState> queue;
    OMMMetalBufferRef buffer = nullptr;
    ComputeContext* context = nullptr;
    size_t size = 0;
    int elementSize = 0;
    string name;
};

MetalArray::MetalArray() : impl(new Impl()) {
}

MetalArray::MetalArray(MetalQueue& queue, size_t size, int elementSize, const string& name) : impl(new Impl()) {
    initialize(queue, size, elementSize, name);
}

MetalArray::~MetalArray() {
}

void MetalArray::initialize(ComputeContext& context, size_t size, int elementSize, const string& name) {
    (void) context;
    (void) size;
    (void) elementSize;
    (void) name;
    throw OpenMMException("MetalArray::initialize(ComputeContext&) requires a Metal queue; use the Metal context integration overload");
}

void MetalArray::initialize(MetalQueue& queue, size_t size, int elementSize, const string& name) {
    initializeImpl(nullptr, queue, size, elementSize, name);
}

void MetalArray::initialize(ComputeContext& context, MetalQueue& queue, size_t size, int elementSize, const string& name) {
    initializeImpl(&context, queue, size, elementSize, name);
}

void MetalArray::initializeImpl(ComputeContext* context, MetalQueue& queue, size_t size,
                                int elementSize, const string& name) {
    if (isInitialized())
        throw OpenMMException("MetalArray has already been initialized");
    size_t bytes = checkedByteCount(size, elementSize, "MetalArray initialization");
    string storedName(name);
    queue.state->checkForErrors();

    OMMMetalBufferRef buffer = nullptr;
    OMMMetalErrorC error = {};
    {
        lock_guard<mutex> lock(queue.state->submissionMutex);
        int32_t status = OMMMetalBufferCreate(queue.state->queue, bytes,
                                               name.data(), name.size(), &buffer, &error);
        checkMetalBridgeResult(status, error, "Error allocating Metal array "+name);
    }
    if (buffer == nullptr)
        throw OpenMMException("Error allocating Metal array "+name+": Swift bridge returned a null buffer");

    impl->queue = queue.state;
    impl->buffer = buffer;
    impl->context = context;
    impl->size = size;
    impl->elementSize = elementSize;
    impl->name = std::move(storedName);
}

void MetalArray::resize(size_t size) {
    if (!isInitialized())
        throw OpenMMException("MetalArray has not been initialized");
    size_t bytes = checkedByteCount(size, impl->elementSize, "MetalArray resize");
    impl->queue->checkForErrors();

    OMMMetalErrorC error = {};
    lock_guard<mutex> lock(impl->queue->submissionMutex);
    int32_t status = OMMMetalBufferResize(impl->buffer, bytes, &error);
    checkMetalBridgeResult(status, error, "Error resizing Metal array "+impl->name);
    impl->size = size;
}

bool MetalArray::isInitialized() const {
    return impl->queue != nullptr && impl->buffer != nullptr;
}

size_t MetalArray::getSize() const {
    return impl->size;
}

int MetalArray::getElementSize() const {
    return impl->elementSize;
}

const string& MetalArray::getName() const {
    return impl->name;
}

ComputeContext& MetalArray::getContext() {
    if (impl->context == nullptr)
        throw OpenMMException("Standalone MetalArray has no OpenMM ComputeContext");
    return *impl->context;
}

void MetalArray::upload(const void* data, bool blocking) {
    uploadSubArray(data, 0, static_cast<int>(getSize()), blocking);
}

void MetalArray::uploadSubArray(const void* data, int offset, int elements, bool blocking) {
    if (!isInitialized())
        throw OpenMMException("MetalArray has not been initialized");
    validateRange(impl->size, offset, elements, "MetalArray::uploadSubArray");
    size_t bytes = checkedByteCount(static_cast<size_t>(elements), impl->elementSize, "MetalArray upload");
    if (bytes == 0)
        return;
    if (data == nullptr)
        throw OpenMMException("MetalArray upload source is null");
    impl->queue->checkForErrors();

    size_t byteOffset = static_cast<size_t>(offset)*static_cast<size_t>(impl->elementSize);
    OMMMetalErrorC error = {};
    lock_guard<mutex> lock(impl->queue->submissionMutex);
    int32_t status = OMMMetalBufferUpload(impl->buffer, byteOffset, data, bytes,
                                          blocking ? 1 : 0, &error);
    checkMetalBridgeResult(status, error, "Error uploading Metal array "+impl->name);
}

void MetalArray::download(void* data, bool blocking) const {
    downloadSubArray(data, 0, static_cast<int>(getSize()), blocking);
}

void MetalArray::downloadSubArray(void* data, int offset, int elements, bool blocking) const {
    if (!isInitialized())
        throw OpenMMException("MetalArray has not been initialized");
    validateRange(impl->size, offset, elements, "MetalArray::downloadSubArray");
    size_t bytes = checkedByteCount(static_cast<size_t>(elements), impl->elementSize, "MetalArray download");
    if (bytes == 0)
        return;
    if (data == nullptr)
        throw OpenMMException("MetalArray download destination is null");
    impl->queue->checkForErrors();

    size_t byteOffset = static_cast<size_t>(offset)*static_cast<size_t>(impl->elementSize);
    OMMMetalErrorC error = {};
    lock_guard<mutex> lock(impl->queue->submissionMutex);
    int32_t status = OMMMetalBufferDownload(impl->buffer, byteOffset, data, bytes,
                                            blocking ? 1 : 0, &error);
    checkMetalBridgeResult(status, error, "Error downloading Metal array "+impl->name);
}

void MetalArray::copyTo(ArrayInterface& destination) const {
    MetalArray* metalDestination = dynamic_cast<MetalArray*>(&destination);
    if (metalDestination == nullptr)
        throw OpenMMException("Cannot copy Metal array "+impl->name+" to a non-Metal array");
    if (metalDestination->getSize() != impl->size || metalDestination->getElementSize() != impl->elementSize)
        throw OpenMMException("Error copying Metal array "+impl->name+" to "+destination.getName()+": incompatible size or element type");
    copySubArrayTo(*metalDestination, 0, 0, static_cast<int>(impl->size));
}

void MetalArray::copySubArrayTo(MetalArray& destination, int sourceOffset,
                                int destinationOffset, int elements) const {
    if (!isInitialized() || !destination.isInitialized())
        throw OpenMMException("Cannot copy an uninitialized Metal array");
    if (impl->elementSize != destination.impl->elementSize)
        throw OpenMMException("Cannot copy between Metal arrays with different element sizes");
    validateRange(impl->size, sourceOffset, elements, "MetalArray source copy");
    validateRange(destination.impl->size, destinationOffset, elements, "MetalArray destination copy");
    if (impl->queue.get() != destination.impl->queue.get())
        throw OpenMMException("Copying between different Metal command queues is not yet supported");
    size_t bytes = checkedByteCount(static_cast<size_t>(elements), impl->elementSize, "MetalArray copy");
    if (bytes == 0)
        return;
    impl->queue->checkForErrors();

    size_t sourceByteOffset = static_cast<size_t>(sourceOffset)*static_cast<size_t>(impl->elementSize);
    size_t destinationByteOffset = static_cast<size_t>(destinationOffset)*static_cast<size_t>(impl->elementSize);
    OMMMetalErrorC error = {};
    lock_guard<mutex> lock(impl->queue->submissionMutex);
    int32_t status = OMMMetalBufferCopy(impl->buffer, sourceByteOffset,
                                        destination.impl->buffer, destinationByteOffset,
                                        bytes, &error);
    checkMetalBridgeResult(status, error, "Error copying Metal array "+impl->name);
}

shared_ptr<MetalQueueState> MetalArrayAccess::getQueueState(const MetalArray& array) {
    return array.impl->queue;
}

OMMMetalBufferRef MetalArrayAccess::getBuffer(const MetalArray& array) {
    return array.impl->buffer;
}
