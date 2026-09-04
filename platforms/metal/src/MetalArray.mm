#include "MetalArray.h"
#include "MetalInternal.h"

#include <cstring>
#include <limits>
#include <sstream>

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

id<MTLBuffer> createPrivateBuffer(const shared_ptr<MetalQueueState>& queue, size_t bytes, const string& name) {
    id<MTLBuffer> buffer = [queue->device newBufferWithLength:max<size_t>(bytes, 1)
                                                    options:MTLResourceStorageModePrivate];
    if (buffer == nil) {
        stringstream message;
        message << "Error allocating Metal array " << name << " (" << bytes << " bytes)";
        throw OpenMMException(message.str());
    }
    NSString* label = [[NSString alloc] initWithBytes:name.data() length:name.size() encoding:NSUTF8StringEncoding];
    buffer.label = label;
    return buffer;
}

id<MTLBuffer> createStagingBuffer(const shared_ptr<MetalQueueState>& queue, size_t bytes, const string& operation) {
    id<MTLBuffer> buffer = [queue->device newBufferWithLength:max<size_t>(bytes, 1)
                                                    options:MTLResourceStorageModeShared];
    if (buffer == nil) {
        stringstream message;
        message << "Error allocating Metal staging buffer for " << operation << " (" << bytes << " bytes)";
        throw OpenMMException(message.str());
    }
    return buffer;
}

} // namespace

struct MetalArray::Impl {
    shared_ptr<MetalQueueState> queue;
    id<MTLBuffer> buffer = nil;
    ComputeContext* context = NULL;
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
    initializeImpl(NULL, queue, size, elementSize, name);
}

void MetalArray::initialize(ComputeContext& context, MetalQueue& queue, size_t size, int elementSize, const string& name) {
    initializeImpl(&context, queue, size, elementSize, name);
}

void MetalArray::initializeImpl(ComputeContext* context, MetalQueue& queue, size_t size, int elementSize, const string& name) {
    @autoreleasepool {
        if (isInitialized())
            throw OpenMMException("MetalArray has already been initialized");
        size_t bytes = checkedByteCount(size, elementSize, "MetalArray initialization");
        queue.state->checkForErrors();
        impl->queue = queue.state;
        impl->buffer = createPrivateBuffer(impl->queue, bytes, name);
        impl->context = context;
        impl->size = size;
        impl->elementSize = elementSize;
        impl->name = name;
    }
}

void MetalArray::resize(size_t size) {
    @autoreleasepool {
        if (!isInitialized())
            throw OpenMMException("MetalArray has not been initialized");
        size_t bytes = checkedByteCount(size, impl->elementSize, "MetalArray resize");
        impl->queue->checkForErrors();
        id<MTLBuffer> replacement = createPrivateBuffer(impl->queue, bytes, impl->name);
        impl->buffer = replacement;
        impl->size = size;
    }
}

bool MetalArray::isInitialized() const {
    return impl->queue != NULL && impl->buffer != nil;
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
    if (impl->context == NULL)
        throw OpenMMException("Standalone MetalArray has no OpenMM ComputeContext");
    return *impl->context;
}

void MetalArray::upload(const void* data, bool blocking) {
    uploadSubArray(data, 0, static_cast<int>(getSize()), blocking);
}

void MetalArray::uploadSubArray(const void* data, int offset, int elements, bool blocking) {
    @autoreleasepool {
        if (!isInitialized())
            throw OpenMMException("MetalArray has not been initialized");
        validateRange(impl->size, offset, elements, "MetalArray::uploadSubArray");
        size_t bytes = checkedByteCount(static_cast<size_t>(elements), impl->elementSize, "MetalArray upload");
        if (bytes == 0)
            return;
        if (data == NULL)
            throw OpenMMException("MetalArray upload source is null");
        impl->queue->checkForErrors();
        id<MTLBuffer> staging = createStagingBuffer(impl->queue, bytes, "uploading "+impl->name);
        memcpy(staging.contents, data, bytes);

        lock_guard<mutex> lock(impl->queue->submissionMutex);
        id<MTLCommandBuffer> command = impl->queue->makeCommandBuffer("upload "+impl->name);
        id<MTLBlitCommandEncoder> encoder = [command blitCommandEncoder];
        if (encoder == nil)
            throw OpenMMException("Metal failed to create a blit encoder for uploading "+impl->name);
        [encoder copyFromBuffer:staging
                  sourceOffset:0
                      toBuffer:impl->buffer
             destinationOffset:static_cast<NSUInteger>(offset)*impl->elementSize
                          size:bytes];
        [encoder endEncoding];
        impl->queue->submit(command, blocking);
    }
}

void MetalArray::download(void* data, bool blocking) const {
    downloadSubArray(data, 0, static_cast<int>(getSize()), blocking);
}

void MetalArray::downloadSubArray(void* data, int offset, int elements, bool blocking) const {
    @autoreleasepool {
        if (!isInitialized())
            throw OpenMMException("MetalArray has not been initialized");
        validateRange(impl->size, offset, elements, "MetalArray::downloadSubArray");
        size_t bytes = checkedByteCount(static_cast<size_t>(elements), impl->elementSize, "MetalArray download");
        if (bytes == 0)
            return;
        if (data == NULL)
            throw OpenMMException("MetalArray download destination is null");
        impl->queue->checkForErrors();
        id<MTLBuffer> staging = createStagingBuffer(impl->queue, bytes, "downloading "+impl->name);

        lock_guard<mutex> lock(impl->queue->submissionMutex);
        id<MTLCommandBuffer> command = impl->queue->makeCommandBuffer("download "+impl->name);
        id<MTLBlitCommandEncoder> encoder = [command blitCommandEncoder];
        if (encoder == nil)
            throw OpenMMException("Metal failed to create a blit encoder for downloading "+impl->name);
        [encoder copyFromBuffer:impl->buffer
                  sourceOffset:static_cast<NSUInteger>(offset)*impl->elementSize
                      toBuffer:staging
             destinationOffset:0
                          size:bytes];
        [encoder endEncoding];
        impl->queue->submit(command, blocking, [staging, data, bytes](bool succeeded) {
            if (succeeded)
                memcpy(data, staging.contents, bytes);
        });
    }
}

void MetalArray::copyTo(ArrayInterface& destination) const {
    MetalArray* metalDestination = dynamic_cast<MetalArray*>(&destination);
    if (metalDestination == NULL)
        throw OpenMMException("Cannot copy Metal array "+impl->name+" to a non-Metal array");
    if (metalDestination->getSize() != impl->size || metalDestination->getElementSize() != impl->elementSize)
        throw OpenMMException("Error copying Metal array "+impl->name+" to "+destination.getName()+": incompatible size or element type");
    copySubArrayTo(*metalDestination, 0, 0, static_cast<int>(impl->size));
}

void MetalArray::copySubArrayTo(MetalArray& destination, int sourceOffset, int destinationOffset, int elements) const {
    @autoreleasepool {
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

        lock_guard<mutex> lock(impl->queue->submissionMutex);
        id<MTLCommandBuffer> command = impl->queue->makeCommandBuffer("copy "+impl->name+" to "+destination.impl->name);
        id<MTLBlitCommandEncoder> encoder = [command blitCommandEncoder];
        if (encoder == nil)
            throw OpenMMException("Metal failed to create a blit encoder for array copy");
        [encoder copyFromBuffer:impl->buffer
                  sourceOffset:static_cast<NSUInteger>(sourceOffset)*impl->elementSize
                      toBuffer:destination.impl->buffer
             destinationOffset:static_cast<NSUInteger>(destinationOffset)*impl->elementSize
                          size:bytes];
        [encoder endEncoding];
        impl->queue->submit(command, false);
    }
}

shared_ptr<MetalQueueState> MetalArrayAccess::getQueueState(const MetalArray& array) {
    return array.impl->queue;
}

id<MTLBuffer> MetalArrayAccess::getBuffer(const MetalArray& array) {
    return array.impl->buffer;
}
