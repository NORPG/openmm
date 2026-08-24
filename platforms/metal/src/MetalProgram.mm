#include "MetalProgram.h"
#include "MetalArray.h"
#include "MetalInternal.h"

#import <dispatch/dispatch.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <sstream>
#include <utility>
#include <vector>

using namespace OpenMM;
using namespace OpenMM::detail;
using namespace std;

namespace {

NSString* makeNSString(const string& value, const string& description) {
    NSString* result = [[NSString alloc] initWithBytes:value.data()
                                                length:value.size()
                                              encoding:NSUTF8StringEncoding];
    if (result == nil)
        throw OpenMMException(description+" is not valid UTF-8");
    return result;
}

struct KernelArgument {
    enum class Type {
        Empty,
        Array,
        Primitive
    };

    Type type = Type::Empty;
    shared_ptr<MetalBufferState> buffer;
    vector<unsigned char> primitive;
};

struct ArgumentBufferSlot {
    enum class Kind {
        Buffer,
        Constant,
        Unsupported
    };

    Kind kind = Kind::Unsupported;
    size_t byteCount = 0;
    MTLDataType dataType = MTLDataTypeNone;
};

/** Return the in-argument-buffer storage size for non-aggregate MSL values. */
bool argumentConstantSize(MTLDataType type, size_t& byteCount) {
    switch (type) {
        case MTLDataTypeFloat:
        case MTLDataTypeInt:
        case MTLDataTypeUInt:
            byteCount = 4;
            return true;
        case MTLDataTypeFloat2:
        case MTLDataTypeInt2:
        case MTLDataTypeUInt2:
            byteCount = 8;
            return true;
        case MTLDataTypeFloat3:
        case MTLDataTypeFloat4:
        case MTLDataTypeInt3:
        case MTLDataTypeInt4:
        case MTLDataTypeUInt3:
        case MTLDataTypeUInt4:
            byteCount = 16;
            return true;
        case MTLDataTypeHalf:
        case MTLDataTypeShort:
        case MTLDataTypeUShort:
            byteCount = 2;
            return true;
        case MTLDataTypeHalf2:
        case MTLDataTypeShort2:
        case MTLDataTypeUShort2:
            byteCount = 4;
            return true;
        case MTLDataTypeHalf3:
        case MTLDataTypeHalf4:
        case MTLDataTypeShort3:
        case MTLDataTypeShort4:
        case MTLDataTypeUShort3:
        case MTLDataTypeUShort4:
            byteCount = 8;
            return true;
        case MTLDataTypeChar:
        case MTLDataTypeUChar:
        case MTLDataTypeBool:
            byteCount = 1;
            return true;
        case MTLDataTypeChar2:
        case MTLDataTypeUChar2:
        case MTLDataTypeBool2:
            byteCount = 2;
            return true;
        case MTLDataTypeChar3:
        case MTLDataTypeChar4:
        case MTLDataTypeUChar3:
        case MTLDataTypeUChar4:
        case MTLDataTypeBool3:
        case MTLDataTypeBool4:
            byteCount = 4;
            return true;
        case MTLDataTypeLong:
        case MTLDataTypeULong:
            byteCount = 8;
            return true;
        case MTLDataTypeLong2:
        case MTLDataTypeULong2:
            byteCount = 16;
            return true;
        case MTLDataTypeLong3:
        case MTLDataTypeLong4:
        case MTLDataTypeULong3:
        case MTLDataTypeULong4:
            byteCount = 32;
            return true;
        default:
            return false;
    }
}

} // namespace

struct MetalProgram::Impl {
    shared_ptr<MetalQueueState> queue;
    id<MTLLibrary> library = nil;
};

struct MetalKernel::Impl {
    Impl(const shared_ptr<MetalQueueState>& queue, id<MTLFunction> function,
         id<MTLComputePipelineState> pipeline, MetalBindingMode bindingMode,
         int argumentBufferIndex, MTLComputePipelineReflection* reflection) :
         queue(queue), function(function), pipeline(pipeline),
         name(fromNSString(function.name)), bindingMode(bindingMode),
         argumentBufferIndex(argumentBufferIndex) {
        if (bindingMode == MetalBindingMode::ArgumentBuffer) {
            argumentEncoder = [function newArgumentEncoderWithBufferIndex:argumentBufferIndex];
            if (argumentEncoder == nil)
                throw OpenMMException("Kernel "+name+" has no argument buffer at buffer index "+to_string(argumentBufferIndex));

            id<MTLBufferBinding> reflectedBuffer = nil;
            for (id<MTLBinding> binding in reflection.bindings) {
                if (binding.index == static_cast<NSUInteger>(argumentBufferIndex) &&
                        binding.type == MTLBindingTypeBuffer &&
                        [binding conformsToProtocol:@protocol(MTLBufferBinding)]) {
                    id<MTLBufferBinding> candidate = static_cast<id<MTLBufferBinding>>(binding);
                    if (candidate.bufferPointerType.elementIsArgumentBuffer && candidate.bufferStructType != nil) {
                        reflectedBuffer = candidate;
                        break;
                    }
                }
            }
            if (reflectedBuffer == nil)
                throw OpenMMException("Kernel "+name+" has no reflected argument buffer at buffer index "+
                                      to_string(argumentBufferIndex));
            for (MTLStructMember* member in reflectedBuffer.bufferStructType.members) {
                ArgumentBufferSlot slot;
                slot.dataType = member.dataType;
                if (member.dataType == MTLDataTypePointer)
                    slot.kind = ArgumentBufferSlot::Kind::Buffer;
                else if (argumentConstantSize(member.dataType, slot.byteCount))
                    slot.kind = ArgumentBufferSlot::Kind::Constant;
                argumentBufferSlots[member.argumentIndex] = slot;
            }
        }
    }

    void validateBufferSlot(size_t index) const {
        if (bindingMode != MetalBindingMode::ArgumentBuffer)
            return;
        map<size_t, ArgumentBufferSlot>::const_iterator slot = argumentBufferSlots.find(index);
        if (slot == argumentBufferSlots.end())
            throw OpenMMException("Argument "+to_string(index)+
                                  " is not present in the reflected argument buffer for kernel "+name);
        if (slot->second.kind != ArgumentBufferSlot::Kind::Buffer)
            throw OpenMMException("Argument "+to_string(index)+
                                  " is not a buffer slot in the argument buffer for kernel "+name);
    }

    void validateConstantSlot(size_t index, size_t byteCount) const {
        if (bindingMode != MetalBindingMode::ArgumentBuffer)
            return;
        map<size_t, ArgumentBufferSlot>::const_iterator slot = argumentBufferSlots.find(index);
        if (slot == argumentBufferSlots.end())
            throw OpenMMException("Argument "+to_string(index)+
                                  " is not present in the reflected argument buffer for kernel "+name);
        if (slot->second.kind == ArgumentBufferSlot::Kind::Buffer)
            throw OpenMMException("Argument "+to_string(index)+
                                  " is a buffer slot, not a constant slot, in kernel "+name);
        if (slot->second.kind == ArgumentBufferSlot::Kind::Unsupported)
            throw OpenMMException("Argument "+to_string(index)+" for kernel "+name+
                                  " has unsupported reflected Metal data type "+
                                  to_string(static_cast<unsigned long>(slot->second.dataType)));
        if (slot->second.byteCount != byteCount)
            throw OpenMMException("Argument "+to_string(index)+" for kernel "+name+" requires "+
                                  to_string(slot->second.byteCount)+" bytes, not "+to_string(byteCount));
    }

    shared_ptr<MetalQueueState> queue;
    id<MTLFunction> function = nil;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLArgumentEncoder> argumentEncoder = nil;
    string name;
    MetalBindingMode bindingMode;
    int argumentBufferIndex;
    map<size_t, ArgumentBufferSlot> argumentBufferSlots;
    vector<KernelArgument> arguments;
    mutex argumentMutex;
};

MetalProgram::MetalProgram(MetalQueue& queue, const string& source, const MetalProgramOptions& options) : impl(new Impl()) {
    @autoreleasepool {
        if (source.empty())
            throw OpenMMException("Cannot compile an empty Metal program");
        if (options.languageVersionMajor < 0 || options.languageVersionMajor > 0xffff ||
                options.languageVersionMinor < 0 || options.languageVersionMinor > 0xffff)
            throw OpenMMException("Invalid Metal language version");
        queue.state->checkForErrors();
        impl->queue = queue.state;

        NSString* nativeSource = makeNSString(source, "Metal program source");
        MTLCompileOptions* compileOptions = [[MTLCompileOptions alloc] init];
        compileOptions.fastMathEnabled = options.fastMathEnabled;
        if (options.languageVersionMajor != 0) {
            NSUInteger encodedVersion = (static_cast<NSUInteger>(options.languageVersionMajor) << 16) |
                                        static_cast<NSUInteger>(options.languageVersionMinor);
            compileOptions.languageVersion = static_cast<MTLLanguageVersion>(encodedVersion);
        }
        NSError* error = nil;
        impl->library = [impl->queue->device newLibraryWithSource:nativeSource options:compileOptions error:&error];
        if (impl->library == nil)
            throw OpenMMException("Error compiling Metal program: "+describeError(error));
    }
}

MetalProgram::MetalProgram(MetalQueue& queue, const void* libraryData, size_t librarySize) : impl(new Impl()) {
    @autoreleasepool {
        if (libraryData == NULL || librarySize == 0)
            throw OpenMMException("Cannot load an empty Metal library");
        queue.state->checkForErrors();
        impl->queue = queue.state;
        dispatch_data_t data = dispatch_data_create(libraryData, librarySize, NULL,
                                                    DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        if (data == NULL)
            throw OpenMMException("Unable to copy the precompiled Metal library data");
        NSError* error = nil;
        impl->library = [impl->queue->device newLibraryWithData:data error:&error];
        if (impl->library == nil)
            throw OpenMMException("Error loading precompiled Metal library: "+describeError(error));
    }
}

MetalProgram::~MetalProgram() {
}

ComputeKernel MetalProgram::createKernel(const string& name) {
    return static_pointer_cast<ComputeKernelImpl>(createMetalKernel(name));
}

shared_ptr<MetalKernel> MetalProgram::createMetalKernel(const string& name,
                                                        MetalBindingMode bindingMode,
                                                        int argumentBufferIndex) {
    @autoreleasepool {
        if (argumentBufferIndex < 0)
            throw OpenMMException("Metal argument buffer index must not be negative");
        if (bindingMode != MetalBindingMode::Direct && bindingMode != MetalBindingMode::ArgumentBuffer)
            throw OpenMMException("Unknown Metal kernel binding mode");
        NSString* nativeName = makeNSString(name, "Metal kernel name");
        id<MTLFunction> function = [impl->library newFunctionWithName:nativeName];
        if (function == nil)
            throw OpenMMException("Metal program does not contain a kernel named '"+name+"'");

        NSError* error = nil;
        MTLComputePipelineReflection* reflection = nil;
        id<MTLComputePipelineState> pipeline;
        if (bindingMode == MetalBindingMode::ArgumentBuffer) {
            pipeline = [impl->queue->device newComputePipelineStateWithFunction:function
                                                                          options:(MTLPipelineOptionBindingInfo |
                                                                                   MTLPipelineOptionBufferTypeInfo)
                                                                       reflection:&reflection
                                                                            error:&error];
        }
        else
            pipeline = [impl->queue->device newComputePipelineStateWithFunction:function error:&error];
        if (pipeline == nil)
            throw OpenMMException("Error creating Metal pipeline for kernel '"+name+"': "+describeError(error));
        unique_ptr<MetalKernel::Impl> kernelImpl(new MetalKernel::Impl(impl->queue, function, pipeline,
                                                                       bindingMode, argumentBufferIndex, reflection));
        return shared_ptr<MetalKernel>(new MetalKernel(std::move(kernelImpl)));
    }
}

MetalKernel::MetalKernel(unique_ptr<Impl> impl) : impl(std::move(impl)) {
}

MetalKernel::~MetalKernel() {
}

string MetalKernel::getName() const {
    return impl->name;
}

int MetalKernel::getMaxBlockSize() const {
    return static_cast<int>(impl->pipeline.maxTotalThreadsPerThreadgroup);
}

void MetalKernel::execute(int threads, int blockSize) {
    @autoreleasepool {
        if (threads < 0)
            throw OpenMMException("Cannot execute Metal kernel "+impl->name+" with a negative thread count");
        if (threads == 0)
            return;
        int maxBlockSize = getMaxBlockSize();
        if (blockSize == -1) {
            int width = max(1, static_cast<int>(impl->pipeline.threadExecutionWidth));
            blockSize = min(256, maxBlockSize);
            blockSize = max(width, (blockSize/width)*width);
            blockSize = min(blockSize, maxBlockSize);
        }
        if (blockSize <= 0 || blockSize > maxBlockSize) {
            stringstream message;
            message << "Invalid threadgroup size " << blockSize << " for Metal kernel " << impl->name
                    << " (maximum " << maxBlockSize << ")";
            throw OpenMMException(message.str());
        }

        lock_guard<mutex> argumentLock(impl->argumentMutex);
        for (size_t i = 0; i < impl->arguments.size(); i++) {
            if (impl->arguments[i].type == KernelArgument::Type::Empty)
                throw OpenMMException("Argument "+to_string(i)+" has not been set for Metal kernel "+impl->name);
        }
        impl->queue->checkForErrors();
        lock_guard<mutex> queueLock(impl->queue->submissionMutex);
        id<MTLCommandBuffer> command = impl->queue->makeCommandBuffer("execute "+impl->name);
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (encoder == nil)
            throw OpenMMException("Metal failed to create a compute encoder for kernel "+impl->name);
        encoder.label = makeNSString(impl->name, "Metal kernel name");
        [encoder setComputePipelineState:impl->pipeline];

        if (impl->bindingMode == MetalBindingMode::Direct) {
            for (size_t i = 0; i < impl->arguments.size(); i++) {
                const KernelArgument& argument = impl->arguments[i];
                if (argument.type == KernelArgument::Type::Array) {
                    id<MTLBuffer> buffer = argument.buffer->getBuffer();
                    [encoder setBuffer:buffer offset:0 atIndex:i];
                }
                else {
                    [encoder setBytes:argument.primitive.data() length:argument.primitive.size() atIndex:i];
                }
            }
        }
        else {
            NSUInteger encodedLength = max<NSUInteger>(impl->argumentEncoder.encodedLength, 1);
            id<MTLBuffer> argumentBuffer = [impl->queue->device newBufferWithLength:encodedLength
                                                                             options:MTLResourceStorageModeShared];
            if (argumentBuffer == nil)
                throw OpenMMException("Unable to allocate argument buffer for Metal kernel "+impl->name);
            argumentBuffer.label = @"OpenMM Metal encoded arguments";
            [impl->argumentEncoder setArgumentBuffer:argumentBuffer offset:0];
            for (size_t i = 0; i < impl->arguments.size(); i++) {
                const KernelArgument& argument = impl->arguments[i];
                if (argument.type == KernelArgument::Type::Array) {
                    id<MTLBuffer> buffer = argument.buffer->getBuffer();
                    [impl->argumentEncoder setBuffer:buffer offset:0 atIndex:i];
                    [encoder useResource:buffer usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
                }
                else {
                    void* destination = [impl->argumentEncoder constantDataAtIndex:i];
                    if (destination == NULL)
                        throw OpenMMException("Argument "+to_string(i)+" of Metal kernel "+impl->name+
                                              " is not inline data in its argument buffer");
                    uintptr_t baseAddress = reinterpret_cast<uintptr_t>(argumentBuffer.contents);
                    uintptr_t destinationAddress = reinterpret_cast<uintptr_t>(destination);
                    size_t encodedLength = impl->argumentEncoder.encodedLength;
                    if (destinationAddress < baseAddress ||
                            destinationAddress-baseAddress > encodedLength ||
                            argument.primitive.size() > encodedLength-(destinationAddress-baseAddress))
                        throw OpenMMException("Reflected constant slot "+to_string(i)+
                                              " is outside the argument buffer for kernel "+impl->name);
                    memcpy(destination, argument.primitive.data(), argument.primitive.size());
                }
            }
            [encoder setBuffer:argumentBuffer offset:0 atIndex:impl->argumentBufferIndex];
        }

        [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(threads), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(static_cast<NSUInteger>(blockSize), 1, 1)];
        [encoder endEncoding];
        impl->queue->submit(command, false);
    }
}

void MetalKernel::addArrayArg(ArrayInterface& value) {
    lock_guard<mutex> lock(impl->argumentMutex);
    MetalArray* array = dynamic_cast<MetalArray*>(&value);
    if (array == NULL)
        throw OpenMMException("Metal kernel arguments must be MetalArray objects");
    if (!array->isInitialized())
        throw OpenMMException("Cannot bind an uninitialized MetalArray to kernel "+impl->name);
    shared_ptr<MetalBufferState> buffer = MetalArrayAccess::getBufferState(*array);
    if (buffer->queue.get() != impl->queue.get())
        throw OpenMMException("Cannot bind a MetalArray from a different command queue to kernel "+impl->name);
    impl->validateBufferSlot(impl->arguments.size());
    KernelArgument argument;
    argument.type = KernelArgument::Type::Array;
    argument.buffer = buffer;
    impl->arguments.push_back(argument);
}

void MetalKernel::addPrimitiveArg(const void* value, int size) {
    lock_guard<mutex> lock(impl->argumentMutex);
    if (size <= 0 || value == NULL)
        throw OpenMMException("Primitive Metal kernel arguments must contain at least one byte");
    impl->validateConstantSlot(impl->arguments.size(), static_cast<size_t>(size));
    KernelArgument argument;
    argument.type = KernelArgument::Type::Primitive;
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(value);
    argument.primitive.assign(bytes, bytes+size);
    impl->arguments.push_back(argument);
}

void MetalKernel::addEmptyArg() {
    lock_guard<mutex> lock(impl->argumentMutex);
    impl->arguments.push_back(KernelArgument());
}

void MetalKernel::setArrayArg(int index, ArrayInterface& value) {
    lock_guard<mutex> lock(impl->argumentMutex);
    if (index < 0 || static_cast<size_t>(index) >= impl->arguments.size())
        throw OpenMMException("Invalid argument index for Metal kernel "+impl->name);
    MetalArray* array = dynamic_cast<MetalArray*>(&value);
    if (array == NULL)
        throw OpenMMException("Metal kernel arguments must be MetalArray objects");
    if (!array->isInitialized())
        throw OpenMMException("Cannot bind an uninitialized MetalArray to kernel "+impl->name);
    shared_ptr<MetalBufferState> buffer = MetalArrayAccess::getBufferState(*array);
    if (buffer->queue.get() != impl->queue.get())
        throw OpenMMException("Cannot bind a MetalArray from a different command queue to kernel "+impl->name);
    impl->validateBufferSlot(static_cast<size_t>(index));
    KernelArgument& argument = impl->arguments[index];
    argument.type = KernelArgument::Type::Array;
    argument.buffer = buffer;
    argument.primitive.clear();
}

void MetalKernel::setPrimitiveArg(int index, const void* value, int size) {
    lock_guard<mutex> lock(impl->argumentMutex);
    if (index < 0 || static_cast<size_t>(index) >= impl->arguments.size())
        throw OpenMMException("Invalid argument index for Metal kernel "+impl->name);
    if (size <= 0 || value == NULL)
        throw OpenMMException("Primitive Metal kernel arguments must contain at least one byte");
    impl->validateConstantSlot(static_cast<size_t>(index), static_cast<size_t>(size));
    KernelArgument& argument = impl->arguments[index];
    argument.type = KernelArgument::Type::Primitive;
    argument.buffer.reset();
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(value);
    argument.primitive.assign(bytes, bytes+size);
}
