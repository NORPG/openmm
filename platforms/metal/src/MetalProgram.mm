#include "MetalProgram.h"
#include "MetalArray.h"
#include "MetalInternal.h"

#include <algorithm>
#include <cstring>
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
    MetalArray* array = NULL;
    vector<unsigned char> primitive;
};

} // namespace

struct MetalProgram::Impl {
    shared_ptr<MetalQueueState> queue;
    id<MTLLibrary> library = nil;
};

struct MetalKernel::Impl {
    Impl(const shared_ptr<MetalQueueState>& queue, id<MTLFunction> function,
         id<MTLComputePipelineState> pipeline, MetalBindingMode bindingMode,
         int argumentBufferIndex) : queue(queue), function(function), pipeline(pipeline),
         name(fromNSString(function.name)), bindingMode(bindingMode),
         argumentBufferIndex(argumentBufferIndex) {
        if (bindingMode == MetalBindingMode::ArgumentBuffer) {
            argumentEncoder = [function newArgumentEncoderWithBufferIndex:argumentBufferIndex];
            if (argumentEncoder == nil)
                throw OpenMMException("Kernel "+name+" has no argument buffer at buffer index "+to_string(argumentBufferIndex));
        }
    }

    shared_ptr<MetalQueueState> queue;
    id<MTLFunction> function = nil;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLArgumentEncoder> argumentEncoder = nil;
    string name;
    MetalBindingMode bindingMode;
    int argumentBufferIndex;
    vector<KernelArgument> arguments;
    mutex argumentMutex;
};

MetalProgram::MetalProgram(MetalQueue& queue, const string& source, const MetalProgramOptions& options) : impl(new Impl()) {
    @autoreleasepool {
        if (source.empty())
            throw OpenMMException("Cannot compile an empty Metal program");
        if (options.languageVersionMajor < 0 || options.languageVersionMinor < 0 ||
                options.languageVersionMinor > 0xffff)
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
        NSString* nativeName = makeNSString(name, "Metal kernel name");
        id<MTLFunction> function = [impl->library newFunctionWithName:nativeName];
        if (function == nil)
            throw OpenMMException("Metal program does not contain a kernel named '"+name+"'");

        NSError* error = nil;
        id<MTLComputePipelineState> pipeline = [impl->queue->device newComputePipelineStateWithFunction:function error:&error];
        if (pipeline == nil)
            throw OpenMMException("Error creating Metal pipeline for kernel '"+name+"': "+describeError(error));
        unique_ptr<MetalKernel::Impl> kernelImpl(new MetalKernel::Impl(impl->queue, function, pipeline,
                                                                       bindingMode, argumentBufferIndex));
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
                    id<MTLBuffer> buffer = MetalArrayAccess::getBuffer(*argument.array);
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
                    id<MTLBuffer> buffer = MetalArrayAccess::getBuffer(*argument.array);
                    [impl->argumentEncoder setBuffer:buffer offset:0 atIndex:i];
                    [encoder useResource:buffer usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
                }
                else {
                    void* destination = [impl->argumentEncoder constantDataAtIndex:i];
                    if (destination == NULL)
                        throw OpenMMException("Argument "+to_string(i)+" of Metal kernel "+impl->name+
                                              " is not inline data in its argument buffer");
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
    if (MetalArrayAccess::getQueueState(*array).get() != impl->queue.get())
        throw OpenMMException("Cannot bind a MetalArray from a different command queue to kernel "+impl->name);
    KernelArgument argument;
    argument.type = KernelArgument::Type::Array;
    argument.array = array;
    impl->arguments.push_back(argument);
}

void MetalKernel::addPrimitiveArg(const void* value, int size) {
    lock_guard<mutex> lock(impl->argumentMutex);
    if (size <= 0 || value == NULL)
        throw OpenMMException("Primitive Metal kernel arguments must contain at least one byte");
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
    if (MetalArrayAccess::getQueueState(*array).get() != impl->queue.get())
        throw OpenMMException("Cannot bind a MetalArray from a different command queue to kernel "+impl->name);
    KernelArgument& argument = impl->arguments[index];
    argument.type = KernelArgument::Type::Array;
    argument.array = array;
    argument.primitive.clear();
}

void MetalKernel::setPrimitiveArg(int index, const void* value, int size) {
    lock_guard<mutex> lock(impl->argumentMutex);
    if (index < 0 || static_cast<size_t>(index) >= impl->arguments.size())
        throw OpenMMException("Invalid argument index for Metal kernel "+impl->name);
    if (size <= 0 || value == NULL)
        throw OpenMMException("Primitive Metal kernel arguments must contain at least one byte");
    KernelArgument& argument = impl->arguments[index];
    argument.type = KernelArgument::Type::Primitive;
    argument.array = NULL;
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(value);
    argument.primitive.assign(bytes, bytes+size);
}
