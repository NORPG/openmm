#include "MetalProgram.h"
#include "MetalArray.h"
#include "MetalInternal.h"

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <utility>
#include <vector>

using namespace OpenMM;
using namespace OpenMM::detail;
using namespace std;

namespace {

struct KernelArgument {
    enum class Type {
        Empty,
        Array,
        Primitive
    };

    Type type = Type::Empty;
    MetalArray* array = nullptr;
    vector<unsigned char> primitive;
};

} // namespace

struct MetalProgram::Impl {
    ~Impl() {
        if (program != nullptr)
            OMMMetalProgramRelease(program);
    }

    shared_ptr<MetalQueueState> queue;
    OMMMetalProgramRef program = nullptr;
};

struct MetalKernel::Impl {
    Impl(const shared_ptr<MetalQueueState>& queue, OMMMetalKernelRef kernel,
         const string& name, MetalBindingMode bindingMode, int argumentBufferIndex) :
            queue(queue), kernel(kernel), name(name), bindingMode(bindingMode),
            argumentBufferIndex(argumentBufferIndex) {
    }

    ~Impl() {
        if (kernel != nullptr)
            OMMMetalKernelRelease(kernel);
    }

    shared_ptr<MetalQueueState> queue;
    OMMMetalKernelRef kernel = nullptr;
    string name;
    MetalBindingMode bindingMode;
    int argumentBufferIndex;
    vector<KernelArgument> arguments;
    mutex argumentMutex;
};

MetalProgram::MetalProgram(MetalQueue& queue, const string& source,
                           const MetalProgramOptions& options) : impl(new Impl()) {
    if (source.empty())
        throw OpenMMException("Cannot compile an empty Metal program");
    if (options.languageVersionMajor < 0 || options.languageVersionMajor > 0xffff ||
            options.languageVersionMinor < 0 || options.languageVersionMinor > 0xffff)
        throw OpenMMException("Invalid Metal language version");
    queue.state->checkForErrors();

    OMMMetalProgramRef program = nullptr;
    OMMMetalErrorC error = {};
    int32_t status = OMMMetalProgramCreateSource(
            queue.state->queue, source.data(), source.size(),
            options.fastMathEnabled ? 1 : 0,
            static_cast<uint16_t>(options.languageVersionMajor),
            static_cast<uint16_t>(options.languageVersionMinor), &program, &error);
    checkMetalBridgeResult(status, error, "Error compiling Metal program");
    if (program == nullptr)
        throw OpenMMException("Error compiling Metal program: Swift bridge returned a null program");

    impl->queue = queue.state;
    impl->program = program;
}

MetalProgram::MetalProgram(MetalQueue& queue, const void* libraryData,
                           size_t librarySize) : impl(new Impl()) {
    if (libraryData == nullptr || librarySize == 0)
        throw OpenMMException("Cannot load an empty Metal library");
    queue.state->checkForErrors();

    OMMMetalProgramRef program = nullptr;
    OMMMetalErrorC error = {};
    int32_t status = OMMMetalProgramCreateLibrary(queue.state->queue, libraryData,
                                                   librarySize, &program, &error);
    checkMetalBridgeResult(status, error, "Error loading precompiled Metal library");
    if (program == nullptr)
        throw OpenMMException("Error loading precompiled Metal library: Swift bridge returned a null program");

    impl->queue = queue.state;
    impl->program = program;
}

MetalProgram::~MetalProgram() {
}

ComputeKernel MetalProgram::createKernel(const string& name) {
    return static_pointer_cast<ComputeKernelImpl>(createMetalKernel(name));
}

shared_ptr<MetalKernel> MetalProgram::createMetalKernel(const string& name,
                                                        MetalBindingMode bindingMode,
                                                        int argumentBufferIndex) {
    if (argumentBufferIndex < 0)
        throw OpenMMException("Metal argument buffer index must not be negative");

    int32_t nativeBindingMode = (bindingMode == MetalBindingMode::Direct ?
            OMM_METAL_BINDING_DIRECT : OMM_METAL_BINDING_ARGUMENT_BUFFER);
    OMMMetalKernelRef kernel = nullptr;
    OMMMetalErrorC error = {};
    int32_t status = OMMMetalKernelCreate(impl->program, name.data(), name.size(),
                                           nativeBindingMode, argumentBufferIndex,
                                           &kernel, &error);
    checkMetalBridgeResult(status, error, "Error creating Metal pipeline for kernel '"+name+"'");
    if (kernel == nullptr)
        throw OpenMMException("Error creating Metal pipeline for kernel '"+name+
                              "': Swift bridge returned a null kernel");

    try {
        unique_ptr<MetalKernel::Impl> kernelImpl(new MetalKernel::Impl(
                impl->queue, kernel, name, bindingMode, argumentBufferIndex));
        kernel = nullptr;
        return shared_ptr<MetalKernel>(new MetalKernel(std::move(kernelImpl)));
    }
    catch (...) {
        OMMMetalKernelRelease(kernel);
        throw;
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
    int32_t result = 0;
    OMMMetalErrorC error = {};
    int32_t status = OMMMetalKernelMaxBlockSize(impl->kernel, &result, &error);
    checkMetalBridgeResult(status, error, "Cannot query maximum threadgroup size for Metal kernel "+impl->name);
    return static_cast<int>(result);
}

void MetalKernel::execute(int threads, int blockSize) {
    if (threads < 0)
        throw OpenMMException("Cannot execute Metal kernel "+impl->name+" with a negative thread count");
    if (threads == 0)
        return;
    if (blockSize != -1) {
        int maxBlockSize = getMaxBlockSize();
        if (blockSize <= 0 || blockSize > maxBlockSize) {
            throw OpenMMException("Invalid threadgroup size "+to_string(blockSize)+
                                  " for Metal kernel "+impl->name+" (maximum "+
                                  to_string(maxBlockSize)+")");
        }
    }

    lock_guard<mutex> argumentLock(impl->argumentMutex);
    for (size_t index = 0; index < impl->arguments.size(); index++) {
        if (impl->arguments[index].type == KernelArgument::Type::Empty)
            throw OpenMMException("Argument "+to_string(index)+
                                  " has not been set for Metal kernel "+impl->name);
    }
    impl->queue->checkForErrors();

    OMMMetalErrorC error = {};
    lock_guard<mutex> queueLock(impl->queue->submissionMutex);
    int32_t status = OMMMetalKernelExecute(impl->kernel, threads, blockSize, &error);
    checkMetalBridgeResult(status, error, "Error executing Metal kernel "+impl->name);
}

void MetalKernel::addArrayArg(ArrayInterface& value) {
    lock_guard<mutex> lock(impl->argumentMutex);
    MetalArray* array = dynamic_cast<MetalArray*>(&value);
    if (array == nullptr)
        throw OpenMMException("Metal kernel arguments must be MetalArray objects");
    if (!array->isInitialized())
        throw OpenMMException("Cannot bind an uninitialized MetalArray to kernel "+impl->name);
    if (MetalArrayAccess::getQueueState(*array).get() != impl->queue.get())
        throw OpenMMException("Cannot bind a MetalArray from a different command queue to kernel "+impl->name);

    KernelArgument argument;
    argument.type = KernelArgument::Type::Array;
    argument.array = array;
    impl->arguments.push_back(std::move(argument));
    try {
        OMMMetalErrorC error = {};
        int32_t status = OMMMetalKernelAddBuffer(impl->kernel,
                                                  MetalArrayAccess::getBuffer(*array), &error);
        checkMetalBridgeResult(status, error, "Cannot add a buffer argument to Metal kernel "+impl->name);
    }
    catch (...) {
        impl->arguments.pop_back();
        throw;
    }
}

void MetalKernel::addPrimitiveArg(const void* value, int size) {
    lock_guard<mutex> lock(impl->argumentMutex);
    if (size <= 0 || value == nullptr)
        throw OpenMMException("Primitive Metal kernel arguments must contain at least one byte");

    KernelArgument argument;
    argument.type = KernelArgument::Type::Primitive;
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(value);
    argument.primitive.assign(bytes, bytes+size);
    impl->arguments.push_back(std::move(argument));
    try {
        OMMMetalErrorC error = {};
        int32_t status = OMMMetalKernelAddBytes(impl->kernel, value,
                                                 static_cast<size_t>(size), &error);
        checkMetalBridgeResult(status, error, "Cannot add an inline argument to Metal kernel "+impl->name);
    }
    catch (...) {
        impl->arguments.pop_back();
        throw;
    }
}

void MetalKernel::addEmptyArg() {
    lock_guard<mutex> lock(impl->argumentMutex);
    impl->arguments.push_back(KernelArgument());
    try {
        OMMMetalErrorC error = {};
        int32_t status = OMMMetalKernelAddEmpty(impl->kernel, &error);
        checkMetalBridgeResult(status, error, "Cannot add an empty argument to Metal kernel "+impl->name);
    }
    catch (...) {
        impl->arguments.pop_back();
        throw;
    }
}

void MetalKernel::setArrayArg(int index, ArrayInterface& value) {
    lock_guard<mutex> lock(impl->argumentMutex);
    if (index < 0 || static_cast<size_t>(index) >= impl->arguments.size())
        throw OpenMMException("Invalid argument index for Metal kernel "+impl->name);
    MetalArray* array = dynamic_cast<MetalArray*>(&value);
    if (array == nullptr)
        throw OpenMMException("Metal kernel arguments must be MetalArray objects");
    if (!array->isInitialized())
        throw OpenMMException("Cannot bind an uninitialized MetalArray to kernel "+impl->name);
    if (MetalArrayAccess::getQueueState(*array).get() != impl->queue.get())
        throw OpenMMException("Cannot bind a MetalArray from a different command queue to kernel "+impl->name);

    OMMMetalErrorC error = {};
    int32_t status = OMMMetalKernelSetBuffer(impl->kernel, index,
                                              MetalArrayAccess::getBuffer(*array), &error);
    checkMetalBridgeResult(status, error, "Cannot set a buffer argument on Metal kernel "+impl->name);
    KernelArgument& argument = impl->arguments[index];
    argument.type = KernelArgument::Type::Array;
    argument.array = array;
    argument.primitive.clear();
}

void MetalKernel::setPrimitiveArg(int index, const void* value, int size) {
    lock_guard<mutex> lock(impl->argumentMutex);
    if (index < 0 || static_cast<size_t>(index) >= impl->arguments.size())
        throw OpenMMException("Invalid argument index for Metal kernel "+impl->name);
    if (size <= 0 || value == nullptr)
        throw OpenMMException("Primitive Metal kernel arguments must contain at least one byte");

    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(value);
    vector<unsigned char> primitive(bytes, bytes+size);
    OMMMetalErrorC error = {};
    int32_t status = OMMMetalKernelSetBytes(impl->kernel, index, value,
                                             static_cast<size_t>(size), &error);
    checkMetalBridgeResult(status, error, "Cannot set an inline argument on Metal kernel "+impl->name);
    KernelArgument& argument = impl->arguments[index];
    argument.type = KernelArgument::Type::Primitive;
    argument.array = nullptr;
    argument.primitive = std::move(primitive);
}
