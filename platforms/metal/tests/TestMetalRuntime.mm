#include "MetalArray.h"
#include "MetalDeviceCaps.h"
#include "MetalEvent.h"
#include "MetalProgram.h"
#include "MetalTestKernelSources.h"
#include "openmm/OpenMMException.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace OpenMM;
using namespace std;

namespace {

void assertVector(const vector<float>& actual, const vector<float>& expected, const string& description) {
    if (actual.size() != expected.size())
        throw OpenMMException(description+": vector length mismatch");
    for (size_t i = 0; i < actual.size(); i++) {
        if (fabs(actual[i]-expected[i]) > 1.0e-6f)
            throw OpenMMException(description+": mismatch at index "+to_string(i));
    }
}

void bindVectorKernel(const shared_ptr<MetalKernel>& kernel, MetalArray& a, MetalArray& b,
                      MetalArray& result, unsigned int count) {
    kernel->addArg(a);
    kernel->addArg(b);
    kernel->addArg(result);
    kernel->addArg(count);
}

} // namespace

int main() {
    try {
        @autoreleasepool {
            vector<MetalDeviceCaps> devices = MetalDeviceCaps::enumerate();
            if (devices.empty()) {
                cout << "Test skipped: no Metal devices are visible" << endl;
                return 0;
            }

            MetalQueue queue;
            const MetalDeviceCaps& caps = queue.getDeviceCaps();
            if (caps.getName().empty() || caps.getMaxBufferLength() == 0 || caps.getMaxThreadsPerThreadgroup() == 0)
                throw OpenMMException("Metal device capability query returned incomplete data");

            const unsigned int count = 257;
            vector<float> aData(count), bData(count), expected(count);
            for (unsigned int i = 0; i < count; i++) {
                aData[i] = 0.25f*i;
                bData[i] = 3.0f-0.5f*i;
            }
            const float patch[] = {17.0f, 19.0f, 23.0f};
            aData[31] = patch[0];
            aData[32] = patch[1];
            aData[33] = patch[2];
            for (unsigned int i = 0; i < count; i++)
                expected[i] = aData[i]+bData[i];

            MetalArray a(queue, count, sizeof(float), "a");
            MetalArray b(queue, count, sizeof(float), "b");
            MetalArray directResult(queue, count, sizeof(float), "direct result");
            MetalArray argumentResult(queue, count, sizeof(float), "argument result");
            MetalArray copiedResult(queue, count, sizeof(float), "copied result");

            // Nonblocking uploads followed by a subrange upload exercise blit
            // ordering before compute dispatch.
            a.upload(aData.data(), false);
            b.upload(bData.data(), false);
            a.uploadSubArray(patch, 31, 3, false);

            MetalProgram program(queue, MetalTestKernelSources::vectorAdd);
            bool caughtInvalidLanguageVersion = false;
            try {
                MetalProgramOptions invalidVersion;
                invalidVersion.languageVersionMajor = 0x10000;
                MetalProgram invalid(queue, MetalTestKernelSources::vectorAdd, invalidVersion);
            }
            catch (const OpenMMException&) {
                caughtInvalidLanguageVersion = true;
            }
            if (!caughtInvalidLanguageVersion)
                throw OpenMMException("Out-of-range Metal language version was not rejected");

            bool caughtInvalidBindingMode = false;
            try {
                program.createMetalKernel("vectorAdd", static_cast<MetalBindingMode>(99));
            }
            catch (const OpenMMException&) {
                caughtInvalidBindingMode = true;
            }
            if (!caughtInvalidBindingMode)
                throw OpenMMException("Unknown Metal binding mode was not rejected");

            shared_ptr<MetalKernel> direct = program.createMetalKernel("vectorAdd");
            bindVectorKernel(direct, a, b, directResult, count);
            direct->execute(count);

            vector<float> downloaded(count);
            directResult.download(downloaded.data(), true);
            assertVector(downloaded, expected, "direct binding");

            shared_ptr<MetalKernel> argumentBuffer = program.createMetalKernel(
                    "vectorAddArgumentBuffer", MetalBindingMode::ArgumentBuffer);
            bindVectorKernel(argumentBuffer, a, b, argumentResult, count);
            argumentBuffer->execute(count, 64);
            argumentResult.download(downloaded.data(), true);
            assertVector(downloaded, expected, "argument-buffer binding");

            bool caughtArgumentBufferTypeError = false;
            try {
                shared_ptr<MetalKernel> wrongType = program.createMetalKernel(
                        "vectorAddArgumentBuffer", MetalBindingMode::ArgumentBuffer);
                wrongType->addArg(count); // Slot 0 is a device pointer.
            }
            catch (const OpenMMException&) {
                caughtArgumentBufferTypeError = true;
            }
            if (!caughtArgumentBufferTypeError)
                throw OpenMMException("Argument-buffer slot type mismatch was not rejected");

            bool caughtArgumentBufferSizeError = false;
            try {
                shared_ptr<MetalKernel> wrongSize = program.createMetalKernel(
                        "vectorAddArgumentBuffer", MetalBindingMode::ArgumentBuffer);
                wrongSize->addArg(a);
                wrongSize->addArg(b);
                wrongSize->addArg(argumentResult);
                wrongSize->addArg(uint64_t(count)); // Slot 3 is a 32-bit uint.
            }
            catch (const OpenMMException&) {
                caughtArgumentBufferSizeError = true;
            }
            if (!caughtArgumentBufferSizeError)
                throw OpenMMException("Argument-buffer constant size mismatch was not rejected");

            argumentResult.copyTo(copiedResult);
            vector<float> subrange(7);
            copiedResult.downloadSubArray(subrange.data(), 23, static_cast<int>(subrange.size()), true);
            vector<float> expectedSubrange(expected.begin()+23, expected.begin()+30);
            assertVector(subrange, expectedSubrange, "buffer copy and subrange download");

            vector<float> asynchronous(count, 0.0f);
            copiedResult.download(asynchronous.data(), false);
            queue.waitUntilIdle();
            assertVector(asynchronous, expected, "nonblocking download");

            copiedResult.resize(11);
            vector<float> resized(11);
            for (size_t i = 0; i < resized.size(); i++)
                resized[i] = static_cast<float>(i*i);
            copiedResult.upload(resized);
            vector<float> resizedDownload;
            copiedResult.download(resizedDownload);
            assertVector(resizedDownload, resized, "resized buffer");

            // A bound argument refers to stable buffer state, not to the
            // MTLBuffer that happened to exist when it was bound.
            MetalArray reboundResult(queue, 1, sizeof(float), "resize-after-bind result");
            shared_ptr<MetalKernel> resizeAfterBind = program.createMetalKernel("vectorAdd");
            bindVectorKernel(resizeAfterBind, a, b, reboundResult, count);
            reboundResult.resize(count);
            resizeAfterBind->execute(count);
            reboundResult.download(downloaded);
            assertVector(downloaded, expected, "resize after kernel binding");

            // Verify the shared-event synchronization path on two queues.
            shared_ptr<MetalQueue> sibling = queue.createSiblingQueue();
            MetalEvent event(queue);
            event.enqueue();
            event.queueWait(static_pointer_cast<ComputeQueueImpl>(sibling));
            sibling->waitUntilIdle();
            event.wait();

            // Native queue and pipeline state must outlive their public C++
            // wrappers while arrays and a kernel still refer to them.
            unique_ptr<MetalArray> lifetimeA;
            unique_ptr<MetalArray> lifetimeB;
            unique_ptr<MetalArray> lifetimeResult;
            shared_ptr<MetalKernel> lifetimeKernel;
            {
                unique_ptr<MetalQueue> temporaryQueue(new MetalQueue());
                lifetimeA.reset(new MetalArray(*temporaryQueue, count, sizeof(float), "lifetime a"));
                lifetimeB.reset(new MetalArray(*temporaryQueue, count, sizeof(float), "lifetime b"));
                lifetimeResult.reset(new MetalArray(*temporaryQueue, count, sizeof(float), "lifetime result"));
                lifetimeA->upload(aData);
                lifetimeB->upload(bData);
                {
                    MetalProgram temporaryProgram(*temporaryQueue, MetalTestKernelSources::vectorAdd);
                    lifetimeKernel = temporaryProgram.createMetalKernel("vectorAdd");
                    bindVectorKernel(lifetimeKernel, *lifetimeA, *lifetimeB, *lifetimeResult, count);
                }
            }
            lifetimeKernel->execute(count);
            lifetimeResult->download(downloaded);
            assertVector(downloaded, expected, "queue and program wrapper lifetime");

            // A kernel retains bound buffer state even if input MetalArray
            // wrappers are destroyed before the kernel is dispatched.
            MetalArray retainedResult(queue, count, sizeof(float), "retained argument result");
            shared_ptr<MetalKernel> retainedArguments = program.createMetalKernel("vectorAdd");
            {
                unique_ptr<MetalArray> temporaryA(new MetalArray(queue, count, sizeof(float), "temporary a"));
                unique_ptr<MetalArray> temporaryB(new MetalArray(queue, count, sizeof(float), "temporary b"));
                temporaryA->upload(aData);
                temporaryB->upload(bData);
                bindVectorKernel(retainedArguments, *temporaryA, *temporaryB, retainedResult, count);
            }
            retainedArguments->execute(count);
            retainedResult.download(downloaded);
            assertVector(downloaded, expected, "bound array wrapper lifetime");

            bool caughtCompileError = false;
            try {
                MetalProgram invalid(queue, "this is not MSL");
            }
            catch (const OpenMMException&) {
                caughtCompileError = true;
            }
            if (!caughtCompileError)
                throw OpenMMException("Invalid MSL did not report a compilation error");

            cout << "Metal runtime tests passed on " << caps.getName()
                 << " (Apple GPU family " << caps.getHighestAppleGpuFamily()
                 << ", argument buffer tier " << caps.getArgumentBufferTier() << ")" << endl;
        }
        return 0;
    }
    catch (const exception& e) {
        cerr << "TestMetalRuntime failed: " << e.what() << endl;
        return 1;
    }
}
