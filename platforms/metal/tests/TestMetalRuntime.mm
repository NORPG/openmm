#include "MetalArray.h"
#include "MetalDeviceCaps.h"
#include "MetalEvent.h"
#include "MetalProgram.h"
#include "MetalTestKernelSources.h"
#include "openmm/OpenMMException.h"

#include <cmath>
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
            if (devices.empty())
                throw OpenMMException("No Metal devices are visible");

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

            // Verify the shared-event synchronization path on two queues.
            shared_ptr<MetalQueue> sibling = queue.createSiblingQueue();
            MetalEvent event(queue);
            event.enqueue();
            event.queueWait(static_pointer_cast<ComputeQueueImpl>(sibling));
            sibling->waitUntilIdle();
            event.wait();

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
