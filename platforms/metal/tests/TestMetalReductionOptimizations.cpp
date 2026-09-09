/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Metal Platform code: Portions copyright (c) 2026 Chun-Chi Hung.             *
 * Authors: Chun-Chi Hung                                                     *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    *
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OF THE SOFTWARE.*
 * -------------------------------------------------------------------------- */

#include "MetalContext.h"
#include "MetalKernel.h"
#include "MetalReductionOptimizations.h"
#include "CommonKernelSources.h"
#include "MetalKernelSources.h"
#include "openmm/common/ComputeArray.h"
#include "openmm/internal/AssertionUtilities.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>

using namespace OpenMM;
using namespace std;

/** @brief Extract a test kernel without compiling unrelated force templates. */
string extractFunction(const string& source, const string& signature) {
    size_t begin = source.find(signature);
    ASSERT(begin != string::npos);
    size_t body = source.find('{', begin);
    int depth = 1;
    size_t end = body+1;
    for (; end < source.size() && depth != 0; end++) {
        if (source[end] == '{') depth++;
        if (source[end] == '}') depth--;
    }
    ASSERT_EQUAL(0, depth);
    return source.substr(begin, end-begin)+"\n";
}

/** Native sorting helpers are source-owned MSL and keep their explicit context. */
string nativeSortEntry(const string& signature) {
    const string& source = MetalKernelSources::sort;
    const size_t first = source.find("KERNEL void sortShortList(");
    ASSERT(first != string::npos);
    return source.substr(0, first)+extractFunction(source, signature);
}

/** @brief Verify independent defines and preserve each shared source during composition. */
void testSelection() {
    using Settings = MetalReductionOptimizations::Settings;
    const char* options[] = {
        "OPENMM_METAL_FAST_CENTROID_REDUCTION", "OPENMM_METAL_FAST_RG_REDUCTION",
        "OPENMM_METAL_FAST_RMSD_REDUCTION", "OPENMM_METAL_FAST_ORIENTATION_REDUCTION",
        "OPENMM_METAL_FAST_LCPO_NEIGHBOR_SCAN", "OPENMM_METAL_FAST_CUSTOM_MANY_PARTICLE_NEIGHBOR_SCAN",
        "OPENMM_METAL_FAST_SORT_BUCKET_SCAN", "OPENMM_METAL_FAST_SORT_REGISTER_BITONIC"};
    const string unrelated = "DEVICE real reduceValue(real value, LOCAL_ARG volatile real* temp) { return value; }";
    // Shared Common sources and native Metal sorting own branches gated by independent
    // switches and the selected MSL version; their kernel signatures stay fixed.
    ASSERT(CommonKernelSources::customCentroidBond.find("#if defined(__METAL__) && OPENMM_METAL_FAST_CENTROID_REDUCTION && __METAL_VERSION__ >= 300") != string::npos);
    ASSERT(CommonKernelSources::rg.find("#if defined(__METAL__) && OPENMM_METAL_FAST_RG_REDUCTION && __METAL_VERSION__ >= 300") != string::npos);
    ASSERT(CommonKernelSources::rmsd.find("#if defined(__METAL__) && OPENMM_METAL_FAST_RMSD_REDUCTION && __METAL_VERSION__ >= 300") != string::npos);
    ASSERT(CommonKernelSources::orientationRestraintForce.find("#if defined(__METAL__) && OPENMM_METAL_FAST_ORIENTATION_REDUCTION && __METAL_VERSION__ >= 300") != string::npos);
    ASSERT(CommonKernelSources::lcpo.find("#if defined(__METAL__) && OPENMM_METAL_FAST_LCPO_NEIGHBOR_SCAN && __METAL_VERSION__ >= 300") != string::npos);
    ASSERT(CommonKernelSources::customManyParticle.find("#if defined(__METAL__) && OPENMM_METAL_FAST_CUSTOM_MANY_PARTICLE_NEIGHBOR_SCAN && __METAL_VERSION__ >= 300") != string::npos);
    ASSERT(MetalKernelSources::sort.find("#if OPENMM_METAL_FAST_SORT_BUCKET_SCAN && __METAL_VERSION__ >= 300") != string::npos);
    ASSERT(MetalKernelSources::sort.find("#if OPENMM_METAL_FAST_SORT_REGISTER_BITONIC && __METAL_VERSION__ >= 300") != string::npos);
    for (int option = -1; option < 8; option++) {
        Settings settings;
        settings.centroid = option == 0;
        settings.rg = option == 1;
        settings.rmsd = option == 2;
        settings.orientation = option == 3;
        settings.lcpoScan = option == 4;
        settings.manyParticleScan = option == 5;
        settings.sortBucketScan = option == 6;
        settings.sortRegisterBitonic = option == 7;
        for (const string* source : {&CommonKernelSources::customCentroidBond, &CommonKernelSources::rg,
                &CommonKernelSources::rmsd, &CommonKernelSources::orientationRestraintForce,
                &CommonKernelSources::lcpo, &CommonKernelSources::customManyParticle,
                &MetalKernelSources::sort, &unrelated}) {
            const string configured = MetalReductionOptimizations::apply(*source, settings);
            ASSERT_EQUAL(*source, configured.substr(configured.size()-source->size()));
            ASSERT(configured.find("KERNEL_FUNCTION(") == string::npos);
            ASSERT(configured.find("CONTEXT_PARAM") == string::npos);
            ASSERT(configured.find("CONTEXT_ARG") == string::npos);
            for (int i = 0; i < 8; i++) {
                const string define = string("#define ")+options[i]+(option == i ? " 1\n" : " 0\n");
                ASSERT(configured.find(define) != string::npos);
            }
            // Explicit OFF and sort eligibility must survive compilation's
            // second application of build defaults, including all-ON builds.
            ASSERT_EQUAL(configured, MetalReductionOptimizations::apply(configured));
            const string prefixed = "#define UNRELATED_OPTION 1\n"+configured;
            ASSERT_EQUAL(prefixed, MetalReductionOptimizations::apply(prefixed));
        }
    }
}

/** @brief Compare every lane's reduction result, including partially occupied SIMD groups and scratch reuse. */
void testReductions(MetalContext& context) {
    const string probe = R"(
KERNEL void reduceProbe(GLOBAL const real* input, GLOBAL real* output, int count) {
    LOCAL volatile real temp[1024];
    real first = 0, second = 0;
    for (int i = LOCAL_ID; i < count; i += LOCAL_SIZE) {
        first += input[i];
        second += input[i]*input[i];
    }
    first = reduceValue(first, temp);
    second = reduceValue(second, temp);
    output[LOCAL_ID] = first;
    output[LOCAL_ID+LOCAL_SIZE] = second;
}
)";
    const int count = 4001;
    vector<float> values(count);
    double first = 0, second = 0;
    for (int i = 0; i < count; i++) {
        values[i] = i%17-8;
        first += values[i];
        second += values[i]*values[i];
    }
    ComputeArray input, output;
    input.initialize<float>(context, count, "reductionInput");
    output.initialize<float>(context, 2048, "reductionOutput");
    input.upload(values);
    for (bool enabled : {false, true}) {
        MetalReductionOptimizations::Settings settings;
        settings.rg = enabled;
        const string source = extractFunction(CommonKernelSources::rg, "DEVICE real reduceValue(")+probe;
        ComputeKernel kernel = context.compileProgram(MetalReductionOptimizations::apply(source, settings))->createKernel("reduceProbe");
        kernel->addArg(input);
        kernel->addArg(output);
        kernel->addArg(count);
        for (int width : {1, 7, 31, 32, 33, 63, 64, 65, 127, 128, 255, 256, 1024}) {
            if (width > kernel->getMaxBlockSize())
                continue;
            kernel->execute(width, width);
            vector<float> result;
            output.download(result);
            for (int lane = 0; lane < width; lane++) {
                ASSERT_EQUAL(first, result[lane]);
                ASSERT_EQUAL(second, result[lane+width]);
            }
        }
    }
}

/** @brief Check weighted centroid reductions across short groups and repeated group iterations. */
void testCentroids(MetalContext& context) {
    vector<int> offsets{0};
    vector<int> particles;
    vector<float> weights;
    vector<mm_float4> positions;
    vector<Vec3> expected;
    for (int count : {1, 7, 31, 32, 33, 64, 257}) {
        Vec3 sum;
        for (int i = 0; i < count; i++) {
            mm_float4 value(i%17-8, i%9-4, i%13-6, 0);
            float weight = 1.0f/count;
            particles.push_back(positions.size());
            positions.push_back(value);
            weights.push_back(weight);
            sum += weight*Vec3(value.x, value.y, value.z);
        }
        expected.push_back(sum);
        offsets.push_back(particles.size());
    }
    ComputeArray posq, atomIndices, groupWeights, groupOffsets, centers;
    posq.initialize<mm_float4>(context, positions.size(), "centroidPositions");
    atomIndices.initialize<int>(context, particles.size(), "centroidParticles");
    groupWeights.initialize<float>(context, weights.size(), "centroidWeights");
    groupOffsets.initialize<int>(context, offsets.size(), "centroidOffsets");
    centers.initialize<mm_float4>(context, expected.size(), "centroidOutput");
    posq.upload(positions);
    atomIndices.upload(particles);
    groupWeights.upload(weights);
    groupOffsets.upload(offsets);
    for (bool enabled : {false, true}) {
        MetalReductionOptimizations::Settings settings;
        settings.centroid = enabled;
        const string source = extractFunction(CommonKernelSources::customCentroidBond, "KERNEL void computeGroupCenters(");
        ComputeKernel kernel = context.compileProgram(MetalReductionOptimizations::apply(source, settings))->createKernel("computeGroupCenters");
        kernel->addArg((int) expected.size());
        kernel->addArg(posq);
        kernel->addArg(atomIndices);
        kernel->addArg(groupWeights);
        kernel->addArg(groupOffsets);
        kernel->addArg(centers);
        for (int groups : {1, 2, 7}) {
            kernel->execute(64*groups, 64);
            vector<mm_float4> result;
            centers.download(result);
            for (int i = 0; i < (int) expected.size(); i++) {
                ASSERT_EQUAL_VEC(expected[i], Vec3(result[i].x, result[i].y, result[i].z), 1e-5);
                ASSERT_EQUAL(0, result[i].w);
            }
        }
    }
}

/** @brief Exercise inclusive scans, chunk carries, overflow exits, and neighbor-counter clearing. */
void testNeighborScans(MetalContext& context) {
    for (bool lcpo : {false, true}) {
        const string& original = lcpo ? CommonKernelSources::lcpo : CommonKernelSources::customManyParticle;
        for (bool enabled : {false, true}) {
            MetalReductionOptimizations::Settings settings;
            settings.lcpoScan = lcpo && enabled;
            settings.manyParticleScan = !lcpo && enabled;
            const string kernelSource = MetalReductionOptimizations::apply(
                    extractFunction(original, "KERNEL void computeNeighborStartIndices("), settings);
            for (int count : {0, 1, 31, 33, 257, 4099}) {
                vector<int> counts(max(1, count));
                for (int i = 0; i < count; i++)
                    counts[i] = i%7;
                int total = accumulate(counts.begin(), counts.end(), 0);
                ComputeArray countsBuffer, startsBuffer, pairCount;
                countsBuffer.initialize<int>(context, counts.size(), "scanCounts");
                startsBuffer.initialize<int>(context, count+1, "scanStarts");
                pairCount.initialize<int>(context, 1, "scanPairCount");
                map<string, string> defines{{"NUM_ACTIVE", to_string(count)}, {"NUM_ATOMS", to_string(count)}, {"THREAD_BLOCK_SIZE", "256"}};
                ComputeKernel kernel = context.compileProgram(kernelSource, defines)->createKernel("computeNeighborStartIndices");
                if (lcpo) {
                    kernel->addArg(pairCount);
                    kernel->addArg(countsBuffer);
                    kernel->addArg(startsBuffer);
                }
                else {
                    kernel->addArg(countsBuffer);
                    kernel->addArg(startsBuffer);
                    kernel->addArg(pairCount);
                }
                kernel->addArg(total);
                for (int width : {7, 32, 33, 64, 256}) {
                    for (bool overflow : {false, true}) {
                        countsBuffer.upload(counts);
                        startsBuffer.upload(vector<int>(count+1, -7));
                        pairCount.upload(vector<int>{overflow ? total+1 : total});
                        kernel->execute(width, width);
                        vector<int> starts, remaining;
                        startsBuffer.download(starts);
                        countsBuffer.download(remaining);
                        int expected = 0;
                        for (int i = 0; i <= count; i++) {
                            ASSERT_EQUAL(overflow ? (lcpo ? -7 : 0) : expected, starts[i]);
                            if (i < count) {
                                expected += counts[i];
                                ASSERT_EQUAL(overflow ? counts[i] : 0, remaining[i]);
                            }
                        }
                    }
                }
            }
        }
    }
}

/** @brief Verify bucket scans for empty/partial SIMD groups and multiple chunks. */
void testBucketScans(MetalContext& context) {
    for (bool enabled : {false, true}) {
        MetalReductionOptimizations::Settings settings;
        settings.sortBucketScan = enabled;
        const string source = MetalReductionOptimizations::apply(
                nativeSortEntry("KERNEL void computeBucketPositions("), settings);
        ComputeProgram program = context.compileProgram(source, {{"DATA_TYPE", "uint"}, {"KEY_TYPE", "uint"}, {"SORT_KEY", "value"}});
        for (int count : {0, 1, 31, 33, 257, 4099}) {
            vector<unsigned int> values(max(1, count));
            for (int i = 0; i < count; i++)
                values[i] = i%13;
            ComputeArray buffer;
            buffer.initialize<unsigned int>(context, values.size(), "bucketScanValues");
            ComputeKernel kernel = program->createKernel("computeBucketPositions");
            kernel->addArg((unsigned int) count);
            kernel->addArg(buffer);
            static_cast<MetalKernel&>(*kernel).addLocalArg(256*sizeof(unsigned int));
            for (int width : {1, 7, 32, 33, 64, 256}) {
                buffer.upload(values);
                static_cast<MetalKernel&>(*kernel).setLocalArg(2, width*sizeof(unsigned int));
                kernel->execute(width, width);
                vector<unsigned int> result;
                buffer.download(result);
                unsigned int expected = 0;
                for (int i = 0; i < count; i++) {
                    expected += values[i];
                    ASSERT_EQUAL(expected, result[i]);
                }
            }
        }
    }
}

/** @brief Test register sorting with duplicate keys, padding lanes, maximum unsigned keys, and payloads. */
void testRegisterSort(MetalContext& context) {
    MetalReductionOptimizations::Settings settings;
    settings.sortRegisterBitonic = true;
    const string source = MetalReductionOptimizations::apply(
            nativeSortEntry("KERNEL void sortShortList(")+R"(
#if !defined(__METAL__)
#error Metal language marker is missing
#endif
#if !OPENMM_METAL_FAST_SORT_REGISTER_BITONIC
#error Explicit register sorting was not enabled
#endif
)", settings);
    for (int kind = 0; kind < 3; kind++) {
        bool records = kind != 0;
        bool floatRecords = kind == 2;
        map<string, string> defines;
        defines["DATA_TYPE"] = floatRecords ? "float4" : (records ? "int2" : "unsigned int");
        defines["KEY_TYPE"] = floatRecords ? "float" : (records ? "int" : "unsigned int");
        defines["SORT_KEY"] = records ? "value.x" : "value";
        defines["MAX_VALUE"] = floatRecords ? "make_float4(MAXFLOAT)" : (records ? "make_int2(0x7FFFFFFF, 0)" : "0xFFFFFFFFu");
        defines["OPENMM_METAL_REQUIRE_SAFE_MATH"] = "1";
        ComputeProgram program = context.compileProgram(source, defines);
        for (int count : {0, 1, 2, 3, 7, 15, 16, 17, 31, 32}) {
            ComputeArray data;
            int elementSize = floatRecords ? sizeof(mm_float4) : (records ? sizeof(mm_int2) : sizeof(unsigned int));
            data.initialize(context, max(count, 1), elementSize, "registerSortData");
            vector<mm_int2> pairs(max(1, count));
            vector<mm_float4> vectors(max(1, count));
            vector<unsigned int> values(max(1, count));
            for (int i = 0; i < count; i++) {
                pairs[i] = mm_int2(i%7-3, i);
                vectors[i] = mm_float4(i%7-3, i, 2*i, -i);
                values[i] = (i%3 == 0 ? 0xFFFFFFFFu : (i%3 == 1 ? 0x80000000u : 0u));
            }
            if (floatRecords)
                data.upload(vectors);
            else if (records)
                data.upload(pairs);
            else
                data.upload(values);
            ComputeKernel kernel = program->createKernel("sortShortList");
            kernel->addArg(data);
            kernel->addArg((unsigned int) count);
            static_cast<MetalKernel&>(*kernel).addLocalArg(max(1, count)*elementSize);
            kernel->execute(32, 32);
            if (floatRecords) {
                vector<mm_float4> result;
                data.download(result);
                stable_sort(vectors.begin(), vectors.begin()+count, [](const mm_float4& first, const mm_float4& second) { return first.x < second.x; });
                for (int i = 0; i < count; i++) {
                    ASSERT_EQUAL(vectors[i].x, result[i].x);
                    ASSERT_EQUAL(vectors[i].y, result[i].y);
                    ASSERT_EQUAL(vectors[i].z, result[i].z);
                    ASSERT_EQUAL(vectors[i].w, result[i].w);
                }
            }
            else if (records) {
                vector<mm_int2> result;
                data.download(result);
                stable_sort(pairs.begin(), pairs.begin()+count, [](const mm_int2& first, const mm_int2& second) { return first.x < second.x; });
                for (int i = 0; i < count; i++) {
                    ASSERT_EQUAL(pairs[i].x, result[i].x);
                    ASSERT_EQUAL(pairs[i].y, result[i].y);
                }
            }
            else {
                vector<unsigned int> result;
                data.download(result);
                sort(values.begin(), values.begin()+count);
                for (int i = 0; i < count; i++)
                    ASSERT_EQUAL(values[i], result[i]);
            }
        }
    }
}

/** @brief Safe sorting must preserve nonfinite values across key-helper calls. */
void testSafeMathHelperCalls(MetalContext& context) {
    ComputeArray input, output;
    input.initialize<float>(context, 2, "sortClassificationInput");
    output.initialize<int>(context, 4, "sortClassificationOutput");
    input.upload(vector<float>{nanf(""), 1.0f});
    ComputeKernel kernel = context.compileProgram(R"(
DEVICE float classificationKey(float value) { return value; }
KERNEL void classifySortKey(GLOBAL const float* input, GLOBAL int* output) {
    if (GLOBAL_ID < 2) {
        float direct = input[GLOBAL_ID];
        float key = classificationKey(direct);
        output[2*GLOBAL_ID] = direct != direct;
        output[2*GLOBAL_ID+1] = key != key;
    }
}
)", {{"OPENMM_METAL_REQUIRE_SAFE_MATH", "1"}})->createKernel("classifySortKey");
    kernel->addArg(input);
    kernel->addArg(output);
    kernel->execute(2, 32);
    vector<int> result;
    output.download(result);
    ASSERT_EQUAL(1, result[0]);
    ASSERT_EQUAL(1, result[1]);
    ASSERT_EQUAL(0, result[2]);
    ASSERT_EQUAL(0, result[3]);
}

/** @brief Preserve every key/payload bit for NaNs, infinities, signed zeros, and padded lanes. */
void testRegisterSortSpecialValues(MetalContext& context) {
    MetalReductionOptimizations::Settings settings;
    settings.sortRegisterBitonic = true; // Exercise this even in a default-OFF build.
    const string source = MetalReductionOptimizations::apply(
            nativeSortEntry("KERNEL void sortShortList("), settings);
    const map<string, string> defines = {{"DATA_TYPE", "float4"}, {"KEY_TYPE", "float"},
        {"SORT_KEY", "value.x"}, {"MAX_VALUE", "make_float4(MAXFLOAT)"}, {"OPENMM_METAL_REQUIRE_SAFE_MATH", "1"}};
    ComputeProgram program = context.compileProgram(source, defines);
    const uint32_t patterns[] = {0x7fc12345u, 0x3f800000u, 0xff800000u, 0x00000000u,
        0x80000000u, 0x7f800000u, 0xffc6789au, 0xc0000000u, 0x7f7fffffu, 0xff7fffffu,
        0x7fc00001u, 0x3f800000u};
    for (int count : {2, 3, 7, 15, 16, 17, 31, 32}) {
        vector<mm_float4> values(count);
        for (int i = 0; i < count; i++) {
            float key;
            uint32_t bits = patterns[i%12];
            memcpy(&key, &bits, sizeof(key));
            values[i] = mm_float4(key, i, 100+i, -100-i);
        }
        ComputeArray data;
        data.initialize<mm_float4>(context, count, "specialRegisterSortData");
        data.upload(values);
        ComputeKernel kernel = program->createKernel("sortShortList");
        kernel->addArg(data);
        kernel->addArg((unsigned int) count);
        static_cast<MetalKernel&>(*kernel).addLocalArg(count*sizeof(mm_float4));
        kernel->execute(32, 32);
        vector<mm_float4> result;
        data.download(result);
        stable_sort(values.begin(), values.end(), [](const mm_float4& a, const mm_float4& b) {
            const bool aNaN = isnan(a.x), bNaN = isnan(b.x);
            if (aNaN != bNaN) return !aNaN;
            return !aNaN && a.x < b.x;
        });
        // Bitwise comparison checks NaN payloads and -0 as well as stable record identity.
        for (int i = 0; i < count; i++) {
            uint32_t expected[4], actual[4];
            memcpy(expected, &values[i], sizeof(expected));
            memcpy(actual, &result[i], sizeof(actual));
            for (int j = 0; j < 4; j++) ASSERT_EQUAL(expected[j], actual[j]);
        }
    }
}

int main(int argc, char* argv[]) {
    try {
        testSelection();
        if (argc == 2 && string(argv[1]) == "--source-only") {
            cout << "Metal reduction/scan source-selection tests passed" << endl;
            return 0;
        }
        System system;
        system.addParticle(1.0);
        MetalContext context(system);
        testReductions(context);
        testCentroids(context);
        testNeighborScans(context);
        testBucketScans(context);
        testRegisterSort(context);
        testSafeMathHelperCalls(context);
        testRegisterSortSpecialValues(context);
    }
    catch (const exception& error) {
        cerr << "exception: " << error.what() << endl;
        return 1;
    }
    cout << "Metal reduction/scan optimization tests passed" << endl;
    return 0;
}
