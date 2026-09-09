/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Metal Platform code:                                                       *
 * Portions copyright (c) 2026 Chun-Chi Hung.                                 *
 * Authors: Chun-Chi Hung                                                     *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 * This program is distributed WITHOUT ANY WARRANTY; without even the        *
 * implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. *
 * See the GNU Lesser General Public License for more details.                *
 * You should have received a copy of the GNU Lesser General Public License  *
 * along with this program. If not, see <http://www.gnu.org/licenses/>.       *
 * -------------------------------------------------------------------------- */

#include "MetalContext.h"
#include "MetalSourceAdapter.h"
#include "CommonKernelSources.h"
#include "openmm/System.h"
#include "openmm/common/ComputeArray.h"
#include "openmm/internal/AssertionUtilities.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

using namespace OpenMM;
using namespace std;

/** Preserve source-owned fast-path branches while adapting only their ABI. */
void testSelection() {
    for (bool floating : {false, true}) {
        string groups = MetalSourceAdapter::translate(CommonKernelSources::customNonbondedGroups, floating);
        string many = MetalSourceAdapter::translate(CommonKernelSources::customManyParticle, floating);
        string lcpo = MetalSourceAdapter::translate(CommonKernelSources::lcpo, floating);
        // Both alternatives stay in the source until the Metal preprocessor
        // resolves each independent switch. The adapter must not select one.
        ASSERT(groups.find("OPENMM_METAL_FAST_CUSTOM_NONBONDED_GROUPS_SHUFFLE") != string::npos);
        ASSERT(many.find("OPENMM_METAL_FAST_CUSTOM_MANY_PARTICLE_BALLOT") != string::npos);
        ASSERT(lcpo.find("OPENMM_METAL_FAST_LCPO_BALLOT") != string::npos);
        for (const string& source : {groups, many, lcpo}) {
            ASSERT(source.find("#define __CUDA_ARCH__") == string::npos);
            ASSERT(source.find("#define USE_HIP") == string::npos);
            ASSERT(source.find("Scoped Metal CUDA-derived fast path") == string::npos);
            ASSERT(source.find("KERNEL_FUNCTION") == string::npos);
        }
    }
    // A user helper with the same name must not acquire an unrelated fast path.
    string unrelated = "DEVICE int reduceMax(int val, LOCAL_ARG int* temp) { return val; }\n"
        "KERNEL void probe(GLOBAL int* result) { result[GLOBAL_ID] = 0; }\n";
    ASSERT(MetalSourceAdapter::translate(unrelated).find("simd_shuffle_xor(") == string::npos);
    unrelated = "DEVICE void atomicAddMixed(GLOBAL mixed* target, mixed value) { *target = value; }\n"
        "KERNEL void probe(GLOBAL int* result) { result[GLOBAL_ID] = 0; }\n";
    ASSERT(MetalSourceAdapter::translate(unrelated).find("metalAtomicAdd(target, value)") == string::npos);
}

/** Private lane-storage types are ordinary MSL, not Common helper functions. */
void testSourceOwnedTypes() {
    const string types = R"(
#define ADAPTER_TEST_THREAD thread
template<class T> struct LaneValue {
    T value = T(0);
    ADAPTER_TEST_THREAD T& operator[](uint) ADAPTER_TEST_THREAD { return value; }
};
struct LaneRecord {
    float value;
    ADAPTER_TEST_THREAD float& operator[](uint) ADAPTER_TEST_THREAD { return value; }
};
#undef ADAPTER_TEST_THREAD
)";
    const string source = "#if USE_LANE_STORAGE\n"+types+R"(
#endif
DEVICE int firstValue(int value) { return value+LOCAL_ID; }
KERNEL void sourceOwnedTypes(GLOBAL int* output) {
#if USE_LANE_STORAGE
    LaneValue<int> lane;
    lane[0] = firstValue(7);
    output[GLOBAL_ID] = lane[0];
#else
    output[GLOBAL_ID] = firstValue(7);
#endif
}
)";
    for (bool floating : {false, true}) {
        const string translated = MetalSourceAdapter::translate(source, floating);
        ASSERT(translated.find(types) != string::npos);
        ASSERT(translated.find("#if USE_LANE_STORAGE") != string::npos);
        ASSERT(translated.find("firstValue(MetalExecutionContext _metal, int value)") != string::npos);
        ASSERT(translated.find("firstValue(_metal, 7)") != string::npos);
    }
}

/** Keep the version-gated shape attribute on tiled entry points, not helpers. */
void testTiledForceAttributes() {
    const string source = R"(
DEVICE void computeBornSumHelper() {}
KERNEL void computeNonbonded(GLOBAL int* output) { output[0] = 1; }
KERNEL void computeBornSum(GLOBAL int* output) { output[0] = 2; }
KERNEL void computeGBSAForce1(GLOBAL int* output) { output[0] = 3; }
KERNEL void probe(GLOBAL int* output) { output[0] = 4; }
)";
    for (bool floating : {false, true}) {
        string translated = MetalSourceAdapter::translate(source, floating);
        for (const string& name : {"computeNonbonded", "computeBornSum", "computeGBSAForce1"})
            ASSERT(translated.find("OPENMM_METAL_TILED_FORCE_THREADS\nkernel void "+name+"(") != string::npos);
        ASSERT(translated.find("OPENMM_METAL_TILED_FORCE_THREADS\nkernel void probe(") == string::npos);
        int attributes = 0;
        size_t position = 0;
        while ((position = translated.find("OPENMM_METAL_TILED_FORCE_THREADS", position)) != string::npos) {
            attributes++;
            position++;
        }
        ASSERT_EQUAL(3, attributes);
        // Shared helpers may read the selected MSL version, while the adapter
        // must leave the compiler's builtin version and shape selection intact.
        ASSERT(translated.find("required_threads_per_threadgroup") == string::npos);
        ASSERT(translated.find("#define __METAL_VERSION__") == string::npos);
        ASSERT(translated.find("#undef __METAL_VERSION__") == string::npos);
    }
}

/** Native MSL helpers and algorithm bodies remain byte-identical in both ABIs. */
void testNativeBodyABI() {
    const string helpers = R"(
inline float nativeHelper(MetalExecutionContext _metal, thread float* value) {
    return *value+float(_metal.localId);
}
)";
    const string body = R"(
    float value = 2.0f;
    float result = nativeHelper(_metal, &value);
    scratch[_metal.localId] = result;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    output[_metal.globalId] = scratch[_metal.localId];
)";
    const string source = "// OPENMM_METAL_NATIVE_BODY\n"+helpers+
            "KERNEL void nativeProbe(device float* output, threadgroup float* scratch) {"+body+"}\n";
    for (bool floating : {false, true}) {
        const string translated = MetalSourceAdapter::translate(source, floating);
        ASSERT(translated.find(helpers) != string::npos);
        ASSERT(translated.find(body) != string::npos);
        ASSERT(translated.find("_metal_nativeProbe_arguments") != string::npos);
        ASSERT(translated.find("_metal_local_scratch") != string::npos);
        ASSERT(translated.find("[[threadgroup(_metal_nativeProbe_id_scratch)]]") != string::npos);
        ASSERT(translated.find("nativeHelper(_metal, _metal,") == string::npos);
        ASSERT(translated.find("_metal_thread") == string::npos);
    }
}

/** Stress both float atomic primitives and their fetch-add return contract. */
void testFloatAtomics() {
    System system;
    system.addParticle(1);
    MetalContext context(system);
    const string& minimize = CommonKernelSources::minimize;
    size_t begin = minimize.find("DEVICE void atomicAddMixed(");
    size_t end = minimize.find("KERNEL void recordInitialPos(", begin);
    ASSERT(begin != string::npos && end != string::npos);
    const string source = minimize.substr(begin, end-begin)+R"(
KERNEL void accumulate(GLOBAL float* sums, GLOBAL float* previous, int count) {
    for (int i = GLOBAL_ID; i < count; i += GLOBAL_SIZE) {
        atomicAddMixed(sums, 1.0f);
        ATOMIC_ADD(sums+1, -1.0f);
        previous[i] = ATOMIC_ADD(sums+2, 1.0f);
    }
}

)";
    const int count = 10001;
    ComputeArray sums, previous, mode;
    sums.initialize<float>(context, 3, "floatAtomicSums");
    previous.initialize<float>(context, count, "floatAtomicPrevious");
    mode.initialize<int>(context, 1, "configuredFloatAtomicMode");
    ComputeKernel configured = context.compileProgram(
        "KERNEL void configuredMode(GLOBAL int* output) { if (GLOBAL_ID == 0) output[0] = OPENMM_METAL_NATIVE_FLOAT_ATOMICS; }")
        ->createKernel("configuredMode");
    configured->addArg(mode);
    configured->execute(32, 32);
    vector<int> selected;
    mode.download(selected);
    ASSERT_EQUAL(OPENMM_METAL_NATIVE_FLOAT_ATOMICS, selected[0]);
    for (int native = 0; native <= 1; native++) {
        // Per-program overrides verify both primitives even in the OFF build.
        map<string, string> defines;
        defines["OPENMM_METAL_NATIVE_FLOAT_ATOMICS"] = to_string(native);
        ComputeKernel kernel = context.compileProgram(source, defines)->createKernel("accumulate");
        kernel->addArg(sums);
        kernel->addArg(previous);
        kernel->addArg(count);
        context.clearBuffer(sums);
        kernel->execute(count);
        vector<float> values;
        sums.download(values);
        ASSERT_EQUAL(float(count), values[0]);
        ASSERT_EQUAL(-float(count), values[1]);
        ASSERT_EQUAL(float(count), values[2]);
        previous.download(values);
        sort(values.begin(), values.end());
        for (int i = 0; i < count; i++) ASSERT_EQUAL(float(i), values[i]);
    }
}

/** Verify the checked conversion and its hidden Common buffer binding on the GPU. */
void testFixedPointRangeDiagnostic() {
    System system;
    system.addParticle(1);
    MetalContext context(system);
    const string source = R"(
KERNEL void diagnosticState(GLOBAL uint* state, int action) {
    if (GLOBAL_ID != 0) return;
#if OPENMM_METAL_CHECK_FIXED_POINT_RANGE
    if (action == 1) {
        atomic_store_explicit(_metal.fixedPointRange, 1u, memory_order_relaxed);
        atomic_store_explicit(_metal.fixedPointRange+1, 0u, memory_order_relaxed);
    }
    if (action == 2)
        atomic_store_explicit(_metal.fixedPointRange, 0u, memory_order_relaxed);
    state[0] = 1;
    state[1] = atomic_load_explicit(_metal.fixedPointRange+1, memory_order_relaxed);
#else
    state[0] = 0;
    state[1] = 0;
#endif
}
KERNEL void convertChecked(GLOBAL const float* input, GLOBAL mm_long* output) {
    if (GLOBAL_ID < 8) output[GLOBAL_ID] = realToFixedPoint(input[GLOBAL_ID]);
}
)";
    ComputeArray state, input, output;
    state.initialize<unsigned int>(context, 2, "fixedPointDiagnosticState");
    input.initialize<float>(context, 8, "fixedPointDiagnosticInput");
    output.initialize<int64_t>(context, 8, "fixedPointDiagnosticOutput");
    ComputeProgram program = context.compileProgram(source);
    ComputeKernel inspect = program->createKernel("diagnosticState");
    inspect->addArg(state);
    inspect->addArg(1);
    inspect->execute(32, 32);
    vector<unsigned int> flags;
    state.download(flags);
    if (flags[0] == 0) return; // Float-minimization builds do not require this diagnostic ABI.
    ASSERT_EQUAL(0u, flags[1]);
    const float limit = 2147483648.0f;
    input.upload(vector<float>{1.25f, -3.5f, limit, -limit,
        numeric_limits<float>::infinity(), -numeric_limits<float>::infinity(),
        numeric_limits<float>::quiet_NaN(), nextafter(limit, 0.0f)});
    ComputeKernel convert = program->createKernel("convertChecked");
    convert->addArg(input);
    convert->addArg(output);
    convert->execute(32, 32);
    vector<int64_t> values;
    output.download(values);
    ASSERT_EQUAL(int64_t(5368709120), values[0]);
    ASSERT_EQUAL(int64_t(-15032385536), values[1]);
    for (int i = 2; i <= 6; i++) ASSERT_EQUAL(int64_t(0), values[i]);
    ASSERT_EQUAL(int64_t(2147483520)*int64_t(4294967296), values[7]);
    inspect->setArg(1, 0);
    inspect->execute(32, 32);
    state.download(flags);
    ASSERT_EQUAL(1u, flags[1]);
    inspect->setArg(1, 2);
    inspect->execute(32, 32);
}

/** Exercise the exact MSL3 primitives, including empty and lane-31-only masks. */
void testVoteAndShuffle() {
    System system;
    system.addParticle(1);
    MetalContext context(system);
    const string source = R"(
#include <metal_stdlib>
using namespace metal;
kernel void voteAndShuffle(device uint4* output [[buffer(0)]], constant uint& selected [[buffer(1)]],
        uint lane [[thread_index_in_simdgroup]], uint gid [[thread_position_in_grid]]) {
    uint bits = uint(simd_vote::vote_t(simd_ballot((selected & (1u << lane)) != 0)));
    uint remaining = bits, count = 0, sum = 0;
    while (remaining != 0) {
        int index = (ctz(remaining)+1)-1;
        remaining &= remaining-1;
        count++;
        sum += index;
    }
    uint maximum = gid;
    for (int mask = 16; mask > 0; mask /= 2)
        maximum = max(maximum, simd_shuffle_xor(maximum, mask));
    output[gid] = uint4(bits, count, sum, maximum);
}
)";
    ComputeArray output;
    output.initialize<mm_int4>(context, 64, "voteAndShuffleOutput");
    ComputeKernel kernel = context.compileProgram(source)->createKernel("voteAndShuffle");
    kernel->addArg(output);
    kernel->addArg(0u);
    for (unsigned int mask : {0u, 1u, 0x80000000u, 0xffffffffu, 5u}) {
        kernel->setArg(1, mask);
        kernel->execute(64, 64);
        vector<mm_int4> result;
        output.download(result);
        int count = 0, sum = 0;
        for (int lane = 0; lane < 32; lane++)
            if ((mask & (1u << lane)) != 0) { count++; sum += lane; }
        for (int i = 0; i < 64; i++) {
            ASSERT_EQUAL(mask, (unsigned int) result[i].x);
            ASSERT_EQUAL(count, result[i].y);
            ASSERT_EQUAL(sum, result[i].z);
            ASSERT_EQUAL(32*(i/32)+31, result[i].w);
        }
    }
}

int main() {
    try {
        testSelection();
        testSourceOwnedTypes();
        testTiledForceAttributes();
        testNativeBodyABI();
        testFloatAtomics();
        testFixedPointRangeDiagnostic();
        testVoteAndShuffle();
    }
    catch (const exception& error) {
        if (string(error.what()).find("No Metal device") != string::npos)
            return 77;
        cerr << error.what() << endl;
        return 1;
    }
    cout << "Metal source adapter tests passed" << endl;
    return 0;
}
