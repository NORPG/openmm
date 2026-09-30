/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Adapted from Common reduction kernels and OpenCL sort kernels.             *
 * Original OpenMM code:                                                      *
 * Portions copyright (c) 2008-2026 Stanford University and the Authors.       *
 * Authors: Peter Eastman, Evan Pretti                                         *
 * Metal Platform code:                                                       *
 * Portions copyright (c) 2026 Chun-Chi Hung.                                 *
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

#include "MetalReductionOptimizations.h"
#include "MetalKernelSources.h"
#include <sstream>

using namespace OpenMM;
using namespace std;

MetalReductionOptimizations::Settings MetalReductionOptimizations::getBuildSettings() {
    Settings settings;
#if OPENMM_METAL_FAST_CENTROID_REDUCTION
    settings.centroid = true;
#endif
#if OPENMM_METAL_FAST_RG_REDUCTION
    settings.rg = true;
#endif
#if OPENMM_METAL_FAST_RMSD_REDUCTION
    settings.rmsd = true;
#endif
#if OPENMM_METAL_FAST_ORIENTATION_REDUCTION
    settings.orientation = true;
#endif
#if OPENMM_METAL_FAST_LCPO_NEIGHBOR_SCAN
    settings.lcpoScan = true;
#endif
#if OPENMM_METAL_FAST_CUSTOM_MANY_PARTICLE_NEIGHBOR_SCAN
    settings.manyParticleScan = true;
#endif
#if OPENMM_METAL_FAST_SORT_BUCKET_SCAN
    settings.sortBucketScan = true;
#endif
#if OPENMM_METAL_FAST_SORT_REGISTER_BITONIC
    settings.sortRegisterBitonic = true;
#endif
    return settings;
}

string MetalReductionOptimizations::apply(const string& source) {
    return apply(source, getBuildSettings());
}

string MetalReductionOptimizations::apply(const string& source, const Settings& settings) {
    const string marker = "#define OPENMM_METAL_REDUCTION_SETTINGS 1\n";
    // Explicit caller settings (for example sort type eligibility or a test's
    // OFF control) take precedence over the build defaults on a second pass.
    if (source.find(marker) != string::npos)
        return source;
    string preamble = marker;
    auto define = [&](const string& name, bool enabled) {
        preamble += "#define "+name+(enabled ? " 1\n" : " 0\n");
    };
    define("OPENMM_METAL_FAST_CENTROID_REDUCTION", settings.centroid);
    define("OPENMM_METAL_FAST_RG_REDUCTION", settings.rg);
    define("OPENMM_METAL_FAST_RMSD_REDUCTION", settings.rmsd);
    define("OPENMM_METAL_FAST_ORIENTATION_REDUCTION", settings.orientation);
    define("OPENMM_METAL_FAST_LCPO_NEIGHBOR_SCAN", settings.lcpoScan);
    define("OPENMM_METAL_FAST_CUSTOM_MANY_PARTICLE_NEIGHBOR_SCAN", settings.manyParticleScan);
    define("OPENMM_METAL_FAST_SORT_BUCKET_SCAN", settings.sortBucketScan);
    define("OPENMM_METAL_FAST_SORT_REGISTER_BITONIC", settings.sortRegisterBitonic);
    // The kernel body is never inspected or modified. All alternatives are
    // selected by the shader preprocessor at their original source locations.
    // Native architecture kernels provide their own MSL reduction helpers.
    // Common Compute sources retain the shared portability helper adaptation.
    istringstream lines(source);
    string line;
    while (getline(lines, line))
        if (line == "// OPENMM_METAL_NATIVE_BODY")
            return preamble+source;
    return preamble+MetalKernelSources::reductions+source;
}
