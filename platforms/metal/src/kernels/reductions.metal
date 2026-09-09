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

#if __METAL_VERSION__ >= 300 && (OPENMM_METAL_FAST_CENTROID_REDUCTION || OPENMM_METAL_FAST_RG_REDUCTION || OPENMM_METAL_FAST_RMSD_REDUCTION || OPENMM_METAL_FAST_ORIENTATION_REDUCTION)
/**
 * @brief Sum within SIMD groups, then sum their totals using shared scratch.
 *
 * Every thread calls this helper.  The final rendezvous protects repeated
 * reductions that reuse scratch after every lane has read the result.
 */
DEVICE real metalReduceGroupValue(real value, LOCAL_ARG volatile real* temp) {
    const int lane = LOCAL_ID%32;
    const int warp = LOCAL_ID/32;
    const int active = min(32, (int) LOCAL_SIZE-32*warp);
    const int warps = ((int) LOCAL_SIZE+31)/32;
    for (int offset = 16; offset > 0; offset /= 2) {
        real other = simd_shuffle_down(value, offset);
        if (lane+offset < active)
            value += other;
    }
    SYNC_THREADS;
    if (lane == 0)
        temp[warp] = value;
    SYNC_THREADS;
    value = (LOCAL_ID < warps ? temp[LOCAL_ID] : (real) 0);
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset /= 2) {
            real other = simd_shuffle_down(value, offset);
            if (lane+offset < min(32, (int) LOCAL_SIZE))
                value += other;
        }
        if (lane == 0)
            temp[0] = value;
    }
    SYNC_THREADS;
    real result = temp[0];
    SYNC_THREADS;
    return result;
}
#endif

#if __METAL_VERSION__ >= 300 && (OPENMM_METAL_FAST_LCPO_NEIGHBOR_SCAN || OPENMM_METAL_FAST_CUSTOM_MANY_PARTICLE_NEIGHBOR_SCAN || OPENMM_METAL_FAST_SORT_BUCKET_SCAN)
/** @brief Inclusive uint scan over at most 1024 threads, with converged shuffles. */
DEVICE unsigned int metalBlockInclusiveScan(unsigned int value, LOCAL_ARG unsigned int* scratch) {
    const int lane = LOCAL_ID%32;
    const int warp = LOCAL_ID/32;
    const int warps = ((int) LOCAL_SIZE+31)/32;
    // Read every input before a warp total overwrites the same scratch array.
    SYNC_THREADS;
    for (int offset = 1; offset < 32; offset *= 2) {
        unsigned int other = simd_shuffle_up(value, offset);
        if (lane >= offset)
            value += other;
    }
    if (lane == 31 || LOCAL_ID == LOCAL_SIZE-1)
        scratch[warp] = value;
    SYNC_THREADS;
    unsigned int total = (LOCAL_ID < warps ? scratch[LOCAL_ID] : 0u);
    if (warp == 0) {
        for (int offset = 1; offset < 32; offset *= 2) {
            unsigned int other = simd_shuffle_up(total, offset);
            if (lane >= offset)
                total += other;
        }
        if (lane < warps)
            scratch[lane] = total;
    }
    SYNC_THREADS;
    unsigned int result = value+(warp == 0 ? 0u : scratch[warp-1]);
    SYNC_THREADS;
    return result;
}
#endif
