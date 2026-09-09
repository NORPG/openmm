/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * Native Metal nonbonded interactions.                                       *
 * Original OpenMM algorithms: copyright (c) Stanford University and Authors.  *
 * Metal implementation: copyright (c) 2026 Chun-Chi Hung.                     *
 * This program is free software under the GNU Lesser General Public License, *
 * version 3 or (at your option) any later version, without any warranty.       *
 * See <http://www.gnu.org/licenses/> for the license.                          *
 * -------------------------------------------------------------------------- */

// OPENMM_METAL_NATIVE_BODY

// The backend supplies Apple7-or-newer support, complete 32-lane SIMD groups,
// and a selected language version. Switches never raise that selected version.
#define METAL_NONBONDED_HYBRID 0
#if defined(__METAL__) && __METAL_VERSION__ >= 300 && OPENMM_METAL_NONBONDED_SHUFFLE && !METAL_NONBONDED_HYBRID
#define METAL_NONBONDED_SHUFFLE 1
#else
#define METAL_NONBONDED_SHUFFLE 0
#endif
#if TILE_SIZE != 32 || FORCE_WORK_GROUP_SIZE%32 != 0
#error Native Metal nonbonded tiles require complete 32-lane SIMD groups
#endif

// Preserve the existing formula hook's tiled vector width in each path.
#if METAL_NONBONDED_SHUFFLE
typedef float3 MetalNonbondedTileDerivative;
#else
typedef float4 MetalNonbondedTileDerivative;
#endif

// Global force planes keep the existing accumulator ABI. Tile counts and
// particle indices remain integers in both accumulation modes.
#ifdef OPENMM_METAL_FLOAT_ACCUMULATORS
typedef float MetalNonbondedAccumulator;
typedef float MetalNonbondedSignedAccumulator;
#else
typedef ulong MetalNonbondedAccumulator;
typedef long MetalNonbondedSignedAccumulator;
#endif

inline void metalNonbondedSynchronize() {
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

// Parameter-generation hooks use this pure lane transport helper. The two
// integer words preserve all signed and unsigned 64-bit parameter bits at 3.0.
template<class T> inline T real_shfl(T value, uint sourceLane) {
    return simd_shuffle(value, sourceLane&31u);
}
inline ulong real_shfl(ulong value, uint sourceLane) {
    uint lower = simd_shuffle(uint(value), sourceLane&31u);
    uint upper = simd_shuffle(uint(value>>32), sourceLane&31u);
    return ulong(lower)|(ulong(upper)<<32);
}
inline long real_shfl(long value, uint sourceLane) {
    return as_type<long>(real_shfl(as_type<ulong>(value), sourceLane));
}

inline float metalNonbondedSqrt(float value) {
#if OPENMM_METAL_USE_NATIVE_SQRT && !defined(OPENMM_METAL_FLOAT_ACCUMULATORS) && !defined(OPENMM_METAL_REQUIRE_SAFE_MATH)
    return metal::fast::sqrt(value);
#else
    return metal::sqrt(value);
#endif
}
inline float metalNonbondedReciprocalSqrt(float value) {
#if OPENMM_METAL_USE_NATIVE_RSQRT && !defined(OPENMM_METAL_FLOAT_ACCUMULATORS) && !defined(OPENMM_METAL_REQUIRE_SAFE_MATH)
    return metal::fast::rsqrt(value);
#else
    return metal::rsqrt(value);
#endif
}

inline MetalNonbondedAccumulator metalNonbondedEncodeForce(MetalExecutionContext _metal, float value) {
#ifdef OPENMM_METAL_FLOAT_ACCUMULATORS
    return value;
#else
#if OPENMM_METAL_CHECK_FIXED_POINT_RANGE
    return as_type<ulong>(metalRealToFixedPoint(_metal, value));
#else
    return as_type<ulong>(long(value*4294967296.0f));
#endif
#endif
}

inline void metalNonbondedSaveForce(MetalExecutionContext _metal,
        device MetalNonbondedAccumulator* buffer, uint atom, float3 value) {
    metalAtomicAdd(buffer+atom, metalNonbondedEncodeForce(_metal, value.x));
    metalAtomicAdd(buffer+atom+PADDED_NUM_ATOMS, metalNonbondedEncodeForce(_metal, value.y));
    metalAtomicAdd(buffer+atom+2*PADDED_NUM_ATOMS, metalNonbondedEncodeForce(_metal, value.z));
}

#ifdef USE_SPARSE_PAIRS
inline void metalNonbondedSaveSparseForce(MetalExecutionContext _metal,
        device MetalNonbondedAccumulator* buffer, uint atom, float3 value) {
    // Preserve the existing sparse path's zero guards and conversion-before-
    // aggregation contract. Floating accumulators use the ordinary float add.
    if (value.x != 0.0f)
        metalAccumulateSparseForce(_metal, buffer, atom, metalNonbondedEncodeForce(_metal, value.x));
    if (value.y != 0.0f)
        metalAccumulateSparseForce(_metal, buffer, atom+PADDED_NUM_ATOMS, metalNonbondedEncodeForce(_metal, value.y));
    if (value.z != 0.0f)
        metalAccumulateSparseForce(_metal, buffer, atom+2*PADDED_NUM_ATOMS, metalNonbondedEncodeForce(_metal, value.z));
}
#endif

#if !METAL_NONBONDED_SHUFFLE
struct MetalNonbondedTileData {
    float x, y, z, q;
    float fx, fy, fz;
    ATOM_PARAMETER_DATA
#ifndef PARAMETER_SIZE_IS_EVEN
    float padding;
#endif
};
#endif

// Entry adaptation supplies the existing argument-buffer boundary and _metal;
// these native bodies, helper signatures, and explicit calls remain untouched.
KERNEL void computeNonbonded(device MetalNonbondedAccumulator* forceBuffers,
        device float* energyBuffer, device const float4* posq, device const uint* exclusions,
        device const int2* exclusionTiles, uint startTileIndex, ulong numTileIndices
#ifdef USE_CUTOFF
        , device const int* tiles, device const uint* interactionCount,
        float4 periodicBoxSize, float4 invPeriodicBoxSize,
        float4 periodicBoxVecX, float4 periodicBoxVecY, float4 periodicBoxVecZ,
        uint maxTiles, device const float4* blockCenter,
        device const float4* blockSize, device const int* interactingAtoms
#ifdef USE_SPARSE_PAIRS
        , uint maxSinglePairs, device const int2* singlePairs
#endif
#endif
        PARAMETER_ARGUMENTS) {
    const uint totalThreads = _metal.localSize*_metal.numGroups;
    const uint totalWarps = totalThreads/32u;
    const uint warp = _metal.globalId/32u;
    const uint tgx = _metal.localId&31u;
    const uint tbx = _metal.localId-tgx;
    const uint localAtomIndex = _metal.localId;
    float energy = 0.0f;
    INIT_DERIVATIVES
#if !METAL_NONBONDED_SHUFFLE
    threadgroup MetalNonbondedTileData localData[FORCE_WORK_GROUP_SIZE];
#endif

    // Exclusion tiles are partitioned by SIMD group. Diagonal tiles visit
    // every partner in lane order; other tiles rotate the secondary owners.
    const uint first = FIRST_EXCLUSION_TILE+warp*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    const uint last = FIRST_EXCLUSION_TILE+(warp+1u)*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    for (uint tileIndex = first; tileIndex < last; ++tileIndex) {
        const int2 tile = exclusionTiles[tileIndex];
        const uint x = uint(tile.x), y = uint(tile.y);
        const uint atom1 = 32u*x+tgx;
        float4 posq1 = posq[atom1];
        float3 force = float3(0.0f);
        LOAD_ATOM1_PARAMETERS
        const bool hasExclusions = true;
#ifdef USE_EXCLUSIONS
        uint excl = exclusions[32u*tileIndex+tgx];
#endif
        if (x == y) {
#if METAL_NONBONDED_SHUFFLE
            float4 shflPosq = posq1;
#else
            localData[localAtomIndex].x = posq1.x;
            localData[localAtomIndex].y = posq1.y;
            localData[localAtomIndex].z = posq1.z;
            localData[localAtomIndex].q = posq1.w;
            LOAD_LOCAL_PARAMETERS_FROM_1
            metalNonbondedSynchronize();
#endif
            for (uint j = 0; j < 32u; ++j) {
                int atom2 = int(tbx+j);
#if METAL_NONBONDED_SHUFFLE
                float4 posq2;
                BROADCAST_WARP_DATA
#else
                float4 posq2 = float4(localData[atom2].x, localData[atom2].y, localData[atom2].z, localData[atom2].q);
#endif
                float3 delta = posq2.xyz-posq1.xyz;
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(delta)
#endif
                float r2 = delta.x*delta.x+delta.y*delta.y+delta.z*delta.z;
                float invR = metalNonbondedReciprocalSqrt(r2);
                float r = r2*invR;
                LOAD_ATOM2_PARAMETERS
                atom2 = int(32u*y+j);
#ifdef USE_EXCLUSIONS
                bool isExcluded = atom1 >= NUM_ATOMS || uint(atom2) >= NUM_ATOMS || !(excl&1u);
#endif
#ifdef USE_SYMMETRIC
                float dEdR = 0.0f;
#else
                MetalNonbondedTileDerivative dEdR1 = MetalNonbondedTileDerivative(0.0f);
                MetalNonbondedTileDerivative dEdR2 = MetalNonbondedTileDerivative(0.0f);
#endif
                float tempEnergy = 0.0f;
                const float interactionScale = 0.5f;
                COMPUTE_INTERACTION
                energy += 0.5f*tempEnergy;
#ifdef INCLUDE_FORCES
#ifdef USE_SYMMETRIC
                force -= delta*dEdR;
#else
                force -= dEdR1.xyz;
#endif
#endif
#ifdef USE_EXCLUSIONS
                excl >>= 1;
#endif
#if !METAL_NONBONDED_SHUFFLE
                metalNonbondedSynchronize();
#endif
            }
        }
        else {
            uint j = 32u*y+tgx;
#if METAL_NONBONDED_SHUFFLE
            float4 shflPosq = posq[j];
            float3 shflForce = float3(0.0f);
            DECLARE_LOCAL_PARAMETERS
#else
            float4 partner = posq[j];
            localData[localAtomIndex].x = partner.x;
            localData[localAtomIndex].y = partner.y;
            localData[localAtomIndex].z = partner.z;
            localData[localAtomIndex].q = partner.w;
#endif
            LOAD_LOCAL_PARAMETERS_FROM_GLOBAL
#if !METAL_NONBONDED_SHUFFLE
            localData[localAtomIndex].fx = 0.0f;
            localData[localAtomIndex].fy = 0.0f;
            localData[localAtomIndex].fz = 0.0f;
            metalNonbondedSynchronize();
#endif
#ifdef USE_EXCLUSIONS
            // Rotate-right has defined behavior even for lane zero.
            excl = (excl>>tgx)|(excl<<((32u-tgx)&31u));
#endif
            uint tj = tgx;
            for (j = 0; j < 32u; ++j) {
                int atom2 = int(tbx+tj);
#if METAL_NONBONDED_SHUFFLE
                float4 posq2 = shflPosq;
#else
                float4 posq2 = float4(localData[atom2].x, localData[atom2].y, localData[atom2].z, localData[atom2].q);
#endif
                float3 delta = posq2.xyz-posq1.xyz;
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(delta)
#endif
                float r2 = delta.x*delta.x+delta.y*delta.y+delta.z*delta.z;
#if defined(PRUNE_BY_CUTOFF) && !METAL_NONBONDED_SHUFFLE
                if (r2 < MAX_CUTOFF*MAX_CUTOFF) {
#endif
                    float invR = metalNonbondedReciprocalSqrt(r2);
                    float r = r2*invR;
                    LOAD_ATOM2_PARAMETERS
                    atom2 = int(32u*y+tj);
#ifdef USE_EXCLUSIONS
                    bool isExcluded = atom1 >= NUM_ATOMS || uint(atom2) >= NUM_ATOMS || !(excl&1u);
#endif
#ifdef USE_SYMMETRIC
                    float dEdR = 0.0f;
#else
                    MetalNonbondedTileDerivative dEdR1 = MetalNonbondedTileDerivative(0.0f);
                    MetalNonbondedTileDerivative dEdR2 = MetalNonbondedTileDerivative(0.0f);
#endif
                    float tempEnergy = 0.0f;
                    const float interactionScale = 1.0f;
                    COMPUTE_INTERACTION
                    energy += tempEnergy;
#ifdef INCLUDE_FORCES
#ifdef USE_SYMMETRIC
                    delta *= dEdR;
                    force -= delta;
                    float3 secondaryForce = delta;
#else
                    force -= dEdR1.xyz;
                    float3 secondaryForce = dEdR2.xyz;
#endif
#if METAL_NONBONDED_SHUFFLE
                    shflForce += secondaryForce;
#else
                    localData[tbx+tj].fx += secondaryForce.x;
                    localData[tbx+tj].fy += secondaryForce.y;
                    localData[tbx+tj].fz += secondaryForce.z;
#endif
#endif
#if defined(PRUNE_BY_CUTOFF) && !METAL_NONBONDED_SHUFFLE
                }
#endif
#if METAL_NONBONDED_SHUFFLE
                SHUFFLE_WARP_DATA
#else
                metalNonbondedSynchronize();
#endif
#ifdef USE_EXCLUSIONS
                excl >>= 1;
#endif
                tj = (tj+1u)&31u;
            }
#if defined(INCLUDE_FORCES) && METAL_NONBONDED_SHUFFLE
            metalNonbondedSaveForce(_metal, forceBuffers, 32u*y+tgx, shflForce);
#endif
        }
#ifdef INCLUDE_FORCES
        metalNonbondedSaveForce(_metal, forceBuffers, atom1, force);
#if !METAL_NONBONDED_SHUFFLE
        if (x != y)
            metalNonbondedSaveForce(_metal, forceBuffers, 32u*y+tgx,
                    float3(localData[localAtomIndex].fx, localData[localAtomIndex].fy, localData[localAtomIndex].fz));
#endif
#endif
    }

    // Ordinary tiles are either neighbor-list entries or the exclusion-free
    // part of the triangular all-pairs enumeration. Both use the same lane
    // visitation and accumulation order, with an optional single-image shift.
#ifdef USE_NEIGHBOR_LIST
    const uint count = interactionCount[0];
    if (count > maxTiles)
        return;
    int pos = int(ulong(warp)*count/totalWarps);
    const int end = int(ulong(warp+1u)*count/totalWarps);
#else
    int pos = int(startTileIndex+ulong(warp)*numTileIndices/totalWarps);
    const int end = int(startTileIndex+ulong(warp+1u)*numTileIndices/totalWarps);
    int skipBase = 0;
    uint currentSkipIndex = tbx;
#if METAL_NONBONDED_SHUFFLE
    int skipTile = -1;
#else
    threadgroup int skipTiles[FORCE_WORK_GROUP_SIZE];
    skipTiles[localAtomIndex] = -1;
#endif
#endif
#if !METAL_NONBONDED_SHUFFLE
    threadgroup uint atomIndices[FORCE_WORK_GROUP_SIZE];
#endif
    for (; pos < end; ++pos) {
        uint x, y = 0;
        bool singlePeriodicCopy = false;
#ifdef USE_NEIGHBOR_LIST
        x = uint(tiles[pos]);
        const float4 bounds = blockSize[x];
        singlePeriodicCopy = 0.5f*periodicBoxSize.x-bounds.x >= MAX_CUTOFF &&
                0.5f*periodicBoxSize.y-bounds.y >= MAX_CUTOFF &&
                0.5f*periodicBoxSize.z-bounds.z >= MAX_CUTOFF;
#else
        int row = int(floor(NUM_BLOCKS+0.5f-metalNonbondedSqrt((NUM_BLOCKS+0.5f)*(NUM_BLOCKS+0.5f)-2*pos)));
        int column = pos-row*NUM_BLOCKS+row*(row+1)/2;
        if (column < row || column >= NUM_BLOCKS) {
            row += column < row ? -1 : 1;
            column = pos-row*NUM_BLOCKS+row*(row+1)/2;
        }
        x = uint(column);
        y = uint(row);
#if METAL_NONBONDED_SHUFFLE
        while (real_shfl(skipTile, 31u) < pos) {
#else
        metalNonbondedSynchronize();
        while (skipTiles[tbx+31u] < pos) {
            metalNonbondedSynchronize();
#endif
            int next = end;
            if (skipBase+int(tgx) < NUM_TILES_WITH_EXCLUSIONS) {
                const int2 excluded = exclusionTiles[skipBase+tgx];
                next = excluded.x+excluded.y*NUM_BLOCKS-excluded.y*(excluded.y+1)/2;
            }
#if METAL_NONBONDED_SHUFFLE
            skipTile = next;
#else
            skipTiles[localAtomIndex] = next;
            metalNonbondedSynchronize();
#endif
            skipBase += 32;
            currentSkipIndex = tbx;
        }
#if METAL_NONBONDED_SHUFFLE
        while (real_shfl(skipTile, currentSkipIndex-tbx) < pos)
            ++currentSkipIndex;
        if (real_shfl(skipTile, currentSkipIndex-tbx) == pos)
            continue;
#else
        while (skipTiles[currentSkipIndex] < pos)
            ++currentSkipIndex;
        if (skipTiles[currentSkipIndex] == pos)
            continue;
#endif
#endif
        const bool hasExclusions = false;
        const uint atom1 = 32u*x+tgx;
        float4 posq1 = posq[atom1];
        float3 force = float3(0.0f);
        LOAD_ATOM1_PARAMETERS
#ifdef USE_NEIGHBOR_LIST
        uint j = uint(interactingAtoms[32*pos+tgx]);
#else
        uint j = 32u*y+tgx;
#endif
#if METAL_NONBONDED_SHUFFLE
        const uint shflAtomIndex = j;
        float4 shflPosq = float4(0.0f);
        float3 shflForce = float3(0.0f);
#else
        atomIndices[localAtomIndex] = j;
#endif
        DECLARE_LOCAL_PARAMETERS
        if (j < PADDED_NUM_ATOMS) {
#if METAL_NONBONDED_SHUFFLE
            shflPosq = posq[j];
#else
            float4 partner = posq[j];
            localData[localAtomIndex].x = partner.x;
            localData[localAtomIndex].y = partner.y;
            localData[localAtomIndex].z = partner.z;
            localData[localAtomIndex].q = partner.w;
#endif
            LOAD_LOCAL_PARAMETERS_FROM_GLOBAL
        }
        else {
#if !METAL_NONBONDED_SHUFFLE && !METAL_NONBONDED_HYBRID
            localData[localAtomIndex].x = 0.0f;
            localData[localAtomIndex].y = 0.0f;
            localData[localAtomIndex].z = 0.0f;
            localData[localAtomIndex].q = 0.0f;
#endif
            CLEAR_LOCAL_PARAMETERS
        }
#if !METAL_NONBONDED_SHUFFLE
        localData[localAtomIndex].fx = 0.0f;
        localData[localAtomIndex].fy = 0.0f;
        localData[localAtomIndex].fz = 0.0f;
        metalNonbondedSynchronize();
#endif
#ifdef USE_PERIODIC
        if (singlePeriodicCopy) {
            float4 blockCenterX = blockCenter[x];
            APPLY_PERIODIC_TO_POS_WITH_CENTER(posq1, blockCenterX)
#if METAL_NONBONDED_SHUFFLE
            APPLY_PERIODIC_TO_POS_WITH_CENTER(shflPosq, blockCenterX)
#else
            APPLY_PERIODIC_TO_POS_WITH_CENTER(localData[localAtomIndex], blockCenterX)
#endif
#if !METAL_NONBONDED_SHUFFLE
            metalNonbondedSynchronize();
#endif
        }
#endif
        uint tj = tgx;
        for (j = 0; j < 32u; ++j) {
            int atom2 = int(tbx+tj);
#if METAL_NONBONDED_SHUFFLE
            float4 posq2 = shflPosq;
#else
            float4 posq2 = float4(localData[atom2].x, localData[atom2].y, localData[atom2].z, localData[atom2].q);
#endif
            float3 delta = posq2.xyz-posq1.xyz;
#ifdef USE_PERIODIC
            if (!singlePeriodicCopy) {
                APPLY_PERIODIC_TO_DELTA(delta)
            }
#endif
            float r2 = delta.x*delta.x+delta.y*delta.y+delta.z*delta.z;
#if defined(PRUNE_BY_CUTOFF) && !METAL_NONBONDED_SHUFFLE
            if (r2 < MAX_CUTOFF*MAX_CUTOFF) {
#endif
                float invR = metalNonbondedReciprocalSqrt(r2);
                float r = r2*invR;
                LOAD_ATOM2_PARAMETERS
#if METAL_NONBONDED_SHUFFLE
                atom2 = int(real_shfl(shflAtomIndex, tj));
#else
                atom2 = int(atomIndices[tbx+tj]);
#endif
#ifdef USE_EXCLUSIONS
                bool isExcluded = atom1 >= NUM_ATOMS || uint(atom2) >= NUM_ATOMS;
#endif
#ifdef USE_SYMMETRIC
                float dEdR = 0.0f;
#else
                MetalNonbondedTileDerivative dEdR1 = MetalNonbondedTileDerivative(0.0f);
                MetalNonbondedTileDerivative dEdR2 = MetalNonbondedTileDerivative(0.0f);
#endif
                float tempEnergy = 0.0f;
                const float interactionScale = 1.0f;
                COMPUTE_INTERACTION
                energy += tempEnergy;
#ifdef INCLUDE_FORCES
#ifdef USE_SYMMETRIC
                delta *= dEdR;
                force -= delta;
                float3 secondaryForce = delta;
#else
                force -= dEdR1.xyz;
                float3 secondaryForce = dEdR2.xyz;
#endif
#if METAL_NONBONDED_SHUFFLE
                shflForce += secondaryForce;
#else
                localData[tbx+tj].fx += secondaryForce.x;
                localData[tbx+tj].fy += secondaryForce.y;
                localData[tbx+tj].fz += secondaryForce.z;
#endif
#endif
#if defined(PRUNE_BY_CUTOFF) && !METAL_NONBONDED_SHUFFLE
            }
#endif
#if METAL_NONBONDED_SHUFFLE
            SHUFFLE_WARP_DATA
#else
            metalNonbondedSynchronize();
#endif
            tj = (tj+1u)&31u;
        }
#ifdef INCLUDE_FORCES
        metalNonbondedSaveForce(_metal, forceBuffers, atom1, force);
#if METAL_NONBONDED_SHUFFLE
        const uint secondaryAtom = shflAtomIndex;
        float3 secondaryForce = shflForce;
#else
        const uint secondaryAtom = atomIndices[localAtomIndex];
        float3 secondaryForce = float3(localData[localAtomIndex].fx, localData[localAtomIndex].fy, localData[localAtomIndex].fz);
#endif
        if (secondaryAtom < PADDED_NUM_ATOMS)
            metalNonbondedSaveForce(_metal, forceBuffers, secondaryAtom, secondaryForce);
#endif
    }

    // Separate sparse pairs retain the direct global parameter loads and the
    // same atom/component ordering for their fixed or floating reductions.
#ifdef USE_SPARSE_PAIRS
    const uint pairCount = interactionCount[1];
    if (pairCount > maxSinglePairs)
        return;
    for (uint pairIndex = _metal.globalId; pairIndex < pairCount; pairIndex += totalThreads) {
        const int2 pair = singlePairs[pairIndex];
        const int atom1 = pair.x, atom2 = pair.y;
        float4 posq1 = posq[atom1], posq2 = posq[atom2];
        LOAD_ATOM1_PARAMETERS
        LOAD_ATOM2_PARAMETERS_FROM_GLOBAL
        float3 delta = posq2.xyz-posq1.xyz;
#ifdef USE_PERIODIC
        APPLY_PERIODIC_TO_DELTA(delta)
#endif
        float r2 = delta.x*delta.x+delta.y*delta.y+delta.z*delta.z;
        float invR = metalNonbondedReciprocalSqrt(r2);
        float r = r2*invR;
        const bool hasExclusions = false;
        const bool isExcluded = false;
#ifdef USE_SYMMETRIC
        float dEdR = 0.0f;
#else
        float3 dEdR1 = float3(0.0f), dEdR2 = float3(0.0f);
#endif
        float tempEnergy = 0.0f;
        const float interactionScale = 1.0f;
        COMPUTE_INTERACTION
        energy += tempEnergy;
#ifdef INCLUDE_FORCES
#ifdef USE_SYMMETRIC
        float3 firstForce = -delta*dEdR;
        float3 secondForce = -firstForce;
#else
        float3 firstForce = -dEdR1.xyz;
        float3 secondForce = -dEdR2.xyz;
#endif
        metalNonbondedSaveSparseForce(_metal, forceBuffers, uint(atom1), firstForce);
        metalNonbondedSaveSparseForce(_metal, forceBuffers, uint(atom2), secondForce);
#endif
    }
#endif
#ifdef INCLUDE_ENERGY
    energyBuffer[_metal.globalId] += energy;
#endif
    SAVE_DERIVATIVES
}

#undef METAL_NONBONDED_HYBRID
#undef METAL_NONBONDED_SHUFFLE
