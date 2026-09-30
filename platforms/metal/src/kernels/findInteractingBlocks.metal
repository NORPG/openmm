// OPENMM_METAL_NATIVE_BODY
/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * Neighbor-list algorithms adapted from OpenMM's OpenCL and HIP Platforms.   *
 * Original code: copyright (c) Stanford University and the Authors.           *
 * Authors: Peter Eastman and the OpenMM contributors.                        *
 * Metal code: copyright (c) 2026 Chun-Chi Hung. Author: Chun-Chi Hung.          *
 * This program is free software under the GNU Lesser General Public License, *
 * version 3 or (at your option) any later version, without any warranty.      *
 * See <http://www.gnu.org/licenses/> for the license.                         *
 * -------------------------------------------------------------------------- */

// The entry adapter supplies _metal and the existing argument-buffer ABI.
// The host validates 32-lane SIMD groups and dispatches complete groups.
#if defined(SIMD_WIDTH) && SIMD_WIDTH != 32
#error Metal neighbor lists require a 32-lane SIMD group
#endif

typedef float4 MetalNeighborBound;
inline MetalNeighborBound metalNeighborStoreBound(float4 value) { return value; }

/** Keep the pre-existing independently selected logarithm accuracy policy. */
inline float metalNeighborLog(float value) {
#if OPENMM_METAL_USE_NATIVE_LOG && !defined(OPENMM_METAL_FLOAT_ACCUMULATORS) && !defined(OPENMM_METAL_REQUIRE_SAFE_MATH)
    return metal::fast::log(value);
#else
    return log(value);
#endif
}

/** Only allocation of output ranges is concurrent; consumers use later dispatches. */
inline uint metalNeighborReserve(device uint* counter, uint count) {
    return atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint*>(counter), count, memory_order_relaxed);
}

/** Nearest-image displacement, preserving the ordered triclinic reductions. */
inline float3 metalNeighborDelta(float3 delta, float4 periodicBoxSize, float4 invPeriodicBoxSize,
        float4 periodicBoxVecX, float4 periodicBoxVecY, float4 periodicBoxVecZ) {
#ifdef TRICLINIC
    float scale = floor(delta.z*invPeriodicBoxSize.z+0.5f);
    delta -= scale*periodicBoxVecZ.xyz;
    scale = floor(delta.y*invPeriodicBoxSize.y+0.5f);
    delta.xy -= scale*periodicBoxVecY.xy;
    scale = floor(delta.x*invPeriodicBoxSize.x+0.5f);
    delta.x -= scale*periodicBoxVecX.x;
#else
    delta -= floor(delta*invPeriodicBoxSize.xyz+0.5f)*periodicBoxSize.xyz;
#endif
    return delta;
}

/** Translate a position into the primary box or the box nearest a center. */
inline float3 metalNeighborPosition(float3 pos, float3 center, bool useCenter, float4 periodicBoxSize,
        float4 invPeriodicBoxSize, float4 periodicBoxVecX, float4 periodicBoxVecY, float4 periodicBoxVecZ) {
#ifdef TRICLINIC
    float scale = useCenter ? floor((pos.z-center.z)*invPeriodicBoxSize.z+0.5f) : floor(pos.z*invPeriodicBoxSize.z);
    pos -= scale*periodicBoxVecZ.xyz;
    scale = useCenter ? floor((pos.y-center.y)*invPeriodicBoxSize.y+0.5f) : floor(pos.y*invPeriodicBoxSize.y);
    pos.xy -= scale*periodicBoxVecY.xy;
    scale = useCenter ? floor((pos.x-center.x)*invPeriodicBoxSize.x+0.5f) : floor(pos.x*invPeriodicBoxSize.x);
    pos.x -= scale*periodicBoxVecX.x;
#else
    float3 scale = useCenter ? floor((pos-center)*invPeriodicBoxSize.xyz+0.5f) : floor(pos*invPeriodicBoxSize.xyz);
    pos -= scale*periodicBoxSize.xyz;
#endif
    return pos;
}

/** Find per-block boxes and the range of box sizes, using 64-thread groups. */
KERNEL void findBlockBounds(int numAtoms, float4 periodicBoxSize, float4 invPeriodicBoxSize,
        float4 periodicBoxVecX, float4 periodicBoxVecY, float4 periodicBoxVecZ,
        device const float4* posq, device float4* blockCenter, device float4* blockBoundingBox,
        device int* rebuildNeighborList, device float2* blockSizeRange) {
    const int localId = int(_metal.localId);
    const int globalId = int(_metal.globalId);
    const int globalSize = int(_metal.localSize*_metal.numGroups);
    float minSize = 1e38f, maxSize = 0;
    // Baseline: each thread expands one block serially, with the same periodic
    // image choices and radius expression as the optimized periodic path.
    for (int index = globalId; index*TILE_SIZE < numAtoms; index += globalSize) {
        int base = index*TILE_SIZE;
        int last = min(base+TILE_SIZE, numAtoms);
        float4 pos = posq[base];
#ifdef USE_PERIODIC
        pos.xyz = metalNeighborPosition(pos.xyz, float3(0), false, periodicBoxSize, invPeriodicBoxSize,
                periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
#endif
        float4 minPos = pos, maxPos = pos;
        for (int atom = base+1; atom < last; atom++) {
            pos = posq[atom];
#ifdef USE_PERIODIC
            float4 center = 0.5f*(maxPos+minPos);
            pos.xyz = metalNeighborPosition(pos.xyz, center.xyz, true, periodicBoxSize, invPeriodicBoxSize,
                    periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
#endif
            minPos = min(minPos, pos);
            maxPos = max(maxPos, pos);
        }
        float4 width = 0.5f*(maxPos-minPos);
        float4 center = 0.5f*(maxPos+minPos);
        center.w = 0;
        for (int atom = base; atom < last; atom++) {
            float4 delta = posq[atom]-center;
#ifdef USE_PERIODIC
            delta.xyz = metalNeighborDelta(delta.xyz, periodicBoxSize, invPeriodicBoxSize,
                    periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
#endif
            center.w = max(center.w, delta.x*delta.x+delta.y*delta.y+delta.z*delta.z);
        }
        center.w = sqrt(center.w);
        blockCenter[index] = center;
        blockBoundingBox[index] = width;
        float size = width.x+width.y+width.z;
        minSize = min(minSize, size);
        maxSize = max(maxSize, size);
    }
    threadgroup float minBuffer[64], maxBuffer[64];
    minBuffer[localId] = minSize;
    maxBuffer[localId] = maxSize;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int step = 1; step < 64; step *= 2) {
        if (localId+step < 64 && localId%(2*step) == 0) {
            minBuffer[localId] = min(minBuffer[localId], minBuffer[localId+step]);
            maxBuffer[localId] = max(maxBuffer[localId], maxBuffer[localId+step]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (localId == 0)
        blockSizeRange[_metal.groupId] = float2(minBuffer[0], maxBuffer[0]);
    if (globalId == 0)
        rebuildNeighborList[0] = 0;
}

/** Encode the logarithmic size bin in the high bits and block index below it. */
KERNEL void computeSortKeys(device const float4* blockBoundingBox, device uint* sortedBlocks,
        device float2* blockSizeRange, int numSizes) {
    threadgroup float2 sizeRange;
    if (_metal.localId == 0) {
        sizeRange = blockSizeRange[0];
        for (int i = 1; i < numSizes; i++) {
            float2 size = blockSizeRange[i];
            if (size.x > 0)
                sizeRange.x = min(sizeRange.x, size.x);
            sizeRange.y = max(sizeRange.y, size.y);
        }
        sizeRange.x = metalNeighborLog(sizeRange.x);
        sizeRange.y = metalNeighborLog(sizeRange.y);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int numSizeBins = 20;
    float scale = numSizeBins/(sizeRange.y-sizeRange.x);
    for (uint i = _metal.globalId; i < NUM_BLOCKS; i += _metal.localSize*_metal.numGroups) {
        float4 box = blockBoundingBox[i];
        float size = metalNeighborLog(box.x+box.y+box.z);
        int bin = int((size-sizeRange.x)*scale);
        bin = max(0, min(bin, numSizeBins-1));
        sortedBlocks[i] = (uint(bin)<<BIN_SHIFT)+i;
    }
}

/** Reorder bounding boxes and determine whether the neighbor list is stale. */
KERNEL void sortBoxData(device const uint* sortedBlocks, device const float4* blockCenter,
        device const float4* blockBoundingBox, device float4* sortedBlockCenter,
        device MetalNeighborBound* sortedBlockBoundingBox, device const float4* posq,
        device const float4* oldPositions, device uint* interactionCount,
        device int* rebuildNeighborList, int forceRebuild
#ifdef USE_LARGE_BLOCKS
        , device float4* largeBlockCenter, device MetalNeighborBound* largeBlockBoundingBox,
        float4 periodicBoxSize, float4 invPeriodicBoxSize, float4 periodicBoxVecX,
        float4 periodicBoxVecY, float4 periodicBoxVecZ
#endif
        ) {
    const int globalSize = int(_metal.localSize*_metal.numGroups);
    for (int i = int(_metal.globalId); i < NUM_BLOCKS; i += globalSize) {
        uint index = sortedBlocks[i]&BLOCK_INDEX_MASK;
        sortedBlockCenter[i] = blockCenter[index];
        sortedBlockBoundingBox[i] = metalNeighborStoreBound(blockBoundingBox[index]);
#ifdef USE_LARGE_BLOCKS
        float4 minPos = blockCenter[index]-blockBoundingBox[index];
        float4 maxPos = blockCenter[index]+blockBoundingBox[index];
        int last = min(i+32, NUM_BLOCKS);
        for (int j = i+1; j < last; j++) {
            uint index2 = sortedBlocks[j]&BLOCK_INDEX_MASK;
            float4 blockPos = blockCenter[index2];
            float4 width = blockBoundingBox[index2];
#ifdef USE_PERIODIC
            float4 center = 0.5f*(maxPos+minPos);
            blockPos.xyz = metalNeighborPosition(blockPos.xyz, center.xyz, true, periodicBoxSize, invPeriodicBoxSize,
                    periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
#endif
            minPos = min(minPos, blockPos-width);
            maxPos = max(maxPos, blockPos+width);
        }
        largeBlockCenter[i] = 0.5f*(maxPos+minPos);
        largeBlockBoundingBox[i] = metalNeighborStoreBound(0.5f*(maxPos-minPos));
#endif
    }
    bool rebuild = forceRebuild;
    for (int i = int(_metal.globalId); i < NUM_ATOMS; i += globalSize) {
        float4 delta = oldPositions[i]-posq[i];
        if (delta.x*delta.x+delta.y*delta.y+delta.z*delta.z > 0.25f*PADDING*PADDING)
            rebuild = true;
    }
    if (rebuild) {
        rebuildNeighborList[0] = 1;
        interactionCount[0] = 0;
    }
}

/**
 * Search one first block per SIMD group. Each lane initially screens a second
 * block, then the whole SIMD group tests every selected second block's atoms.
 * Output tile and sparse-pair allocation preserve the existing buffer layouts.
 */
KERNEL void findBlocksWithInteractions(float4 periodicBoxSize, float4 invPeriodicBoxSize,
        float4 periodicBoxVecX, float4 periodicBoxVecY, float4 periodicBoxVecZ,
        device uint* interactionCount, device int* interactingTiles, device uint* interactingAtoms,
        device const float4* posq, uint maxTiles, uint startBlockIndex, uint numBlocks,
        device uint* sortedBlocks, device const float4* sortedBlockCenter,
        device const MetalNeighborBound* sortedBlockBoundingBox, device const uint* exclusionIndices,
        device const uint* exclusionRowIndices, device float4* oldPositions,
        device const int* rebuildNeighborList
#ifdef USE_LARGE_BLOCKS
        , device float4* largeBlockCenter, device MetalNeighborBound* largeBlockBoundingBox
#endif
        ) {
    if (rebuildNeighborList[0] == 0)
        return;
    const int localId = int(_metal.localId);
    const int lane = localId%32;
    const int simdStart = localId-lane;
    const int globalSize = int(_metal.localSize*_metal.numGroups);
    const int totalSimdGroups = globalSize/32;
    const int simdIndex = int(_metal.globalId)/32;
    const int bufferSize = 256;
    threadgroup int workgroupBuffer[256*(GROUP_SIZE/32)];
    threadgroup int groupExclusions[MAX_EXCLUSIONS*(GROUP_SIZE/32)];
    threadgroup float3 positions[GROUP_SIZE];
    threadgroup volatile uint tileIndices[GROUP_SIZE/32];
    threadgroup bool includeFlags[GROUP_SIZE];
    threadgroup volatile short2 atomCounts[GROUP_SIZE];
#ifdef USE_LARGE_BLOCKS
    threadgroup bool largeFlags[GROUP_SIZE];
#endif
    threadgroup int* buffer = workgroupBuffer+bufferSize*(simdStart/32);
    threadgroup int* exclusions = groupExclusions+MAX_EXCLUSIONS*(simdStart/32);
    threadgroup volatile uint* tileIndex = tileIndices+(simdStart/32);

    for (int block1 = int(startBlockIndex)+simdIndex; block1 < startBlockIndex+numBlocks; block1 += totalSimdGroups) {
        int x = sortedBlocks[block1]&BLOCK_INDEX_MASK;
        float4 centerX = sortedBlockCenter[block1];
        float4 widthX = float4(sortedBlockBoundingBox[block1]);
        int neighbors = 0;
        float3 pos1 = posq[x*TILE_SIZE+lane].xyz;
#ifdef USE_PERIODIC
        const bool singlePeriodicCopy = (0.5f*periodicBoxSize.x-widthX.x >= PADDED_CUTOFF &&
                0.5f*periodicBoxSize.y-widthX.y >= PADDED_CUTOFF &&
                0.5f*periodicBoxSize.z-widthX.z >= PADDED_CUTOFF);
        if (singlePeriodicCopy)
            pos1 = metalNeighborPosition(pos1, centerX.xyz, true, periodicBoxSize, invPeriodicBoxSize,
                    periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
#endif
        positions[localId] = pos1;
        const int exclusionStart = exclusionRowIndices[x];
        const int exclusionEnd = exclusionRowIndices[x+1];
        const int numExclusions = exclusionEnd-exclusionStart;
        for (int i = lane; i < numExclusions; i += 32)
            exclusions[i] = exclusionIndices[exclusionStart+i];
        // Each SIMD group owns these exclusions and positions. More than 32
        // exclusions adds load iterations, without adding cross-SIMD consumers.
        // The independent block loop must not rendezvous the whole threadgroup.
        simdgroup_barrier(mem_flags::mem_threadgroup);
#ifdef USE_LARGE_BLOCKS
        int loadedLargeBlocks = 0;
#endif
        for (int block2Base = block1+1; block2Base < NUM_BLOCKS; block2Base += 32) {
#ifdef USE_LARGE_BLOCKS
            if (loadedLargeBlocks == 0) {
                int largeBlockIndex = block2Base+32*lane;
                bool includeLarge = false;
                if (largeBlockIndex < NUM_BLOCKS) {
                    float4 largeCenter = largeBlockCenter[largeBlockIndex];
                    float4 largeWidth = float4(largeBlockBoundingBox[largeBlockIndex]);
                    float4 delta = centerX-largeCenter;
#ifdef USE_PERIODIC
                    delta.xyz = metalNeighborDelta(delta.xyz, periodicBoxSize, invPeriodicBoxSize,
                            periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
#endif
                    delta.x = max(0.0f, fabs(delta.x)-widthX.x-largeWidth.x);
                    delta.y = max(0.0f, fabs(delta.y)-widthX.y-largeWidth.y);
                    delta.z = max(0.0f, fabs(delta.z)-widthX.z-largeWidth.z);
                    includeLarge = (delta.x*delta.x+delta.y*delta.y+delta.z*delta.z < PADDED_CUTOFF_SQUARED);
#ifdef TRICLINIC
                    // Retain the conservative fallback when the nearest-image
                    // box test cannot guarantee that it found the closest copy.
                    if (periodicBoxSize.z/2-widthX.z-largeWidth.z < PADDED_CUTOFF ||
                            periodicBoxSize.y/2-widthX.y-largeWidth.y < PADDED_CUTOFF)
                        includeLarge = true;
#endif
                }
                largeFlags[localId] = includeLarge;
                loadedLargeBlocks = 32;
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (!largeFlags[simdStart+32-(loadedLargeBlocks--)])
                continue;
#endif
            int block2 = block2Base+lane;
            bool include = block2 < NUM_BLOCKS;
            if (include) {
                float4 centerY = sortedBlockCenter[block2];
                float4 widthY = float4(sortedBlockBoundingBox[block2]);
                float4 delta = centerX-centerY;
#ifdef USE_PERIODIC
                delta.xyz = metalNeighborDelta(delta.xyz, periodicBoxSize, invPeriodicBoxSize,
                        periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
#endif
                include &= (delta.x*delta.x+delta.y*delta.y+delta.z*delta.z <
                        (PADDED_CUTOFF+centerX.w+centerY.w)*(PADDED_CUTOFF+centerX.w+centerY.w));
                delta.x = max(0.0f, fabs(delta.x)-widthX.x-widthY.x);
                delta.y = max(0.0f, fabs(delta.y)-widthX.y-widthY.y);
                delta.z = max(0.0f, fabs(delta.z)-widthX.z-widthY.z);
                include &= (delta.x*delta.x+delta.y*delta.y+delta.z*delta.z < PADDED_CUTOFF_SQUARED);
#ifdef TRICLINIC
                if (periodicBoxSize.z/2-widthX.z-widthY.z < PADDED_CUTOFF ||
                        periodicBoxSize.y/2-widthX.y-widthY.y < PADDED_CUTOFF)
                    include = true;
#endif
                if (include) {
                    int y = sortedBlocks[block2]&BLOCK_INDEX_MASK;
                    for (int i = 0; i < numExclusions; i++)
                        include &= exclusions[i] != y;
                }
            }

            includeFlags[localId] = include;
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (int selected = 0; selected < TILE_SIZE; selected++) {
                while (selected < TILE_SIZE && !includeFlags[simdStart+selected])
                    selected++;
                if (selected < TILE_SIZE) {
                    int y = sortedBlocks[block2Base+selected]&BLOCK_INDEX_MASK;
                    int atom2 = y*TILE_SIZE+lane;
                    float3 pos2 = posq[atom2].xyz;
#ifdef USE_PERIODIC
                    if (singlePeriodicCopy)
                        pos2 = metalNeighborPosition(pos2, centerX.xyz, true, periodicBoxSize, invPeriodicBoxSize,
                                periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
#endif
                    bool interacts = false;
                    if (atom2 < NUM_ATOMS) {
#ifdef USE_PERIODIC
                        if (!singlePeriodicCopy) {
                            for (int atom1 = 0; atom1 < TILE_SIZE; atom1++) {
                                float3 delta = pos2-positions[simdStart+atom1];
                                delta = metalNeighborDelta(delta, periodicBoxSize, invPeriodicBoxSize,
                                        periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
                                interacts |= (delta.x*delta.x+delta.y*delta.y+delta.z*delta.z < PADDED_CUTOFF_SQUARED);
                            }
                        }
                        else {
#endif
                            for (int atom1 = 0; atom1 < TILE_SIZE; atom1++) {
                                float3 delta = pos2-positions[simdStart+atom1];
                                interacts |= (delta.x*delta.x+delta.y*delta.y+delta.z*delta.z < PADDED_CUTOFF_SQUARED);
                            }
#ifdef USE_PERIODIC
                        }
#endif
                    }

                    atomCounts[localId].x = interacts ? 1 : 0;
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                    int whichBuffer = 0;
                    for (int offset = 1; offset < TILE_SIZE; offset *= 2) {
                        if (whichBuffer == 0)
                            atomCounts[localId].y = lane < offset ? atomCounts[localId].x :
                                    atomCounts[localId].x+atomCounts[localId-offset].x;
                        else
                            atomCounts[localId].x = lane < offset ? atomCounts[localId].y :
                                    atomCounts[localId].y+atomCounts[localId-offset].y;
                        whichBuffer = 1-whichBuffer;
                        simdgroup_barrier(mem_flags::mem_threadgroup);
                    }
                    if (interacts)
                        buffer[neighbors+atomCounts[localId].y-1] = atom2;
                    neighbors += atomCounts[simdStart+TILE_SIZE-1].y;
                    if (neighbors > bufferSize-TILE_SIZE) {
                        uint tiles = neighbors/TILE_SIZE;
                        if (lane == 0)
                            *tileIndex = metalNeighborReserve(interactionCount, tiles);
                        simdgroup_barrier(mem_flags::mem_threadgroup);
                        uint firstTile = *tileIndex;
                        if (firstTile+tiles <= maxTiles) {
                            if (lane < tiles)
                                interactingTiles[firstTile+lane] = x;
                            for (int tile = 0; tile < tiles; tile++)
                                interactingAtoms[(firstTile+tile)*TILE_SIZE+lane] = buffer[lane+tile*TILE_SIZE];
                        }
                        if (lane+TILE_SIZE*tiles < bufferSize)
                            buffer[lane] = buffer[lane+TILE_SIZE*tiles];
                        neighbors -= TILE_SIZE*tiles;
                    }
                }
                else {
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
        }

        if (neighbors > 0) {
            uint tiles = (neighbors+TILE_SIZE-1)/TILE_SIZE;
            if (lane == 0)
                *tileIndex = metalNeighborReserve(interactionCount, tiles);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            uint firstTile = *tileIndex;
            if (firstTile+tiles <= maxTiles) {
                if (lane < tiles)
                    interactingTiles[firstTile+lane] = x;
                for (int tile = 0; tile < tiles; tile++)
                    interactingAtoms[(firstTile+tile)*TILE_SIZE+lane] =
                            lane+tile*TILE_SIZE < neighbors ? buffer[lane+tile*TILE_SIZE] : NUM_ATOMS;
            }
        }
    }
    for (int atom = int(_metal.globalId); atom < NUM_ATOMS; atom += globalSize)
        oldPositions[atom] = posq[atom];
}
