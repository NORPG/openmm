/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * Native Metal bucket and bitonic sorting kernels.                           *
 * Portions copyright (c) 2008-2026 Stanford University and the Authors.       *
 * Portions copyright (c) 2026 Chun-Chi Hung.                                  *
 * Authors: Peter Eastman, Chun-Chi Hung                                      *
 * This program is free software under the GNU Lesser General Public License, *
 * version 3 or (at your option) any later version.                            *
 * -------------------------------------------------------------------------- */

// OPENMM_METAL_NATIVE_BODY

// DATA_TYPE, KEY_TYPE, SORT_KEY, MIN_KEY, MAX_KEY, MAX_VALUE and UNIFORM are
// supplied by the existing sort trait. Helpers receive their context explicitly.
inline KEY_TYPE metalSortKey(MetalExecutionContext _metal, DATA_TYPE value) {
    return SORT_KEY;
}

inline bool metalSortExchange(MetalExecutionContext _metal, DATA_TYPE first, DATA_TYPE second, bool ascending) {
    KEY_TYPE low = metalSortKey(_metal, ascending ? first : second);
    KEY_TYPE high = metalSortKey(_metal, ascending ? second : first);
    return low > high;
}

inline bool metalSortDirection(MetalExecutionContext _metal, uint index, uint width, uint length) {
    bool ascending = (index&width) == 0;
    for (uint mask = 2*width; mask < 2*length; mask *= 2)
        if ((index&mask) == 0)
            ascending = !ascending;
    return ascending;
}

#if OPENMM_METAL_FAST_SORT_BUCKET_SCAN && __METAL_VERSION__ >= 300
// Inclusive uint scan, using the existing two-level 32-lane SIMD algorithm.
inline uint metalSortInclusiveScan(MetalExecutionContext _metal, uint value, threadgroup uint* scratch) {
    const uint lane = _metal.localId%32;
    const uint simdGroup = _metal.localId/32;
    const uint groupCount = (_metal.localSize+31)/32;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (ushort offset = 1; offset < 32; offset *= 2) {
        uint previous = simd_shuffle_up(value, offset);
        if (lane >= offset)
            value += previous;
    }
    if (lane == 31 || _metal.localId == _metal.localSize-1)
        scratch[simdGroup] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint total = _metal.localId < groupCount ? scratch[_metal.localId] : 0u;
    if (simdGroup == 0) {
        for (ushort offset = 1; offset < 32; offset *= 2) {
            uint previous = simd_shuffle_up(total, offset);
            if (lane >= offset)
                total += previous;
        }
        if (lane < groupCount)
            scratch[lane] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint result = value+(simdGroup == 0 ? 0u : scratch[simdGroup-1]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}
#endif

KERNEL void sortShortList(device DATA_TYPE* data, uint length, threadgroup DATA_TYPE* dataBuffer) {
    const uint lane = _metal.localId;
#if OPENMM_METAL_FAST_SORT_REGISTER_BITONIC && __METAL_VERSION__ >= 300
    if (length <= 32 && _metal.localSize == 32) {
        DATA_TYPE value = lane < length ? data[lane] : static_cast<DATA_TYPE>(MAX_VALUE);
        uint valid = lane < length ? 1u : 0u;
        uint originalIndex = lane;
        for (uint width = 2; width <= 32; width *= 2) {
            for (ushort distance = ushort(width/2); distance > 0; distance /= 2) {
                DATA_TYPE other = simd_shuffle_xor(value, distance);
                uint otherValid = simd_shuffle_xor(valid, distance);
                uint otherIndex = simd_shuffle_xor(originalIndex, distance);
                KEY_TYPE key = metalSortKey(_metal, value);
                KEY_TYPE otherKey = metalSortKey(_metal, other);
                // NaNs follow finite keys, ties retain the original record order,
                // and invalid lanes follow all real records. This strict order
                // prevents both partner lanes from keeping the same payload.
                bool keyNaN = key != key;
                bool otherNaN = otherKey != otherKey;
                bool keyBefore = (!keyNaN && otherNaN) || (!keyNaN && !otherNaN && key < otherKey);
                bool tied = (keyNaN && otherNaN) || key == otherKey;
                bool before = valid > otherValid || (valid == otherValid &&
                        (keyBefore || (tied && originalIndex < otherIndex)));
                bool keepLower = ((lane&width) == 0) == ((lane&distance) == 0);
                if (keepLower != before) {
                    value = other;
                    valid = otherValid;
                    originalIndex = otherIndex;
                }
            }
        }
        if (lane < length)
            data[lane] = value;
        return;
    }
#endif
    for (uint index = lane; index < length; index += _metal.localSize)
        dataBuffer[index] = data[index];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = 2; width < 2*length; width *= 2) {
        for (uint distance = width/2; distance > 0; distance /= 2) {
            for (uint index = lane; index < length; index += _metal.localSize) {
                uint partner = index^distance;
                if (partner > index && partner < length) {
                    DATA_TYPE first = dataBuffer[index];
                    DATA_TYPE second = dataBuffer[partner];
                    if (metalSortExchange(_metal, first, second, metalSortDirection(_metal, index, width, length))) {
                        dataBuffer[index] = second;
                        dataBuffer[partner] = first;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (uint index = lane; index < length; index += _metal.localSize)
        data[index] = dataBuffer[index];
}

// The independent SHORT_LIST_SORT host switch selects this existing full-scan
// algorithm with 64-thread groups; no implicit selection occurs in the shader.
KERNEL void sortShortList2(device const DATA_TYPE* dataIn, device DATA_TYPE* dataOut, int length) {
    threadgroup DATA_TYPE tile[64];
    const uint lane = _metal.localId;
    const uint index = _metal.globalId;
    DATA_TYPE value = dataIn[index < uint(length) ? index : 0u];
    KEY_TYPE key = metalSortKey(_metal, value);
    uint destination = 0;
    for (uint start = 0; start < uint(length); start += _metal.localSize) {
        uint count = min(_metal.localSize, uint(length)-start);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < count)
            tile[lane] = dataIn[start+lane];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint other = 0; other < count; ++other) {
            KEY_TYPE otherKey = metalSortKey(_metal, tile[other]);
            if (otherKey < key || (otherKey == key && start+other < index))
                ++destination;
        }
    }
    if (index < uint(length))
        dataOut[destination] = value;
}

KERNEL void computeRange(device const DATA_TYPE* data, uint length, device KEY_TYPE* range,
        threadgroup KEY_TYPE* minBuffer, threadgroup KEY_TYPE* maxBuffer,
        uint numBuckets, device uint* bucketOffset) {
    const uint lane = _metal.localId;
#if UNIFORM
    KEY_TYPE lowest = MAX_KEY;
    KEY_TYPE highest = MIN_KEY;
    for (uint index = lane; index < length; index += _metal.localSize) {
        KEY_TYPE value = metalSortKey(_metal, data[index]);
        lowest = min(lowest, value);
        highest = max(highest, value);
    }
    minBuffer[lane] = lowest;
    maxBuffer[lane] = highest;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 1; step < _metal.localSize; step *= 2) {
        if (lane+step < _metal.localSize && lane%(2*step) == 0) {
            minBuffer[lane] = min(minBuffer[lane], minBuffer[lane+step]);
            maxBuffer[lane] = max(maxBuffer[lane], maxBuffer[lane+step]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        range[0] = minBuffer[0];
        range[1] = maxBuffer[0];
    }
#endif
    for (uint bucket = lane; bucket < numBuckets; bucket += _metal.localSize)
        bucketOffset[bucket] = 0;
}

KERNEL void assignElementsToBuckets(device const DATA_TYPE* data, uint length, uint numBuckets,
        device const KEY_TYPE* range, device atomic_uint* bucketOffset,
        device uint* bucketOfElement, device uint* offsetInBucket) {
    const float lower = float(range[0]);
    const float upper = float(range[1]);
    const float width = (upper-lower)/numBuckets;
    const uint gridSize = _metal.localSize*_metal.numGroups;
    for (uint index = _metal.globalId; index < length; index += gridSize) {
        float key = float(metalSortKey(_metal, data[index]));
        uint bucket = min(uint((key-lower)/width), numBuckets-1);
        offsetInBucket[index] = atomic_fetch_add_explicit(bucketOffset+bucket, 1u, memory_order_relaxed);
        bucketOfElement[index] = bucket;
    }
}

KERNEL void assignElementsToBuckets2(device const DATA_TYPE* data, uint length, uint numBuckets,
        device const KEY_TYPE* range, device atomic_uint* bucketOffset,
        device uint* bucketOfElement, device uint* offsetInBucket) {
    const uint lane = _metal.localId;
    threadgroup KEY_TYPE sample[64];
    threadgroup float lowerBound[9];
    threadgroup float baseIndex[9];
    threadgroup float indexScale[9];
    if (lane < 64) {
        uint index = uint((lane*length)/64.0f);
        sample[lane] = metalSortKey(_metal, data[index]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = 2; width <= 64; width *= 2) {
        for (uint distance = width/2; distance > 0; distance /= 2) {
            if (lane < 64) {
                uint partner = lane^distance;
                if (partner > lane) {
                    KEY_TYPE first = sample[lane];
                    KEY_TYPE second = sample[partner];
                    KEY_TYPE low = (lane&width) == 0 ? first : second;
                    KEY_TYPE high = (lane&width) == 0 ? second : first;
                    if (low > high) {
                        sample[lane] = second;
                        sample[partner] = first;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (lane == 0) {
        lowerBound[0] = sample[0]-0.2f*(sample[5]-sample[0]);
        lowerBound[1] = sample[5];
        lowerBound[2] = sample[10];
        lowerBound[3] = sample[20];
        lowerBound[4] = sample[30];
        lowerBound[5] = sample[40];
        lowerBound[6] = sample[50];
        lowerBound[7] = sample[60];
        lowerBound[8] = sample[63]+0.2f*(sample[63]-sample[58]);
        for (uint segment = 0; segment < 8; ++segment)
            baseIndex[segment] = (2*segment+1)*numBuckets/16;
        baseIndex[8] = numBuckets;
        for (uint segment = 0; segment < 8; ++segment)
            indexScale[segment] = lowerBound[segment+1] == lowerBound[segment] ? 0.0f :
                    (baseIndex[segment+1]-baseIndex[segment])/(lowerBound[segment+1]-lowerBound[segment]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint gridSize = _metal.localSize*_metal.numGroups;
    for (uint index = _metal.globalId; index < length; index += gridSize) {
        float key = float(metalSortKey(_metal, data[index]));
        uint segment = 0;
        while (segment < 7 && key > lowerBound[segment+1])
            ++segment;
        uint bucket = uint(baseIndex[segment]+(key-lowerBound[segment])*indexScale[segment]);
        bucket = min(bucket, numBuckets-1);
        offsetInBucket[index] = atomic_fetch_add_explicit(bucketOffset+bucket, 1u, memory_order_relaxed);
        bucketOfElement[index] = bucket;
    }
}

KERNEL void computeBucketPositions(uint numBuckets, device uint* bucketOffset, threadgroup uint* buffer) {
    const uint lane = _metal.localId;
    uint preceding = 0;
    for (uint start = 0; start < numBuckets; start += _metal.localSize) {
        uint bucket = start+lane;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buffer[lane] = bucket < numBuckets ? bucketOffset[bucket] : 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
#if OPENMM_METAL_FAST_SORT_BUCKET_SCAN && __METAL_VERSION__ >= 300
        uint scanned = metalSortInclusiveScan(_metal, buffer[lane], buffer);
        buffer[lane] = scanned;
        threadgroup_barrier(mem_flags::mem_threadgroup);
#else
        for (uint step = 1; step < _metal.localSize; step *= 2) {
            uint add = lane >= step ? buffer[lane-step] : 0u;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            buffer[lane] += add;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
#endif
        if (bucket < numBuckets)
            bucketOffset[bucket] = preceding+buffer[lane];
        preceding += buffer[_metal.localSize-1];
    }
}

KERNEL void copyDataToBuckets(device const DATA_TYPE* data, device DATA_TYPE* buckets, uint length,
        device const uint* bucketOffset, device const uint* bucketOfElement, device const uint* offsetInBucket) {
    const uint gridSize = _metal.localSize*_metal.numGroups;
    for (uint index = _metal.globalId; index < length; index += gridSize) {
        uint bucket = bucketOfElement[index];
        uint start = bucket == 0 ? 0u : bucketOffset[bucket-1];
        buckets[start+offsetInBucket[index]] = data[index];
    }
}

KERNEL void sortBuckets(device DATA_TYPE* data, device const DATA_TYPE* buckets, uint numBuckets,
        device const uint* bucketOffset, threadgroup DATA_TYPE* buffer) {
    const uint lane = _metal.localId;
    for (uint bucket = _metal.groupId; bucket < numBuckets; bucket += _metal.numGroups) {
        uint start = bucket == 0 ? 0u : bucketOffset[bucket-1];
        uint length = bucketOffset[bucket]-start;
        if (length <= _metal.localSize) {
            buffer[lane] = lane < length ? buckets[start+lane] : static_cast<DATA_TYPE>(MAX_VALUE);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint width = 2; width <= _metal.localSize; width *= 2) {
                for (uint distance = width/2; distance > 0; distance /= 2) {
                    uint partner = lane^distance;
                    if (partner > lane) {
                        DATA_TYPE first = buffer[lane];
                        DATA_TYPE second = buffer[partner];
                        if (metalSortExchange(_metal, first, second, (lane&width) == 0)) {
                            buffer[lane] = second;
                            buffer[partner] = first;
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
            if (lane < length)
                data[start+lane] = buffer[lane];
        }
        else {
            for (uint index = lane; index < length; index += _metal.localSize)
                data[start+index] = buckets[start+index];
            threadgroup_barrier(mem_flags::mem_device);
            for (uint width = 2; width < 2*length; width *= 2) {
                for (uint distance = width/2; distance > 0; distance /= 2) {
                    for (uint index = lane; index < length; index += _metal.localSize) {
                        uint partner = index^distance;
                        if (partner > index && partner < length) {
                            DATA_TYPE first = data[start+index];
                            DATA_TYPE second = data[start+partner];
                            if (metalSortExchange(_metal, first, second,
                                    metalSortDirection(_metal, index, width, length))) {
                                data[start+index] = second;
                                data[start+partner] = first;
                            }
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_device);
                }
            }
        }
    }
}
