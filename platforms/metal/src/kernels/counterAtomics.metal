#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

/**
 * Atomically add to an ordinary signed 32-bit counter and return its previous
 * value.  This API is for work allocation and indexing counters, not for either
 * limb of a logical Q32.32 value.
 */
inline int atomicFetchAddCounter32(device atomic_int* counter, int addend) {
    return atomic_fetch_add_explicit(counter, addend, memory_order_relaxed);
}
