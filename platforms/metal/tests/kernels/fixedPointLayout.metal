#include <metal_stdlib>
using namespace metal;

/*
 * The host MetalFixedPoint64Storage ABI maps each logical value to one uint2:
 * x is the low 32-bit word, y is the high 32-bit word, and the array stride is
 * eight bytes.  This kernel is deliberately read-only with respect to the
 * uint2 input.  Future atomic writers must bind the storage as scalar
 * atomic_uint words instead of modifying vector components concurrently.
 */
kernel void inspectFixedPointLayout(device const uint2* values [[buffer(0)]],
                                    device uint* words [[buffer(1)]],
                                    constant uint& count [[buffer(2)]],
                                    uint index [[thread_position_in_grid]]) {
    if (index >= count)
        return;
    const uint2 value = values[index];
    words[2*index] = value.x;
    words[2*index+1] = value.y;
}
