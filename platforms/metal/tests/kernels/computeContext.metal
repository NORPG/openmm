#include <metal_stdlib>
using namespace metal;
kernel void addValue(device float* values [[buffer(0)]],
                     constant uint& count [[buffer(1)]],
                     uint index [[thread_position_in_grid]]) {
    if (index < count)
        values[index] += TEST_INCREMENT;
}
