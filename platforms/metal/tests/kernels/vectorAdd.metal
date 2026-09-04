#include <metal_stdlib>
using namespace metal;

kernel void vectorAdd(device const float* a [[buffer(0)]],
                      device const float* b [[buffer(1)]],
                      device float* result [[buffer(2)]],
                      constant uint& count [[buffer(3)]],
                      uint index [[thread_position_in_grid]]) {
    if (index < count)
        result[index] = a[index]+b[index];
}

struct VectorArguments {
    device const float* a [[id(0)]];
    device const float* b [[id(1)]];
    device float* result [[id(2)]];
    uint count [[id(3)]];
};

kernel void vectorAddArgumentBuffer(constant VectorArguments& args [[buffer(0)]],
                                    uint index [[thread_position_in_grid]]) {
    if (index < args.count)
        args.result[index] = args.a[index]+args.b[index];
}
