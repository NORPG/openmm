#include <metal_stdlib>
#ifndef OPENMM_METAL_FIXED_POINT
#include "fixedPoint.metal"
#endif
using namespace metal;

kernel void integrateVerlet(device float4* positions [[buffer(0)]],
                            device float4* velocities [[buffer(1)]],
                            device const uint2* forces [[buffer(2)]],
                            device const float* inverseMasses [[buffer(3)]],
                            constant float& stepSize [[buffer(4)]],
                            constant uint& numParticles [[buffer(5)]],
                            constant uint& paddedNumAtoms [[buffer(6)]],
                            uint particle [[thread_position_in_grid]]) {
    if (particle >= numParticles)
        return;
    const float inverseMass = inverseMasses[particle];
    if (inverseMass == 0.0f)
        return;
    const float velocityMetadata = velocities[particle].w;
    const float positionMetadata = positions[particle].w;
    float3 velocity = velocities[particle].xyz;
    velocity += stepSize*inverseMass*loadFixedPoint3(forces, particle, paddedNumAtoms);
    const float3 position = positions[particle].xyz+stepSize*velocity;
    velocities[particle] = float4(velocity, velocityMetadata);
    positions[particle] = float4(position, positionMetadata);
}
