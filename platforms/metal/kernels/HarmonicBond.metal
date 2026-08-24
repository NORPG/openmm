#include <metal_stdlib>
using namespace metal;

kernel void computeHarmonicBonds(device const float4* positions [[buffer(0)]],
                                 device float4* forces [[buffer(1)]],
                                 device const uint2* bonds [[buffer(2)]],
                                 device const float2* parameters [[buffer(3)]],
                                 device float* energyByParticle [[buffer(4)]],
                                 constant uint& numParticles [[buffer(5)]],
                                 constant uint& numBonds [[buffer(6)]],
                                 constant uint& includeForces [[buffer(7)]],
                                 constant uint& includeEnergy [[buffer(8)]],
                                 uint particle [[thread_position_in_grid]]) {
    if (particle >= numParticles)
        return;

    float3 particleForce = float3(0.0f);
    float particleEnergy = 0.0f;
    for (uint bond = 0; bond < numBonds; bond++) {
        const uint2 atoms = bonds[bond];
        if (particle != atoms.x && particle != atoms.y)
            continue;

        const float3 delta = positions[atoms.y].xyz-positions[atoms.x].xyz;
        const float distance = length(delta);
        const float2 parameter = parameters[bond];
        const float displacement = distance-parameter.x;
        if (includeForces != 0 && distance > 0.0f) {
            const float3 forceOnFirst = (parameter.y*displacement/distance)*delta;
            if (particle == atoms.x)
                particleForce += forceOnFirst;
            if (particle == atoms.y)
                particleForce -= forceOnFirst;
        }
        // Assign each bond's energy to its first endpoint so it is counted once.
        if (includeEnergy != 0 && particle == atoms.x)
            particleEnergy += 0.5f*parameter.y*displacement*displacement;
    }
    if (includeForces != 0)
        forces[particle] += float4(particleForce, 0.0f);
    if (includeEnergy != 0)
        energyByParticle[particle] += particleEnergy;
    else
        energyByParticle[particle] = 0.0f;
}
