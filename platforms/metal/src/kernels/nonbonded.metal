#include <metal_stdlib>
using namespace metal;

constant float ONE_4PI_EPS0 = 138.935456f;

kernel void computeNoCutoffNonbonded(device const float4* positions [[buffer(0)]],
                                      device float4* forces [[buffer(1)]],
                                      device const float4* particleParameters [[buffer(2)]],
                                      device const uint* exceptionOffsets [[buffer(3)]],
                                      device const uint2* exceptionEntries [[buffer(4)]],
                                      device const float4* exceptionParameters [[buffer(5)]],
                                      device float* energyByParticle [[buffer(6)]],
                                      constant uint& numParticles [[buffer(7)]],
                                      constant uint& includeForces [[buffer(8)]],
                                      constant uint& includeEnergy [[buffer(9)]],
                                      uint particle [[thread_position_in_grid]]) {
    if (particle >= numParticles)
        return;

    const float4 parameters = particleParameters[particle];
    float3 particleForce = float3(0.0f);
    float particleEnergy = 0.0f;
    uint exceptionCursor = exceptionOffsets[particle];
    const uint exceptionEnd = exceptionOffsets[particle+1];

    for (uint other = 0; other < numParticles; other++) {
        if (other == particle)
            continue;

        while (exceptionCursor < exceptionEnd && exceptionEntries[exceptionCursor].x < other)
            exceptionCursor++;
        const bool isException = exceptionCursor < exceptionEnd &&
                                 exceptionEntries[exceptionCursor].x == other;

        float chargeProduct;
        float sigma;
        float epsilon;
        if (isException) {
            const float4 exception = exceptionParameters[exceptionEntries[exceptionCursor].y];
            chargeProduct = exception.x;
            sigma = exception.y;
            epsilon = exception.z;
        }
        else {
            const float4 otherParameters = particleParameters[other];
            chargeProduct = parameters.x*otherParameters.x;
            sigma = 0.5f*(parameters.y+otherParameters.y);
            epsilon = sqrt(parameters.z*otherParameters.z);
        }

        // A zero-valued exception is an exclusion.  Test it before distance so
        // excluded particles may safely occupy the same position.
        if (chargeProduct == 0.0f && epsilon == 0.0f)
            continue;
        const float3 delta = positions[particle].xyz-positions[other].xyz;
        const float r2 = dot(delta, delta);
        const float inverseR = rsqrt(r2);
        float pairEnergy = 0.0f;
        float forceScale = 0.0f;
        if (chargeProduct != 0.0f) {
            const float coulombEnergy = ONE_4PI_EPS0*chargeProduct*inverseR;
            pairEnergy += coulombEnergy;
            forceScale += coulombEnergy/r2;
        }
        if (epsilon != 0.0f) {
            const float sigmaOverR2 = sigma*sigma/r2;
            const float sigmaOverR6 = sigmaOverR2*sigmaOverR2*sigmaOverR2;
            pairEnergy += 4.0f*epsilon*(sigmaOverR6-1.0f)*sigmaOverR6;
            forceScale += 24.0f*epsilon*(2.0f*sigmaOverR6-1.0f)*sigmaOverR6/r2;
        }

        if (includeForces != 0)
            particleForce += forceScale*delta;
        // Each pair is evaluated by both endpoint threads for race-free force
        // accumulation, but its energy is assigned to just one endpoint.
        if (includeEnergy != 0 && particle < other)
            particleEnergy += pairEnergy;
    }

    if (includeForces != 0)
        forces[particle] += float4(particleForce, 0.0f);
    energyByParticle[particle] = includeEnergy == 0 ? 0.0f : particleEnergy;
}
