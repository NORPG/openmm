#ifndef OPENMM_METAL_FIXED_POINT
#include "fixedPoint.metal"
#endif

// Temporary bridge for native producers that still accumulate float4 forces.
// Execute only after all producers finish, before any logical-64 consumer.
kernel void convertFloatForcesToFixedPoint(
        device const float4* floatForces [[buffer(0)]],
        device uint2* longForces [[buffer(1)]],
        device atomic_uint* conversionError [[buffer(2)]],
        constant uint& numAtoms [[buffer(3)]],
        constant uint& paddedNumAtoms [[buffer(4)]],
        uint atom [[thread_position_in_grid]]) {
    if (atom >= paddedNumAtoms)
        return;
    const float3 force = atom < numAtoms ? floatForces[atom].xyz : float3(0.0f);
    // Use representation checks so fast math cannot optimize away NaN/Inf
    // validation.  Q32.32 allows -2^31, but excludes +2^31.
    const uint3 bits = as_type<uint3>(force);
    const bool3 inRange = ((bits & uint3(0x7fffffffu)) < uint3(0x4f000000u)) |
                          (bits == uint3(0xcf000000u));
    if (!all(inRange)) {
        atomic_store_explicit(conversionError, 1u, memory_order_relaxed);
        return;
    }
    longForces[atom] = realToFixedPoint(force.x);
    longForces[atom+paddedNumAtoms] = realToFixedPoint(force.y);
    longForces[atom+2u*paddedNumAtoms] = realToFixedPoint(force.z);
}
