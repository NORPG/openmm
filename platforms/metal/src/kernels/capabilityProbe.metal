#ifndef OPENMM_METAL_FIXED_POINT
#include "fixedPoint.metal"
#endif

static_assert(sizeof(atomic_uint) == 4 && alignof(atomic_uint) == 4,
              "Split fixed-point requires four-byte atomic words");
static_assert(sizeof(uint2) == 8 && alignof(uint2) == 8,
              "Logical fixed-point storage must match the host ABI");

kernel void probeSplitFixedPointWriters(
        device atomic_uint* words [[buffer(0)]],
        device const float* contributions [[buffer(1)]],
        constant uint& numAtoms [[buffer(2)]],
        constant uint& paddedNumAtoms [[buffer(3)]],
        constant uint& numWriters [[buffer(4)]],
        uint writer [[thread_position_in_grid]]) {
    if (writer >= numWriters)
        return;
    const uint atom = writer%numAtoms;
    for (uint axis = 0; axis < 3; axis++) {
        const uint index = atom+axis*paddedNumAtoms;
        atomicAddFixedPoint(words, index, realToFixedPoint(contributions[3u*writer+axis]));
    }
}

// Separate dispatch: never reconstruct while any writer may update a limb.
kernel void probeSplitFixedPointReaders(
        device const uint2* words [[buffer(0)]],
        device uint2* rawValues [[buffer(1)]],
        device float4* reconstructed [[buffer(2)]],
        constant uint& paddedNumAtoms [[buffer(3)]],
        uint atom [[thread_position_in_grid]]) {
    if (atom >= paddedNumAtoms)
        return;
    for (uint axis = 0; axis < 3; axis++) {
        const uint index = atom+axis*paddedNumAtoms;
        rawValues[index] = loadFixedPoint(words, index);
    }
    reconstructed[atom] = float4(loadFixedPoint3(words, atom, paddedNumAtoms), 0.0f);
}
