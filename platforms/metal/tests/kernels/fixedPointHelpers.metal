kernel void convertFixedPointHelpers(device const float* input [[buffer(0)]],
                                     device uint2* converted [[buffer(1)]],
                                     device uint2* split [[buffer(2)]],
                                     device float* reconstructed [[buffer(3)]],
                                     constant uint& count [[buffer(4)]],
                                     uint index [[thread_position_in_grid]]) {
    if (index >= count)
        return;
    const float value = input[index];
    const float integral = trunc(value);
    const float scaledFraction = (value-integral)*0x1.0p32f;
    const uint fractionalMagnitude = uint(fabs(scaledFraction));
    converted[index] = realToFixedPoint(value);
    split[index] = splitFixedPoint(int(integral), fractionalMagnitude,
                                   scaledFraction <= -1.0f);
    reconstructed[index] = reconstructSignedFixedPoint(converted[index]);
}

kernel void loadFixedPointHelpers(device const uint2* input [[buffer(0)]],
                                  device uint2* loaded [[buffer(1)]],
                                  device float* reconstructed [[buffer(2)]],
                                  constant uint& count [[buffer(3)]],
                                  uint index [[thread_position_in_grid]]) {
    if (index >= count)
        return;
    loaded[index] = loadFixedPoint(input, index);
    reconstructed[index] = loadSignedFixedPoint(input, index);
}

kernel void loadFixedPointPlanes(device const uint2* input [[buffer(0)]],
                                 device float4* output [[buffer(1)]],
                                 constant uint& atom [[buffer(2)]],
                                 constant uint& paddedNumAtoms [[buffer(3)]],
                                 uint index [[thread_position_in_grid]]) {
    if (index == 0u)
        output[0] = float4(loadFixedPoint3(input, atom, paddedNumAtoms), 0.0f);
}

kernel void computeFixedPointCarries(
        device const uint* previousLowWords [[buffer(0)]],
        device const uint* lowWordAddends [[buffer(1)]],
        device uint* carries [[buffer(2)]],
        constant uint& count [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
    if (index < count)
        carries[index] = computeFixedPointCarry(previousLowWords[index],
                                                lowWordAddends[index]);
}

kernel void addFixedPointWordsAtomically(
        device atomic_uint* words [[buffer(0)]],
        device uint* previousLowWords [[buffer(1)]],
        device uint* carries [[buffer(2)]],
        constant uint& logicalIndex [[buffer(3)]],
        constant uint& lowWordAddend [[buffer(4)]],
        constant uint& highWordAddend [[buffer(5)]],
        uint index [[thread_position_in_grid]]) {
    threadgroup_barrier(mem_flags::mem_none);
    const uint previousLowWord = atomicAddFixedPointLowWord(words, logicalIndex,
                                                            lowWordAddend);
    const uint carry = computeFixedPointCarry(previousLowWord, lowWordAddend);
    atomicAddFixedPointHighWord(words, logicalIndex, highWordAddend, carry);
    previousLowWords[index] = previousLowWord;
    carries[index] = carry;
}

kernel void addFixedPointAtomically(
        device atomic_uint* words [[buffer(0)]],
        constant uint& logicalIndex [[buffer(1)]],
        constant uint& lowWordAddend [[buffer(2)]],
        constant uint& highWordAddend [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
    atomicAddFixedPoint(words, logicalIndex,
                        uint2(lowWordAddend, highWordAddend));
}
