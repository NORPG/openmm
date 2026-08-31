#include <metal_stdlib>
using namespace metal;

/*
 * Logical signed Q32.32 values use uint2(lo, hi).  These helpers deliberately
 * use only 32-bit integer operations so they remain valid on Metal devices
 * without 64-bit integer atomics.  Conversion is defined for finite float
 * values in [-2^31, 2^31), matching trunc(x*2^32) in the Common backends.
 */

/** Assemble Q32.32 words from a truncation decomposition. */
inline uint2 splitFixedPoint(int integralPart, uint fractionalMagnitude,
                             bool negativeFraction) {
    const uint integralBits = as_type<uint>(integralPart);
    return negativeFraction
            ? uint2(0u-fractionalMagnitude, integralBits-1u)
            : uint2(fractionalMagnitude, integralBits);
}

/** Convert a finite in-range float to its logical signed Q32.32 words. */
inline uint2 realToFixedPoint(float value) {
    const float integralAsFloat = trunc(value);
    const float scaledFraction = (value-integralAsFloat)*0x1.0p32f;
    const int integralPart = static_cast<int>(integralAsFloat);
    const uint fractionalMagnitude = static_cast<uint>(fabs(scaledFraction));

    // The -1 threshold preserves truncation toward zero when a negative
    // remainder has magnitude below one raw Q32.32 unit.
    return splitFixedPoint(integralPart, fractionalMagnitude,
                           scaledFraction <= -1.0f);
}

/** Return the modulo-2^64 two's-complement negation of a word pair. */
inline uint2 negateFixedPoint(uint2 value) {
    const uint low = 0u-value.x;
    return uint2(low, ~value.y+uint(value.x == 0u));
}

/** Load one raw logical Q32.32 value from the host-compatible uint2 ABI. */
inline uint2 loadFixedPoint(device const uint2* values, uint index) {
    return values[index];
}

/**
 * Round an unsigned Q32.32 magnitude to binary32 without double rounding.
 * magnitude must not exceed 2^63.
 */
inline float unsignedFixedPointToReal(uint2 magnitude) {
    if ((magnitude.x | magnitude.y) == 0u)
        return 0.0f;

    uint topBit = magnitude.y == 0u
            ? 31u-clz(magnitude.x)
            : 63u-clz(magnitude.y);
    uint significand;

    if (topBit <= 23u) {
        significand = magnitude.x << (23u-topBit);
    }
    else {
        const uint shift = topBit-23u;
        uint guard;
        bool sticky;

        if (shift < 32u) {
            significand = (magnitude.x >> shift) |
                          (magnitude.y << (32u-shift));
            guard = (magnitude.x >> (shift-1u)) & 1u;
            const uint lowerMask = shift == 1u
                    ? 0u
                    : ((1u << (shift-1u))-1u);
            sticky = (magnitude.x & lowerMask) != 0u;
        }
        else if (shift == 32u) {
            significand = magnitude.y;
            guard = magnitude.x >> 31u;
            sticky = (magnitude.x & 0x7fffffffu) != 0u;
        }
        else {
            const uint highShift = shift-32u;
            significand = magnitude.y >> highShift;
            guard = (magnitude.y >> (highShift-1u)) & 1u;
            const uint lowerMask = highShift == 1u
                    ? 0u
                    : ((1u << (highShift-1u))-1u);
            sticky = magnitude.x != 0u ||
                     (magnitude.y & lowerMask) != 0u;
        }

        if (guard != 0u && (sticky || (significand & 1u) != 0u)) {
            significand++;
            if (significand == 0x01000000u) {
                significand = 0x00800000u;
                topBit++;
            }
        }
    }

    // Scaling an integer by 2^-32 changes the unbiased exponent from topBit
    // to topBit-32.  Adding the binary32 bias therefore gives topBit+95.
    const uint exponent = topBit+95u;
    const uint bits = (exponent << 23u) |
                      (significand & 0x007fffffu);
    return as_type<float>(bits);
}

/** Interpret uint2(lo, hi) as signed two's-complement Q32.32. */
inline float fixedPointToReal(uint2 value) {
    const bool negative = (value.y & 0x80000000u) != 0u;
    const uint2 magnitude = negative ? negateFixedPoint(value) : value;
    const float converted = unsignedFixedPointToReal(magnitude);
    return negative ? -converted : converted;
}

/** Explicitly named signed reconstruction entry point. */
inline float reconstructSignedFixedPoint(uint2 value) {
    return fixedPointToReal(value);
}

/** Load and reconstruct one signed Q32.32 value. */
inline float loadSignedFixedPoint(device const uint2* values, uint index) {
    return reconstructSignedFixedPoint(loadFixedPoint(values, index));
}

/** Load three component planes using atom+axis*paddedNumAtoms indexing. */
inline float3 loadFixedPoint3(device const uint2* values, uint atom,
                              uint paddedNumAtoms) {
    return float3(loadSignedFixedPoint(values, atom),
                  loadSignedFixedPoint(values, atom+paddedNumAtoms),
                  loadSignedFixedPoint(values, atom+2u*paddedNumAtoms));
}
