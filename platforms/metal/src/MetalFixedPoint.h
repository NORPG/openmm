#ifndef OPENMM_METALFIXEDPOINT_H_
#define OPENMM_METALFIXEDPOINT_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2026 Stanford University and the Authors.           *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 * -------------------------------------------------------------------------- */

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace OpenMM {

/**
 * Physical storage for one logical signed 64-bit fixed-point value.
 *
 * This is a bit container, not a C++ integer.  `lo` contains bits 0-31 and
 * `hi` contains bits 32-63 of a two's-complement value.  Read-only arrays of
 * this type map to MSL `uint2` arrays (`x == lo`, `y == hi`) with an eight byte
 * stride.  The word order is part of the ABI and must not be inferred from
 * host byte endianness.
 *
 * Atomic MSL kernels must bind the same bytes as scalar `atomic_uint` words
 * and address words 2*i and 2*i+1.  They must not concurrently update the
 * components of a `uint2` object.  A `uint2` view is only valid after all
 * atomic writers have completed.
 */
struct alignas(8) MetalFixedPoint64Storage {
    uint32_t lo;
    uint32_t hi;
};

static_assert(sizeof(uint32_t) == 4, "Metal fixed-point ABI requires 32-bit limbs");
static_assert(sizeof(MetalFixedPoint64Storage) == 8,
              "Metal logical 64-bit storage must occupy exactly eight bytes");
static_assert(alignof(MetalFixedPoint64Storage) == 8,
              "Metal logical 64-bit storage must be eight-byte aligned");
static_assert(offsetof(MetalFixedPoint64Storage, lo) == 0,
              "The low fixed-point word must be stored first");
static_assert(offsetof(MetalFixedPoint64Storage, hi) == 4,
              "The high fixed-point word must immediately follow the low word");
static_assert(std::is_standard_layout<MetalFixedPoint64Storage>::value,
              "Metal fixed-point storage must have standard layout");
static_assert(std::is_trivially_copyable<MetalFixedPoint64Storage>::value,
              "Metal fixed-point storage must be trivially copyable");

} // namespace OpenMM

#endif /*OPENMM_METALFIXEDPOINT_H_*/
