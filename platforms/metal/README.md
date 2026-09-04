# OpenMM native Metal platform (experimental)

This directory contains a native Metal backend.  Its host runtime is
Objective-C++, confined to private implementation files; installed headers and
the OpenMM integration remain ordinary C++11.  It uses Apple's public Metal API
directly and does not use Metal-cpp, OpenCL, or `cl2Metal`.

Hand-written MSL lives only in standalone `src/kernels/*.metal` files.  The
build always encodes those sources into the plugin for runtime-generated
kernels.  When Xcode's optional Metal Toolchain is available, it also compiles
them offline and embeds one metallib in both shared and static plugins.  Both
paths produce self-contained plugins and never read the source tree at runtime.
Runtime MSL compilation also remains available through `MetalProgram` for
future generated kernels.

`MetalContext` implements OpenMM's `ComputeContext` interface for the core
single-device runtime surface: queues, arrays, events, runtime MSL programs,
single-precision standard state buffers, and context bookkeeping.  This is a
minimal foundation, not yet a claim that the existing Common kernel sources
can be lowered to MSL.  Common sorting, FFT, integration, expression, bonded,
and nonbonded utility objects are rejected explicitly until their Metal
implementations are added.  Arrays also remain bound to the queue on which they
were created, so switching an existing workload to a sibling queue is not yet
supported.

## Logical 64-bit fixed-point storage ABI

The Common-compatible force accumulator uses an explicitly split logical
64-bit storage element.  Its host representation is
`MetalFixedPoint64Storage`; the corresponding read-only MSL view is `uint2`:

- each element is exactly 8 bytes and aligned to 8 bytes;
- word 0 / `x` / `lo` stores bits 0-31;
- word 1 / `y` / `hi` stores bits 32-63;
- logical signed values use two's-complement bit representation; and
- arrays have an 8-byte element stride with no inter-element padding.

For the Common force buffer, component planes retain the existing logical
indexing: `atom + axis*paddedNumAtoms`, where axes 0, 1, and 2 are x, y, and z.
The byte offset of a logical component is therefore
`8*(atom + axis*paddedNumAtoms)`.  The complete buffer contains
`3*paddedNumAtoms` logical elements, or `24*paddedNumAtoms` bytes.

`MetalContext` owns this buffer as a distinct private `MetalArray`, initializes
it to zero, exposes it through `getLongForceBuffer()`, and registers it for
automatic clearing at the start of every force evaluation.  The context's
pinned transfer storage is sized to include the complete long force buffer.
Zeroing an aligned compute buffer is encoded as a GPU blit fill on the buffer's
command queue, so long-force initialization and autoclear do not allocate or
upload a host-sized zero array.  Queue ordering makes a following kernel or
download observe the completed clear.

GPU buffer copies preserve all 8 bytes of every logical element for Common's
save/restore paths.  Metal checkpoint version 2 likewise stores the complete
padded long-force buffer.  The loader accepts legacy version 1 checkpoints and
GPU-clears the buffer because those checkpoints contain no long-force payload.

`src/kernels/fixedPoint.metal` defines the Metal 3.0 helper contract used by
runtime-generated kernels.  `MetalContext::compileProgram()` prepends this
source automatically.  `realToFixedPoint()` implements
`trunc(value*2^32)` for finite binary32 values in `[-2^31, 2^31)`, returning
the two's-complement result as `uint2(lo, hi)`.  `splitFixedPoint()` exposes the
same word assembly operation for already-decomposed values.  `loadFixedPoint()`
loads the raw words, while `reconstructSignedFixedPoint()` and
`loadSignedFixedPoint()` convert a signed Q32.32 value back to binary32.
`loadFixedPoint3()` applies the component-plane indexing described above.

Signed reconstruction performs one round-to-nearest, ties-to-even operation on
the complete two-word magnitude.  Kernels must not reconstruct a negative value
by separately converting and adding its signed high word and unsigned low word:
that loses small negative fractions and can double-round larger values.

`atomicAddFixedPointLowWord()` binds the logical buffer as scalar
`device atomic_uint` words and uses `atomic_fetch_add_explicit()` with
`memory_order_relaxed` on word `2*i`.  It returns the previous low word for a
later carry calculation.  `computeFixedPointCarry()` compares the modulo-2^32
low-word result with that previous value and returns the carry-out as `0u` or
`1u`.  `atomicAddFixedPointHighWord()` adds the high-word addend plus that carry
to word `2*i+1` with another relaxed 32-bit atomic operation, skipping the RMW
when the combined addend is zero.  Together these helpers implement modulo-2^64
addition using only 32-bit operations.  `atomicAddFixedPoint()` is the public
whole-value wrapper and deliberately returns `void`, since the two word updates
do not form a linearizable 64-bit atomic operation.

Ordinary signed 32-bit work counters use the separate
`atomicFetchAddCounter32()` API from `counterAtomics.metal`.  It performs one
relaxed atomic operation and returns the counter value from before that
operation.  Counter code must not call the fixed-point limb helpers, and
fixed-point code must not treat a counter return value as a coherent old
Q32.32 value.  The relaxed operation reserves a unique counter value but does
not publish other payload writes between threads.

The word order above is an ABI rule rather than an inference from byte
endianness.  Atomic writers must bind the buffer as scalar `atomic_uint` words,
using indices `2*i` and `2*i+1`.  They must not concurrently update components
through a `uint2` view.  Read-only `uint2` access is permitted only after all
atomic writers have completed.  Routing Common force producers into this
buffer remains separate work.

## Current support boundary

This is a deliberately small, executable vertical slice:

- Apple Silicon on macOS 13 or newer
- one built-in GPU (`DeviceIndex=0`)
- single precision
- `HarmonicBondForce` without periodic boundary conditions
- `NonbondedForce` with `NoCutoff`, including Coulomb, Lennard-Jones,
  exceptions, and particle/exception parameter updates
- `VerletIntegrator`
- `LocalEnergyMinimizer` (CPU optimization control with Metal force evaluations)
- systems without constraints or virtual sites

The force kernels avoid device atomics by assigning one GPU thread to each
particle.  The initial nonbonded path evaluates all particle pairs and uses a
sorted CSR table for exceptions.  These kernels are correct but intentionally
not the performance design for the full backend.

Cutoff, Ewald, PME, LJPME, and nonbonded parameter offsets are not yet
implemented.  All other forces, integrators, precisions, devices, constraints,
virtual sites, and multi-GPU execution are outside this phase and are rejected
explicitly.
The platform has a lower automatic-selection speed than the Reference platform,
so callers must select `Metal` explicitly during this experimental phase.

## Build and validate

Configure OpenMM with `OPENMM_BUILD_METAL_LIB=ON`.  Production-kernel
compilation is controlled by `OPENMM_METAL_KERNEL_COMPILATION`:

- `AUTO` (default): embed an offline metallib when `metal` and `metallib` are
  available, otherwise embed MSL source for runtime compilation
- `ON`: require the full Xcode Metal Toolchain and offline compilation
- `OFF`: always use the runtime-compilation compatibility path

The focused test targets are:

- `TestMetalComputeContext`: the minimal `ComputeContext` contract, standard
  state-buffer ABI, `ComputeArray` interoperability, events, and runtime MSL
  compilation through the generic compute interfaces
- `TestMetalPlatform`: plugin registration, device properties, and the required
  OpenMM kernel-factory surface
- `TestMetalRuntime`: buffers, transfers, queues, events, runtime MSL
  compilation, direct argument binding, reflected Tier 2 argument buffers,
  resize after kernel binding, and C++ wrapper lifetime independence
- `TestMetalVerticalSlice`: analytic force/energy values, Verlet state changes,
  Reference trajectory comparison, parameter updates, checkpoint replay,
  minimization, shared-particle bond accumulation, and rejection of unsupported
  features
- `TestMetalNonbondedForce`: analytic Coulomb/Lennard-Jones values, exceptions,
  parameter updates, force groups, include flags, multi-threadgroup execution,
  Reference trajectory comparison, and rejection of unsupported methods

Passing these tests proves the native runtime and the documented vertical slice.
It does not claim coverage of OpenMM's full kernel corpus or Intel/AMD Metal
devices.
