# OpenMM native Metal platform (experimental)

This directory contains a native Metal backend.  Its host runtime is
Objective-C++, confined to private implementation files; installed headers and
the OpenMM integration remain ordinary C++11.  It uses Apple's public Metal API
directly and does not use Metal-cpp, OpenCL, or `cl2Metal`.

Hand-written MSL lives only in standalone `src/kernels/*.metal` files.  When
Xcode's optional Metal Toolchain is available, the build compiles those files
offline and embeds one metallib in both shared and static plugins.  Otherwise,
OpenMM's kernel-source encoder embeds the MSL for compilation through the Metal
API at runtime.  Both paths produce self-contained plugins and never read the
source tree at runtime.  Runtime MSL compilation also remains available through
`MetalProgram` for future generated kernels.

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

The future Common force accumulator uses an explicitly split logical 64-bit
storage element.  Its host representation is `MetalFixedPoint64Storage`; the
corresponding read-only MSL view is `uint2`:

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

The word order above is an ABI rule rather than an inference from byte
endianness.  Atomic writers must bind the buffer as scalar `atomic_uint` words,
using indices `2*i` and `2*i+1`.  They must not concurrently update components
through a `uint2` view.  Read-only `uint2` access is permitted only after all
atomic writers have completed.  This section defines storage and layout only;
the two-word atomic-add algorithm and force-buffer routing are separate work.

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
