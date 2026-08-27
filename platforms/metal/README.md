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
