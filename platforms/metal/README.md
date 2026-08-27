# OpenMM native Metal platform (experimental)

This directory contains a native Metal backend.  The host runtime is written
in Swift, while OpenMM's existing implementation remains C++ and calls it
through the private, C-compatible ABI in `src/MetalSwiftBridge.h`.  No
Objective-C++ source or compiler mode is used.

Production Metal Shading Language lives only in `kernels/*.metal`.  The build
uses Xcode's `metal` and `metallib` tools to compile those files offline, then
embeds the resulting metallib in both the shared and static libraries.  The
installed plugin therefore has no external kernel-resource dependency.  The
runtime can still compile MSL source for future generated kernels; that path is
tested with the standalone source in `tests/kernels/RuntimeTestKernels.metal`.

This backend does not use OpenCL or `cl2Metal`.  Removing Objective-C++ does not
mean removing Apple's Objective-C runtime: Swift's public Metal framework
overlay interoperates with the system framework through Apple's supported
runtime.  No Objective-C or Objective-C++ type crosses the C ABI.

## Phase 1 support boundary

Phase 1 is a deliberately small, executable vertical slice:

- Apple Silicon on macOS 13 or newer
- one built-in GPU (`DeviceIndex=0`)
- single precision
- `HarmonicBondForce` without periodic boundary conditions
- `NonbondedForce` with `NoCutoff` Coulomb and Lennard-Jones interactions,
  including exceptions and in-context parameter updates
- `VerletIntegrator`
- `LocalEnergyMinimizer` (CPU optimization control with Metal force evaluations)
- systems without constraints or virtual sites

The harmonic-bond kernel supports any bond topology, including multiple bonds
that share a particle.  It avoids device atomics by assigning one GPU thread to
each particle and scanning the bond list.  This is correct but intentionally not
the performance design for the full backend.

The nonbonded path is owned by a Swift model that validates exception topology,
packs typed parameter buffers, and manages execution and updates.  Cutoff and
periodic methods, Ewald/PME/LJPME, and parameter offsets are not yet supported.

All other forces, integrators, precisions, devices, constraints, virtual sites,
and multi-GPU execution are outside this phase and are rejected explicitly.
The platform has a lower automatic-selection speed than the Reference platform,
so callers must select `Metal` explicitly during this experimental phase.

## Build and validate

The build requires CMake 3.29 or newer, Swift 6.3 or newer, the macOS SDK, and
Xcode's optional Metal Toolchain component.  Configure OpenMM with
`OPENMM_BUILD_METAL_LIB=ON`.  The focused test targets are:

- `TestMetalPlatform`: plugin registration, device properties, and the required
  OpenMM kernel-factory surface
- `TestMetalRuntime`: buffers, transfers, queues, events, runtime MSL
  compilation, direct argument binding, Tier 2 argument buffers, resize after
  kernel binding, and C++ wrapper lifetime independence
- `TestMetalVerticalSlice`: analytic force/energy values, Verlet state changes,
  Reference trajectory comparison, parameter updates, checkpoint replay,
  minimization, shared-particle bond accumulation, and rejection of unsupported
  features
- `TestMetalNonbondedForce`: analytic Coulomb/Lennard-Jones values, exceptions,
  parameter updates, force groups, execution flags, threadgroup boundaries, and
  a trajectory comparison against the Reference platform

Passing these tests proves the native runtime and the documented vertical slice.
It does not claim coverage of OpenMM's full kernel corpus or Intel/AMD Metal
devices.
