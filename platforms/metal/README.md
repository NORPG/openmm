# OpenMM native Metal platform (experimental)

This directory contains a native Metal backend.  It compiles Metal Shading
Language directly through the public Metal API and does not use OpenCL or
`cl2Metal`.

Hand-written MSL lives only in standalone `src/kernels/*.metal` files.  During
the build, OpenMM's kernel-source encoder places those sources in a private
generated C++ container so the installed shared or static plugin remains
self-contained.  The implementation never reads kernel files from the source
tree at runtime.

## Phase 1 support boundary

Phase 1 is a deliberately small, executable vertical slice:

- Apple Silicon on macOS 13 or newer
- one built-in GPU (`DeviceIndex=0`)
- single precision
- `HarmonicBondForce` without periodic boundary conditions
- `VerletIntegrator`
- `LocalEnergyMinimizer` (CPU optimization control with Metal force evaluations)
- systems without constraints or virtual sites

The harmonic-bond kernel supports any bond topology, including multiple bonds
that share a particle.  It avoids device atomics by assigning one GPU thread to
each particle and scanning the bond list.  This is correct but intentionally not
the performance design for the full backend.

All other forces, integrators, precisions, devices, constraints, virtual sites,
and multi-GPU execution are outside this phase and are rejected explicitly.
The platform has a lower automatic-selection speed than the Reference platform,
so callers must select `Metal` explicitly during this experimental phase.

## Build and validate

Configure OpenMM with `OPENMM_BUILD_METAL_LIB=ON`.  The focused test targets are:

- `TestMetalPlatform`: plugin registration, device properties, and the required
  OpenMM kernel-factory surface
- `TestMetalRuntime`: buffers, transfers, queues, events, runtime MSL compilation,
  direct argument binding, and Tier 2 argument buffers
- `TestMetalVerticalSlice`: analytic force/energy values, Verlet state changes,
  Reference trajectory comparison, parameter updates, checkpoint replay,
  minimization, shared-particle bond accumulation, and rejection of unsupported
  features

Passing these tests proves the native runtime and the documented vertical slice.
It does not claim coverage of OpenMM's full kernel corpus or Intel/AMD Metal
devices.
