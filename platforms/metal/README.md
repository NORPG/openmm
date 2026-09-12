# Experimental Metal 4 Platform

- Implements `ComputeContext`, `ArrayInterface`, `ComputeQueue`, `ComputeEvent`,
  `ComputeProgram`, and `ComputeKernel` for one Apple-silicon GPU.
- Follows the CUDA/HIP class responsibilities, array operations, argument
  rebinding, and block-based launch flow.
- Reuses the existing Common context, array wrapper, and force-info sources.
- Uses the Metal 4 host API through private Objective-C++ and ordinary C++11 headers.
- Compiles native MSL 4.0 through `MTL4Compiler`; no offline library or custom
  capability probe is required.

This variant uses the Metal 3 implementation at
[`8f6a7332f`](https://github.com/NORPG/openmm/commit/8f6a7332f4bca326cd43366f2916da396db661ae)
on `Objective-C` as its reference. That branch remains the Metal 3 host API / MSL
3.0 baseline. This branch uses Metal 4 host APIs **and** the MSL 4.0 target;
changing the shader language version alone is not a host API migration. There
is no automatic fallback to the older host API.

The Metal Platform currently contains only the Common runtime foundation and
is not yet registered as an OpenMM simulation Platform. It does not yet execute
existing Common kernel sources or implement force,
integrator, sorting, neighbor-list, or FFT/PME paths. Unsupported interfaces
throw explicitly. There is no separate Metal force algorithm, fixed-point
emulation, or shader translator.

## Porting boundaries

`MetalArray`, `MetalKernel`, `MetalProgram`, `MetalQueue`, and `MetalEvent`
follow their `Cuda*`/`Hip*` counterparts. `MetalContext` includes the runtime and
single-precision state-buffer portion, without the force-dependent platform
initialization. The force buffer uses ordinary 8-byte signed integer elements
and the existing three-component-plane layout; allocating storage does not
enable 64-bit atomic accumulation.

The Metal 4 adaptations are deliberately limited to runtime requirements:

- `MTL4CommandQueue`, command buffers, and compute encoders handle dispatch,
  buffer copies, and clearing. Explicit queue barriers preserve serial ordering.
- Each submission owns its allocator, residency set, and referenced resources
  until commit feedback reports completion. Allocators are not reset in flight.
- `MTL4ArgumentTable` binds GPU addresses. Primitive arguments use an immutable
  per-launch buffer, preserving the existing 32-byte argument slots.
- Queue-level event signals/waits use a tracked one-byte GPU fill marker so
  `flushQueue()` can observe a wait even without a subsequent kernel. Feedback
  supplies both host completion and GPU execution errors.
- Library compilation waits for the `MTL4Compiler` completion handler, keeping
  the Common interface synchronous. This avoids a dangling `NSError` reproduced
  with the synchronous compiler API under Metal API Validation on macOS 26.6.2.

These per-submission allocations are a correctness-first baseline, not a claim
of improved performance or a tuned Metal 4 submission strategy.

Nonblocking uploads
and downloads use the context's `getPinnedBuffer()` (or a range within it), like
CUDA's page-locked-memory requirement. Do not read or reuse that memory until
the relevant queue/event completes. Blocking transfers accept ordinary host
memory. Operations select the current queue at execution time, not at array
creation. Submission to each queue must be serialized by the caller.

Native kernel buffer arguments use sequential `[[buffer(i)]]` slots in Common
argument order; scalars use constant-buffer references. Stage builtins are
explicit MSL parameters. The caller supplies the matching signature and legal,
non-overlapping buffer bindings. Launches use complete threadgroups (64 threads
by default), capped at a conservative 128 groups, so kernels must support the
existing grid-stride convention. This cap is not performance tuning.

## Build and test

Use Xcode 26 or newer with a macOS 26+ SDK and a Metal 4 capable Apple silicon GPU.
Configure an arm64 build on macOS 26 or newer with
`OPENMM_BUILD_METAL_LIB=ON` and `BUILD_TESTING=ON`. The Metal option is off by
default; static builds and installation as a simulation plugin are not part
of this initial target.

```sh
cmake -S . -B build/metal4-common \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=26.0 \
  -DOPENMM_BUILD_METAL_LIB=ON \
  -DOPENMM_BUILD_SHARED_LIB=ON \
  -DOPENMM_BUILD_STATIC_LIB=OFF \
  -DBUILD_TESTING=ON
cmake --build build/metal4-common --target TestMetalComputeContext
env MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 \
  ctest --test-dir build/metal4-common -R '^TestMetalComputeContext$' --output-on-failure
```

The test exercises the Common C++ interfaces using standalone, test-only MSL
smoke kernels. It covers transfers, resize/rebinding, full and partial logical
blocks, grid-stride execution, clearing, queue/event ordering, pinned-memory
readback, and invalid inputs. Additional tests verify the compiled MSL version
is 400, in-flight scalar/buffer ownership, resize before completion, and event
rerecording on another queue. No GPU returns skip code 77, not a successful GPU
validation. Passing this test does not establish simulation or Common shader
compatibility.

Use separate source checkouts and build directories when rerunning both versions:
the test executable loads its shader from the source tree at runtime, and the
Metal 4 shader deliberately rejects any language target other than 4.0.
