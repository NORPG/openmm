# OpenMM native Metal platform (experimental)

This directory contains a native Metal backend.  Its host runtime is
Objective-C++, confined to private implementation files; installed headers and
the OpenMM integration remain ordinary C++11.  It uses Apple's public Metal API
directly and does not use Metal-cpp, OpenCL, or `cl2Metal`.

Production hand-written MSL lives in standalone `src/kernels/*.metal` files.  The
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
save/restore paths.  Metal checkpoint version 3 stores the complete padded
long-force buffer as the authoritative force state.  The loader accepts legacy
versions 1 and 2, rebuilding long forces from their float4 payload on the CPU;
version 2's long payload was scratch and is consumed but discarded.

Verlet reads the logical buffer through `loadFixedPoint3()` and reconstructs
binary32 forces on the GPU.  Force download reconstructs directly to double on
the CPU, retaining fractional bits that an intermediate float would lose;
shifted velocities and kinetic energy use this same download path.  Energy-only
evaluations preserve the force buffers while clearing other autoclear arrays.

The native harmonic-bond and nonbonded producers currently accumulate float4
forces.  At the end of a force evaluation, `forceBuffers.metal` converts the
completed sum into the logical buffer, including zeroed padding.  This interim
bridge follows the same offline/runtime compilation policy as the other
production kernels and adds a GPU dispatch and a small validation readback per
force evaluation.
Inputs must be finite and within `[-2^31, 2^31)`; an unrepresentable value raises
an error before a consumer uses it.  The bridge overwrites the logical buffer,
so producers must be migrated together before enabling direct long-buffer
accumulation.  The Common float-buffer accessors still expose the native
producer accumulator; consumers explicitly bind `getLongForceBuffer()`.

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
atomic writers have completed.  Routing Common force producers directly into
this buffer remains separate work.

### Writer completion and consumer synchronization

This contract is already enforced by the current single-queue runtime; it does
not require an additional shader barrier or a CPU wait after every dispatch.
All writers for a force evaluation must finish both word updates before any
consumer reads or reconstructs the logical value.  A low-word atomic, its carry,
and the high-word atomic are not one indivisible 64-bit operation.  Neither
`loadFixedPoint()` nor `getLongForceBuffer()` waits for writers by itself.

The current force-evaluation order is:

1. Clear the force accumulators on their owning queue.
2. Submit every enabled native float4 force producer on that queue.
3. Submit the float4-to-logical-64 conversion after the producers.  Its blocking
   validation-flag download also waits for this writer dispatch to complete;
   conversion failure prevents normal consumer execution.
4. Submit a logical-64 GPU consumer, such as Verlet, or a force download,
   checkpoint, or GPU save/restore copy after the final writer.
5. Submit the next force clear/write phase only after the preceding consumers.
   Energy-only evaluations preserve the force accumulators.

The implementation supporting this order is:

- `MetalKernel::execute()` in `src/MetalProgram.mm` creates a separate command
  buffer with a normal compute encoder for each dispatch, ends encoding, and
  commits it before returning.  It returns after submission, not GPU completion.
  `submissionMutex` serializes encoding and commit with blits on the same queue;
  callers must still submit producers before consumers, rather than racing
  independent host threads to establish that order.
- `src/MetalArray.mm` allocates individual buffers through `MTLDevice`, without
  opting out of hazard tracking.  Their default is tracked, not the untracked
  default of heaps.  Direct bindings expose the buffers to Metal; argument-buffer
  bindings additionally declare them with `useResource`.  For this
  `MTLCommandQueue` path, automatic hazard tracking supplies the required
  inter-command read/write dependencies.  It is not merely an assumption that
  submitting GPU work makes its memory immediately visible to the CPU.
  See Apple's [default tracking mode](https://developer.apple.com/documentation/metal/mtlhazardtrackingmode/default)
  and [tracked-resource guarantees](https://developer.apple.com/documentation/metal/mtlhazardtrackingmode/tracked).
- Downloads blit private storage to shared staging storage on the same queue.
  Blocking downloads wait for that command buffer and then copy to the caller's
  memory.  For `download(..., false)`, the caller must keep the destination alive
  and wait for `MetalQueue::waitUntilIdle()` before reading it; that method also
  drains host completion callbacks.  An event wait alone is not a replacement
  for draining an asynchronous download's host callback.

For future direct split-atomic producers, retain separate writer and reader
dispatches on this ordered, tracked-resource path, and remove the overwriting
float4 bridge as part of that migration.  Do not reconstruct a value while
other threads or threadgroups in the same dispatch may still update its words.
Relaxed atomics do not publish the completed pair, and a threadgroup barrier
cannot rendezvous all threadgroups; adding a shader fence does not supply that
missing execution boundary.

This contract does not cover cross-queue force-buffer access: kernel binding
and array copying currently reject buffers owned by another queue, even though
`MetalEvent` exposes queue signal/wait primitives.  Enabling cross-queue access,
untracked resources/heaps, concurrent dispatches, or an `MTL4CommandQueue`
requires a new synchronization implementation/review and writer-to-consumer
tests.  In particular, Metal 4 command queues do not apply the automatic hazard
tracking used here; the command-queue API is distinct from the MSL language
version.  See Apple's [hazard-tracking scope](https://developer.apple.com/documentation/metal/mtlhazardtrackingmode).

Existing tests cover ordered blit/compute/readback, contending split-atomic
updates followed by blocking readback, long-buffer save/restore, and production
logical-64 consumers.  The capability probe below also tests two split-atomic
writer dispatches feeding a GPU reconstruction dispatch without an intermediate
CPU wait.  Cross-queue writer/reader payload testing remains separate work.

### Runtime split fixed-point capability probe

`MetalQueue::getSplitFixedPointEmulationSupport()` returns an independent
`MetalCapabilityProbeResult`.  The first query blocks while it compiles the
production `fixedPoint.metal` helpers together with `capabilityProbe.metal`,
creates both compute pipelines, dispatches work, and validates the results.
The probe always compiles embedded MSL source through the selected device's
runtime with Metal 3.0 and production fast-math settings, even when production
kernels use an offline metallib.  The probe source is embedded but excluded
from the production offline library, so probe compilation failure can be
reported at runtime instead of preventing the plugin from building.
It never reads shader files at runtime.

The probe submits two sets of 2,053 contending writers across multiple
threadgroups, targeting five atoms in three padded component planes.  Cases
include positive and negative fractions, sub-Q32.32-unit truncation, low-word
carry, high-word wraparound, and signed endpoints.  A separate reader dispatch
loads the raw word pairs and reconstructs forces before any CPU wait.  Host
validation checks exact modulo-2^64 sums, reconstructed float values, unchanged
padding, and reader metadata.  Successful source compilation alone is not a
positive capability result.

The public result contains `status`, `diagnostic`, `isSupported()`, and
`getStatusName()`.  Status names are `not-run`, `supported`,
`initialization-failed`, `compilation-failed`, `pipeline-creation-failed`,
`execution-failed`, and `validation-failed`.  Runtime source compilation also
creates the Metal library; failure at that API boundary is reported as
`compilation-failed`.  Only a fully validated result reports support.  Failure
diagnostics distinguish the failed stage and do not imply that a transient
allocation/runtime failure proves a permanent hardware limitation.

Probes use an isolated sibling queue and private scratch buffers, drain their
GPU work/errors, and never touch simulation state.  The owning queue and all
its siblings share one thread-safe, immutable cached result.  A new independent
`MetalQueue` can retry, rather than inheriting a process-wide cached failure.
`MetalDeviceCaps::enumerate()` and device-family queries remain static queries
and do not dispatch a probe.

`MetalContext::getSupportsSplitFixedPointEmulation()` exposes the validated
Boolean separately from `getSupports64BitGlobalAtomics()`, which remains false.
Split accumulation still does not provide a linearizable 64-bit fetch-add or
a coherent previous 64-bit value.  Querying this capability does not change the
current float4 producers/bridge, enable future Common producers, or implement
automatic CPU fallback; future split-atomic paths must gate on this result.

`TestMetalRuntime` checks the real probe, concurrent first queries,
repeated/sibling/independent cache behavior, and isolated compile, pipeline,
host-rejected dispatch, and wrong-result failures.  Its execution-failure test
deliberately rejects an invalid block size
on the host; it does not deliberately fault the GPU.  `TestMetalComputeContext`
checks that split emulation and native 64-bit atomic support stay distinct.

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

Run the complete four-way build and GPU validation from the repository root:

```sh
python3 devtools/validate-metal-builds.py --jobs 8
```

This requires Python 3.8+, native Apple Silicon macOS, a visible supported Metal
GPU, CMake, Ninja, and Xcode with its Metal Toolchain installed.  Run outside a
sandbox that blocks the Metal compiler or GPU.  The runner uses Release builds
in four independent directories under `build/metal-matrix`:

| Directory | Kernel compilation | Shared library | Static library |
| --- | --- | --- | --- |
| `offline-shared` | `ON` (required metallib) | `ON` | `OFF` |
| `offline-static` | `ON` (required metallib) | `OFF` | `ON` |
| `runtime-shared` | `OFF` (runtime MSL) | `ON` | `OFF` |
| `runtime-static` | `OFF` (runtime MSL) | `OFF` | `ON` |

Each combination builds and executes all five tests listed below; static-only
builds use their `Static`-suffixed targets and do not build the shared OpenMM
library.  Unrelated platforms, plugins, wrappers, and test suites are disabled.
The runner checks the configured options, exact test inventory, actual CTest
results, and the fixed-point oracle's reported compilation path.  Missing,
failed, or skipped GPU tests fail validation, even when a legacy test returns
success after reporting that no device is visible.  `AUTO` is deliberately not
used: an offline-toolchain failure cannot silently become a runtime-path pass.
Failures do not prevent the remaining combinations from being attempted.

Configure/build/test logs and CTest XML remain in each build directory.  The
parent `validation-results.json` records the source revision, dirty-worktree
flag, per-case configuration, outcome, and test names.  Use `--build-root PATH`
to select a different parent, `--generator NAME` for a different CMake
generator, or repeat `--case NAME` to rerun selected combinations.  A partial
run is marked `full_matrix: false` and is not a four-way validation result.
The runner's result-validation checks can also be tested without a GPU with
`python3 -B -m unittest devtools/test_validate_metal_builds.py`.

In offline builds, `TestMetalComputeContext` also embeds a test-only
`FixedPointTests.metallib`: `tests/FixedPointTests.metal` includes the production
fixed-point/counter helpers and the same thin test entry kernels used by the
runtime path.  Both variants execute identical CPU bit-exact oracle and
contention tests; no arithmetic implementation is duplicated.  The generic
`compileProgram()` test and the capability probe still compile at runtime in
every configuration, since those specifically test the runtime compilation API.
The test library is not linked into the production plugin.

The focused test targets are:

- `TestMetalComputeContext`: the minimal `ComputeContext` contract, standard
  state-buffer ABI, `ComputeArray` interoperability, events, runtime MSL
  compilation through the generic compute interfaces, logical-64 force
  reconstruction, conversion boundaries, and checkpoint v1/v2/v3 compatibility
- `TestMetalPlatform`: plugin registration, device properties, and the required
  OpenMM kernel-factory surface
- `TestMetalRuntime`: buffers, transfers, queues, events, runtime MSL
  compilation, direct argument binding, reflected Tier 2 argument buffers,
  resize after kernel binding, and C++ wrapper lifetime independence
- `TestMetalVerticalSlice`: analytic force/energy values, Verlet state changes,
  Reference trajectory comparison, parameter updates, checkpoint replay,
  minimization, shared-particle bond accumulation, and rejection of unsupported
  features; direct logical-buffer consumer tests also verify Verlet, force
  download, shifted velocities, kinetic energy, and energy-only preservation
- `TestMetalNonbondedForce`: analytic Coulomb/Lennard-Jones values, exceptions,
  parameter updates, force groups, include flags, multi-threadgroup execution,
  Reference trajectory comparison, and rejection of unsupported methods

The CPU bit-exact fixed-point oracle coverage spans `TestMetalComputeContext`
and the capability probe exercised by `TestMetalRuntime`: low-to-high carry,
negative/two's-complement values, exact positive/negative cancellation,
modulo-2^64 wraparound, repeated carries, and unchanged padded regions.
`testFixedPointCancellation()` reuses the production atomic helper through the
existing test entry kernel.  It checks both sign orders with single and
32,768-writer dispatches, zero and nonzero seeds, and fractional low bits that
float reconstruction cannot preserve.  Every dispatch is checked against CPU
`uint64_t` sums by comparing both raw words across all three padded planes;
the intermediate result must change before cancellation restores the seed.

Passing these tests proves the native runtime and the documented vertical slice.
It does not claim coverage of OpenMM's full kernel corpus or Intel/AMD Metal
devices.
