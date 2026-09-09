# Experimental Metal Platform

This revision provides the OpenCL GPU baseline through Common Compute and native Metal sources.
Common kernels retain their original baseline until each feature implementation lands.
Metal does not load architecture sources from OpenCL, CUDA, or HIP.
An exposed alternative requires its independent switch, selected language support, and runtime eligibility.
Switches do not raise the selected language version. No performance claim is made.

Command submission is immediate at this stage. Batching is not implemented or exposed.
Record-and-commit is forced ON even when an existing caller cache requests OFF.

## Available independent build switches

| Build option | OFF behavior | ON behavior |
| --- | --- | --- |
| `OPENMM_METAL_MINIMIZE_FLOAT_ACCUMULATORS` | Q32.32 minimization with limited range diagnostics; no CPU fallback | Scoped GPU floating accumulators for large-force minimization (default) |
| `OPENMM_METAL_FAST_SHORT_LIST_SORT` | Baseline Metal short-list selection | Native Metal parallel short-list kernel |
| `OPENMM_METAL_FAST_SPARSE_PAIRS` | All interactions in tiles | Native Metal sparse-pair phase |

Benchmark metadata and the BenchmarkMetal target are not available at this stage.
