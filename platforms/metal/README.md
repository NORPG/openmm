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
| `OPENMM_METAL_FAST_NONBONDED_SHUFFLE` | Metal threadgroup-memory force tiles | Native Metal register/shuffle force tiles |
| `OPENMM_METAL_FAST_MINIMIZE_SHUFFLE` | Common local-memory reduction | Common CUDA/HIP shuffle reduction in the minimizer only |
| `OPENMM_METAL_FAST_CONSTANT_POTENTIAL_REDUCTION` | Base ConstantPotential local-memory reduction | Common shuffle reduction in the base ConstantPotential kernels only |
| `OPENMM_METAL_FAST_CONSTANT_POTENTIAL_CG_REDUCTION` | CG local-memory reductions | Common scalar and compensated `float2` shuffle reductions in CG only |
| `OPENMM_METAL_FAST_CONSTANT_POTENTIAL_MATRIX_REDUCTION` | Matrix-solver local-memory reduction | Common shuffle reduction in the matrix solver only |
| `OPENMM_METAL_FAST_CONSTANT_POTENTIAL_MATRIX_BROADCAST` | Matrix-solver local-memory broadcasts | Common lane-shuffle broadcasts in the matrix solver only |
| `OPENMM_METAL_FAST_CUSTOM_NONBONDED_GROUPS_SHUFFLE` | Interaction-group local-memory maximum | Common CUDA xor-shuffle maximum for CustomNonbonded interaction groups |
| `OPENMM_METAL_FAST_LCPO_BALLOT` | LCPO local neighbor-block flags | Common CUDA ballot-based neighbor-block scan |
| `OPENMM_METAL_FAST_CUSTOM_MANY_PARTICLE_BALLOT` | CustomManyParticle local neighbor-block flags | Common CUDA ballot-based neighbor-block scan |
| `OPENMM_METAL_FAST_CENTROID_REDUCTION` | Common centroid local reduction | SIMD reductions with per-group shared partials |
| `OPENMM_METAL_FAST_RG_REDUCTION` | Common radius-of-gyration local reduction | SIMD reductions with per-group shared partials |
| `OPENMM_METAL_FAST_RMSD_REDUCTION` | Common RMSD local reduction | SIMD center and correlation reductions |
| `OPENMM_METAL_FAST_ORIENTATION_REDUCTION` | Common orientation local reduction | SIMD center and correlation reductions |
| `OPENMM_METAL_FAST_LCPO_NEIGHBOR_SCAN` | Common LCPO shared prefix scan | Hierarchical SIMD prefix scan |
| `OPENMM_METAL_FAST_CUSTOM_MANY_PARTICLE_NEIGHBOR_SCAN` | Common CustomManyParticle shared prefix scan | Hierarchical SIMD prefix scan |
| `OPENMM_METAL_FAST_CUSTOM_GB_VALUE_SHUFFLE` | Common CustomGB value local arrays | Register exchange for value, parameters, and secondary accumulators |
| `OPENMM_METAL_FAST_CUSTOM_GB_ENERGY_SHUFFLE` | Common CustomGB energy local arrays | Register exchange for force and parameter-derivative state |
| `OPENMM_METAL_FAST_DPD_PARTICLE_SHUFFLE` | Common DPD local particle arrays | Register broadcasts preserving original pair/RNG visitation order |
| `OPENMM_METAL_FAST_DPD_TILE_BROADCAST` | Local-memory tile-counter broadcast | SIMD lane-zero broadcast |

Benchmark metadata and the BenchmarkMetal target are not available at this stage.
