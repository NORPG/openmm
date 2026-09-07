// Compile the production helpers with the same entry points used by the runtime
// oracle tests.  This test-only library never enters the production metallib.
#include "../src/kernels/counterAtomics.metal"
#include "../src/kernels/fixedPoint.metal"
#include "kernels/counterAtomics.metal"
#include "kernels/fixedPointHelpers.metal"
