kernel void addCounter32Atomically(
        device atomic_int* counters [[buffer(0)]],
        device int* previousValues [[buffer(1)]],
        constant uint& counterIndex [[buffer(2)]],
        constant int& addend [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
    previousValues[index] = atomicFetchAddCounter32(
            &counters[counterIndex], addend);
}
