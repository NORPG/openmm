#ifndef CUSTOM_GB_READONLY_FIELDS
#define CUSTOM_GB_READONLY_FIELDS(OP)
#endif
#ifndef CUSTOM_GB_ACCUMULATOR_FIELDS
#define CUSTOM_GB_ACCUMULATOR_FIELDS(OP)
#endif

#if defined(__METAL__) && __METAL_VERSION__ >= 300 && USE_CUSTOM_GB_ENERGY_SHUFFLE
#if TILE_SIZE != 32
#error Metal CustomGB register paths require TILE_SIZE=32
#endif
#define CUSTOM_GB_SHUFFLE 1
#define METAL_GB_THREAD thread
/** Keep Common's generated array indexing while each lane owns its record. */
template<class T> struct MetalGBLaneValue {
    T value = T(0);
    T broadcast = T(0);
    bool diagonal = false;
    METAL_GB_THREAD T& operator[](uint) METAL_GB_THREAD {
        return diagonal ? broadcast : value;
    }
};
#undef METAL_GB_THREAD
#define CUSTOM_GB_DECLARE_PARAMETER(type, name) MetalGBLaneValue<type> name;
#define CUSTOM_GB_BROADCAST_PARAMETER(type, name) name.diagonal = true; name.broadcast = simdShuffle(name.value, j);
#define CUSTOM_GB_END_DIAGONAL(type, name) name.diagonal = false;
#define CUSTOM_GB_ROTATE_PARAMETER(type, name) name.value = simdShuffle(name.value, (tgx+1)&31);
#define CUSTOM_GB_DECLARE(type, name) type name = type(0)
#define CUSTOM_GB_LOCAL(name, index) name
#define CUSTOM_GB_DIAGONAL(name, index) name##_broadcast
#define CUSTOM_GB_BROADCAST(type, name) type name##_broadcast = simdShuffle(name, j);
#define CUSTOM_GB_ROTATE(type, name) name = simdShuffle(name, (tgx+1)&31);
#define CUSTOM_GB_SYNC
#define CUSTOM_GB_SKIP(index) simdShuffle(skipTile, uint((index)-tbx))
#define CUSTOM_GB_SKIP_LOCAL skipTile
#else
#define CUSTOM_GB_SHUFFLE 0
#define CUSTOM_GB_BROADCAST_PARAMETER(type, name)
#define CUSTOM_GB_END_DIAGONAL(type, name)
#define CUSTOM_GB_ROTATE_PARAMETER(type, name)
#define CUSTOM_GB_DECLARE(type, name) LOCAL type name[LOCAL_BUFFER_SIZE]
#define CUSTOM_GB_LOCAL(name, index) name[index]
#define CUSTOM_GB_DIAGONAL(name, index) name[index]
#define CUSTOM_GB_BROADCAST(type, name)
#define CUSTOM_GB_ROTATE(type, name)
#define CUSTOM_GB_SYNC SYNC_WARPS
#define CUSTOM_GB_SKIP(index) skipTiles[index]
#define CUSTOM_GB_SKIP_LOCAL skipTiles[LOCAL_ID]
#endif

#define STORE_DERIVATIVE_1(INDEX) ATOMIC_ADD(&derivBuffers[offset+(INDEX-1)*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(deriv##INDEX##_1));
#define STORE_DERIVATIVE_2(INDEX) ATOMIC_ADD(&derivBuffers[offset+(INDEX-1)*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(local_deriv##INDEX[LOCAL_ID]));

/**
 * Compute a force based on pair interactions.
 */
KERNEL void computeN2Energy(
        GLOBAL mm_ulong* RESTRICT forceBuffers,
        GLOBAL mixed* RESTRICT energyBuffer,
        GLOBAL const real4* RESTRICT posq, GLOBAL const unsigned int* RESTRICT exclusions,
        GLOBAL const int2* exclusionTiles, int needEnergy,
#ifdef USE_CUTOFF
        GLOBAL const int* RESTRICT tiles, GLOBAL const unsigned int* RESTRICT interactionCount, real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ, unsigned int maxTiles, GLOBAL const real4* RESTRICT blockCenter,
        GLOBAL const real4* RESTRICT blockSize, GLOBAL const int* RESTRICT interactingAtoms
#else
        unsigned int numTiles
#endif
        PARAMETER_ARGUMENTS) {
    const unsigned int totalWarps = GLOBAL_SIZE/TILE_SIZE;
    const unsigned int warp = GLOBAL_ID/TILE_SIZE;
    const unsigned int tgx = LOCAL_ID & (TILE_SIZE-1);
    const unsigned int tbx = LOCAL_ID - tgx;
    mixed energy = 0;
    INIT_PARAM_DERIVS
    CUSTOM_GB_DECLARE(real3, local_pos);
    CUSTOM_GB_DECLARE(real3, local_force);
#if CUSTOM_GB_SHUFFLE
    int atomIndices = 0;
#endif
#if CUSTOM_GB_SHUFFLE
    CUSTOM_GB_READONLY_FIELDS(CUSTOM_GB_DECLARE_PARAMETER)
    CUSTOM_GB_ACCUMULATOR_FIELDS(CUSTOM_GB_DECLARE_PARAMETER)
#else
    // OPENMM_METAL_CUSTOM_GB_FIELDS_BEGIN
    ATOM_PARAMETER_DATA
    // OPENMM_METAL_CUSTOM_GB_FIELDS_END
#endif

    // First loop: process tiles that contain exclusions.
    
    const int firstExclusionTile = FIRST_EXCLUSION_TILE+warp*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    const int lastExclusionTile = FIRST_EXCLUSION_TILE+(warp+1)*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    for (int pos = firstExclusionTile; pos < lastExclusionTile; pos++) {
        const int2 tileIndices = exclusionTiles[pos];
        const unsigned int x = tileIndices.x;
        const unsigned int y = tileIndices.y;
        real3 force = make_real3(0);
        DECLARE_ATOM1_DERIVATIVES
        unsigned int atom1 = x*TILE_SIZE + tgx;
        real3 pos1 = trimTo3(posq[atom1]);
        LOAD_ATOM1_PARAMETERS
#ifdef USE_EXCLUSIONS
        unsigned int excl = exclusions[pos*TILE_SIZE+tgx];
#endif
        if (x == y) {
            // This tile is on the diagonal.

            const unsigned int localAtomIndex = LOCAL_ID;
            CUSTOM_GB_LOCAL(local_pos, localAtomIndex) = pos1;
            LOAD_LOCAL_PARAMETERS_FROM_1
            CUSTOM_GB_SYNC;
            for (unsigned int j = 0; j < TILE_SIZE; j++) {
                // Every lane broadcasts before particle/cutoff predicates.
                CUSTOM_GB_BROADCAST(real3, local_pos)
                CUSTOM_GB_READONLY_FIELDS(CUSTOM_GB_BROADCAST_PARAMETER)
                int atom2 = tbx+j;
                real3 pos2 = CUSTOM_GB_DIAGONAL(local_pos, atom2);
                real3 delta = make_real3(pos2.x-pos1.x, pos2.y-pos1.y, pos2.z-pos1.z);
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(delta)
#endif
                real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
#ifdef USE_CUTOFF
                if (r2 < CUTOFF_SQUARED) {
#endif
                    real invR = RSQRT(r2);
                    real r = r2*invR;
                    LOAD_ATOM2_PARAMETERS
                    atom2 = y*TILE_SIZE+j;
                    real dEdR = 0;
                    real tempEnergy = 0;
                    const real interactionScale = 0.5f;
#ifdef USE_EXCLUSIONS
                    bool isExcluded = !(excl & 0x1);
#endif
                    if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS && atom1 != atom2) {
                        COMPUTE_INTERACTION
                        dEdR /= -r;
                    }
                    if (needEnergy)
                        energy += 0.5f*tempEnergy;
                    delta *= dEdR;
                    force.x -= delta.x;
                    force.y -= delta.y;
                    force.z -= delta.z;
#ifdef USE_CUTOFF
                }
#endif
#ifdef USE_EXCLUSIONS
                excl >>= 1;
#endif
                CUSTOM_GB_SYNC;
            }
            CUSTOM_GB_READONLY_FIELDS(CUSTOM_GB_END_DIAGONAL)
        }
        else {
            // This is an off-diagonal tile.

            const unsigned int localAtomIndex = LOCAL_ID;
            unsigned int j = y*TILE_SIZE + tgx;
            CUSTOM_GB_LOCAL(local_pos, localAtomIndex) = trimTo3(posq[j]);
            LOAD_LOCAL_PARAMETERS_FROM_GLOBAL
            CUSTOM_GB_LOCAL(local_force, localAtomIndex) = make_real3(0);
            CLEAR_LOCAL_DERIVATIVES
            CUSTOM_GB_SYNC;
#ifdef USE_EXCLUSIONS
            excl = (excl >> tgx) | (excl << (TILE_SIZE - tgx));
#endif
            unsigned int tj = tgx;
            for (j = 0; j < TILE_SIZE; j++) {
                int atom2 = tbx+tj;
                real3 pos2 = CUSTOM_GB_LOCAL(local_pos, atom2);
                real3 delta = make_real3(pos2.x-pos1.x, pos2.y-pos1.y, pos2.z-pos1.z);
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(delta)
#endif
                real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
#ifdef USE_CUTOFF
                if (r2 < CUTOFF_SQUARED) {
#endif
                    real invR = RSQRT(r2);
                    real r = r2*invR;
                    LOAD_ATOM2_PARAMETERS
                    atom2 = y*TILE_SIZE+tj;
                    real dEdR = 0;
                    real tempEnergy = 0;
                    const real interactionScale = 1;
#ifdef USE_EXCLUSIONS
                    bool isExcluded = !(excl & 0x1);
#endif
                    if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS) {
                        COMPUTE_INTERACTION
                        dEdR /= -r;
                    }
                    if (needEnergy)
                        energy += tempEnergy;
                    delta *= dEdR;
                    force.x -= delta.x;
                    force.y -= delta.y;
                    force.z -= delta.z;
                    atom2 = tbx+tj;
                    CUSTOM_GB_LOCAL(local_force, atom2) += delta;
                    RECORD_DERIVATIVE_2
#ifdef USE_CUTOFF
                }
#endif
#ifdef USE_EXCLUSIONS
                excl >>= 1;
#endif
                tj = (tj + 1) & (TILE_SIZE - 1);
                CUSTOM_GB_ROTATE(real3, local_pos)
                CUSTOM_GB_ROTATE(real3, local_force)
                CUSTOM_GB_READONLY_FIELDS(CUSTOM_GB_ROTATE_PARAMETER)
                CUSTOM_GB_ACCUMULATOR_FIELDS(CUSTOM_GB_ROTATE_PARAMETER)
                CUSTOM_GB_ROTATE(int, atomIndices)
                CUSTOM_GB_SYNC;
            }
        }

        // Write results.

        unsigned int offset = x*TILE_SIZE + tgx;
        ATOMIC_ADD(&forceBuffers[offset], (mm_ulong) realToFixedPoint(force.x));
        ATOMIC_ADD(&forceBuffers[offset+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(force.y));
        ATOMIC_ADD(&forceBuffers[offset+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(force.z));
        STORE_DERIVATIVES_1
        if (x != y) {
            offset = y*TILE_SIZE + tgx;
            ATOMIC_ADD(&forceBuffers[offset], (mm_ulong) realToFixedPoint(CUSTOM_GB_LOCAL(local_force, LOCAL_ID).x));
            ATOMIC_ADD(&forceBuffers[offset+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(CUSTOM_GB_LOCAL(local_force, LOCAL_ID).y));
            ATOMIC_ADD(&forceBuffers[offset+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(CUSTOM_GB_LOCAL(local_force, LOCAL_ID).z));
            STORE_DERIVATIVES_2
        }
    }

    // Second loop: tiles without exclusions, either from the neighbor list (with cutoff) or just enumerating all
    // of them (no cutoff).

#ifdef USE_CUTOFF
    unsigned int numTiles = interactionCount[0];
    if (numTiles > maxTiles)
        return; // There wasn't enough memory for the neighbor list.
    int pos = (int) (warp*(numTiles > maxTiles ? NUM_BLOCKS*((mm_long)NUM_BLOCKS+1)/2 : (mm_long)numTiles)/totalWarps);
    int end = (int) ((warp+1)*(numTiles > maxTiles ? NUM_BLOCKS*((mm_long)NUM_BLOCKS+1)/2 : (mm_long)numTiles)/totalWarps);
#else
    int pos = (int) (warp*(mm_long)numTiles/totalWarps);
    int end = (int) ((warp+1)*(mm_long)numTiles/totalWarps);
#endif
    int skipBase = 0;
    int currentSkipIndex = tbx;
#if CUSTOM_GB_SHUFFLE
    int skipTile = -1;
#else
    LOCAL int atomIndices[LOCAL_BUFFER_SIZE];
    LOCAL volatile int skipTiles[LOCAL_BUFFER_SIZE];
    skipTiles[LOCAL_ID] = -1;
#endif

    while (pos < end) {
        const bool isExcluded = false;
        real3 force = make_real3(0);
        DECLARE_ATOM1_DERIVATIVES
        bool includeTile = true;

        // Extract the coordinates of this tile.
        
        int x, y;
        bool singlePeriodicCopy = false;
#ifdef USE_CUTOFF
        x = tiles[pos];
        real4 blockSizeX = blockSize[x];
        singlePeriodicCopy = (0.5f*periodicBoxSize.x-blockSizeX.x >= CUTOFF &&
                              0.5f*periodicBoxSize.y-blockSizeX.y >= CUTOFF &&
                              0.5f*periodicBoxSize.z-blockSizeX.z >= CUTOFF);
#else
        y = (int) floor(NUM_BLOCKS+0.5f-SQRT((NUM_BLOCKS+0.5f)*(NUM_BLOCKS+0.5f)-2*pos));
        x = (pos-y*NUM_BLOCKS+y*(y+1)/2);
        if (x < y || x >= NUM_BLOCKS) { // Occasionally happens due to roundoff error.
            y += (x < y ? -1 : 1);
            x = (pos-y*NUM_BLOCKS+y*(y+1)/2);
        }

        // Skip over tiles that have exclusions, since they were already processed.

        CUSTOM_GB_SYNC;
        while (CUSTOM_GB_SKIP(tbx+TILE_SIZE-1) < pos) {
            CUSTOM_GB_SYNC;
            if (skipBase+tgx < NUM_TILES_WITH_EXCLUSIONS) {
                int2 tile = exclusionTiles[skipBase+tgx];
                CUSTOM_GB_SKIP_LOCAL = tile.x + tile.y*NUM_BLOCKS - tile.y*(tile.y+1)/2;
            }
            else
                CUSTOM_GB_SKIP_LOCAL = end;
            skipBase += TILE_SIZE;            
            currentSkipIndex = tbx;
            CUSTOM_GB_SYNC;
        }
        while (CUSTOM_GB_SKIP(currentSkipIndex) < pos)
            currentSkipIndex++;
        includeTile = (CUSTOM_GB_SKIP(currentSkipIndex) != pos);
#endif
        if (includeTile) {
            unsigned int atom1 = x*TILE_SIZE + tgx;

            // Load atom data for this tile.

            real3 pos1 = trimTo3(posq[atom1]);
            LOAD_ATOM1_PARAMETERS
            const unsigned int localAtomIndex = LOCAL_ID;
#ifdef USE_CUTOFF
            unsigned int j = interactingAtoms[pos*TILE_SIZE+tgx];
#else
            unsigned int j = y*TILE_SIZE + tgx;
#endif
            CUSTOM_GB_LOCAL(atomIndices, LOCAL_ID) = j;
            if (j < PADDED_NUM_ATOMS) {
                CUSTOM_GB_LOCAL(local_pos, localAtomIndex) = trimTo3(posq[j]);
                LOAD_LOCAL_PARAMETERS_FROM_GLOBAL
                CUSTOM_GB_LOCAL(local_force, localAtomIndex) = make_real3(0);
                CLEAR_LOCAL_DERIVATIVES
            }
            CUSTOM_GB_SYNC;
#ifdef USE_PERIODIC
            if (singlePeriodicCopy) {
                // The box is small enough that we can just translate all the atoms into a single periodic
                // box, then skip having to apply periodic boundary conditions later.

                real4 blockCenterX = blockCenter[x];
                APPLY_PERIODIC_TO_POS_WITH_CENTER(pos1, blockCenterX)
                APPLY_PERIODIC_TO_POS_WITH_CENTER(CUSTOM_GB_LOCAL(local_pos, LOCAL_ID), blockCenterX)
                CUSTOM_GB_SYNC;
                unsigned int tj = tgx;
                for (j = 0; j < TILE_SIZE; j++) {
                    int atom2 = tbx+tj;
                    real3 pos2 = CUSTOM_GB_LOCAL(local_pos, atom2);
                    real3 delta = make_real3(pos2.x-pos1.x, pos2.y-pos1.y, pos2.z-pos1.z);
                    real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                    if (r2 < CUTOFF_SQUARED) {
                        real invR = RSQRT(r2);
                        real r = r2*invR;
                        LOAD_ATOM2_PARAMETERS
                        atom2 = CUSTOM_GB_LOCAL(atomIndices, tbx+tj);
                        real dEdR = 0;
                        real tempEnergy = 0;
                        const real interactionScale = 1;
                        if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS) {
                            COMPUTE_INTERACTION
                            dEdR /= -r;
                        }
                        if (needEnergy)
                            energy += tempEnergy;
                        delta *= dEdR;
                        force.x -= delta.x;
                        force.y -= delta.y;
                        force.z -= delta.z;
                        atom2 = tbx+tj;
                        CUSTOM_GB_LOCAL(local_force, atom2) += delta;
                        RECORD_DERIVATIVE_2
                    }
                    tj = (tj + 1) & (TILE_SIZE - 1);
                    CUSTOM_GB_ROTATE(real3, local_pos)
                    CUSTOM_GB_ROTATE(real3, local_force)
                    CUSTOM_GB_READONLY_FIELDS(CUSTOM_GB_ROTATE_PARAMETER)
                    CUSTOM_GB_ACCUMULATOR_FIELDS(CUSTOM_GB_ROTATE_PARAMETER)
                    CUSTOM_GB_ROTATE(int, atomIndices)
                    CUSTOM_GB_SYNC;
                }
            }
            else
#endif
            {
                // We need to apply periodic boundary conditions separately for each interaction.

                unsigned int tj = tgx;
                for (j = 0; j < TILE_SIZE; j++) {
                    int atom2 = tbx+tj;
                    real3 pos2 = CUSTOM_GB_LOCAL(local_pos, atom2);
                    real3 delta = make_real3(pos2.x-pos1.x, pos2.y-pos1.y, pos2.z-pos1.z);
#ifdef USE_PERIODIC
                    APPLY_PERIODIC_TO_DELTA(delta)
#endif
                    real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
#ifdef USE_CUTOFF
                    if (r2 < CUTOFF_SQUARED) {
#endif
                        real invR = RSQRT(r2);
                        real r = r2*invR;
                        LOAD_ATOM2_PARAMETERS
                        atom2 = CUSTOM_GB_LOCAL(atomIndices, tbx+tj);
                        real dEdR = 0;
                        real tempEnergy = 0;
                        const real interactionScale = 1;
                        if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS) {
                            COMPUTE_INTERACTION
                            dEdR /= -r;
                        }
                        if (needEnergy)
                            energy += tempEnergy;
                        delta *= dEdR;
                        force.x -= delta.x;
                        force.y -= delta.y;
                        force.z -= delta.z;
                        atom2 = tbx+tj;
                        CUSTOM_GB_LOCAL(local_force, atom2) += delta;
                        RECORD_DERIVATIVE_2
#ifdef USE_CUTOFF
                    }
#endif
                    tj = (tj + 1) & (TILE_SIZE - 1);
                    CUSTOM_GB_ROTATE(real3, local_pos)
                    CUSTOM_GB_ROTATE(real3, local_force)
                    CUSTOM_GB_READONLY_FIELDS(CUSTOM_GB_ROTATE_PARAMETER)
                    CUSTOM_GB_ACCUMULATOR_FIELDS(CUSTOM_GB_ROTATE_PARAMETER)
                    CUSTOM_GB_ROTATE(int, atomIndices)
                    CUSTOM_GB_SYNC;
                }
            }
        
            // Write results.

#ifdef USE_CUTOFF
            unsigned int atom2 = CUSTOM_GB_LOCAL(atomIndices, LOCAL_ID);
#else
            unsigned int atom2 = y*TILE_SIZE + tgx;
#endif
            ATOMIC_ADD(&forceBuffers[atom1], (mm_ulong) realToFixedPoint(force.x));
            ATOMIC_ADD(&forceBuffers[atom1+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(force.y));
            ATOMIC_ADD(&forceBuffers[atom1+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(force.z));
            unsigned int offset = atom1;
            STORE_DERIVATIVES_1
            if (atom2 < PADDED_NUM_ATOMS) {
                ATOMIC_ADD(&forceBuffers[atom2], (mm_ulong) realToFixedPoint(CUSTOM_GB_LOCAL(local_force, LOCAL_ID).x));
                ATOMIC_ADD(&forceBuffers[atom2+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(CUSTOM_GB_LOCAL(local_force, LOCAL_ID).y));
                ATOMIC_ADD(&forceBuffers[atom2+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(CUSTOM_GB_LOCAL(local_force, LOCAL_ID).z));
                offset = atom2;
                STORE_DERIVATIVES_2
            }
        }
        pos++;
    }
    energyBuffer[GLOBAL_ID] += energy;
    SAVE_PARAM_DERIVS
}
