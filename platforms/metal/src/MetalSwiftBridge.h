#ifndef OPENMM_METALSWIFTBRIDGE_H_
#define OPENMM_METALSWIFTBRIDGE_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * Private, C-compatible ABI between OpenMM's C++ Metal wrappers and the      *
 * native Swift Metal runtime.  No Swift or Objective-C type crosses this     *
 * boundary.                                                                  *
 * -------------------------------------------------------------------------- */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* OMMMetalQueueRef;
typedef void* OMMMetalBufferRef;
typedef void* OMMMetalProgramRef;
typedef void* OMMMetalKernelRef;
typedef void* OMMMetalEventRef;
typedef void* OMMMetalNonbondedRef;

enum {
    OMM_METAL_SUCCESS = 0,
    OMM_METAL_ERROR_INVALID_ARGUMENT = 1,
    OMM_METAL_ERROR_OUT_OF_RANGE = 2,
    OMM_METAL_ERROR_UNSUPPORTED = 3,
    OMM_METAL_ERROR_NATIVE = 4,
    OMM_METAL_ERROR_INVALID_UTF8 = 5,
    OMM_METAL_ERROR_INTERNAL = 255
};

enum {
    OMM_METAL_BINDING_DIRECT = 0,
    OMM_METAL_BINDING_ARGUMENT_BUFFER = 1
};

/**
 * Error returned by a fallible bridge function.
 *
 * Initialize this structure to zero before its first use.  When a call fails,
 * message is allocated by the Swift runtime.  Call OMMMetalErrorClear() once
 * the message has been copied.  Success leaves code zero and message NULL.
 */
typedef struct OMMMetalErrorC {
    int32_t code;
    char* message;
} OMMMetalErrorC;

/** A C snapshot of one Metal device.  name is bridge-allocated. */
typedef struct OMMMetalDeviceCapsC {
    uint64_t registryId;
    uint64_t maxBufferLength;
    uint64_t recommendedMaxWorkingSetSize;
    uint64_t maxThreadgroupMemoryLength;
    uint64_t maxThreadsPerThreadgroup;
    int32_t argumentBufferTier;
    int32_t highestAppleGpuFamily;
    uint8_t lowPower;
    uint8_t removable;
    uint8_t unifiedMemory;
    uint8_t mac2;
    uint8_t metal3;
    uint8_t metal4;
    char* name;
} OMMMetalDeviceCapsC;

/** Host values used to initialize or update one NonbondedForce particle. */
typedef struct OMMMetalNonbondedParticleC {
    double charge;
    double sigma;
    double epsilon;
} OMMMetalNonbondedParticleC;

/** Host values and immutable endpoints for one NonbondedForce exception. */
typedef struct OMMMetalNonbondedExceptionC {
    uint32_t particle1;
    uint32_t particle2;
    double chargeProd;
    double sigma;
    double epsilon;
} OMMMetalNonbondedExceptionC;

void OMMMetalErrorClear(OMMMetalErrorC* error);
void OMMMetalDeviceCapsClear(OMMMetalDeviceCapsC* caps);

int32_t OMMMetalDeviceCount(size_t* count, OMMMetalErrorC* error);
int32_t OMMMetalDeviceCapsAt(size_t index, OMMMetalDeviceCapsC* caps, OMMMetalErrorC* error);
int32_t OMMMetalDefaultDevice(uint8_t* supported, OMMMetalDeviceCapsC* caps, OMMMetalErrorC* error);

int32_t OMMMetalQueueCreate(size_t deviceIndex, OMMMetalQueueRef* queue,
                            OMMMetalDeviceCapsC* caps, OMMMetalErrorC* error);
int32_t OMMMetalQueueCreateSibling(OMMMetalQueueRef parent, OMMMetalQueueRef* queue,
                                   OMMMetalErrorC* error);
void OMMMetalQueueRelease(OMMMetalQueueRef queue);
int32_t OMMMetalQueueWaitUntilIdle(OMMMetalQueueRef queue, OMMMetalErrorC* error);
int32_t OMMMetalQueueCheckForErrors(OMMMetalQueueRef queue, OMMMetalErrorC* error);

int32_t OMMMetalBufferCreate(OMMMetalQueueRef queue, size_t bytes,
                             const char* name, size_t nameLength,
                             OMMMetalBufferRef* buffer, OMMMetalErrorC* error);
void OMMMetalBufferRelease(OMMMetalBufferRef buffer);
int32_t OMMMetalBufferResize(OMMMetalBufferRef buffer, size_t bytes, OMMMetalErrorC* error);
int32_t OMMMetalBufferUpload(OMMMetalBufferRef buffer, size_t byteOffset,
                             const void* source, size_t bytes, uint8_t blocking,
                             OMMMetalErrorC* error);
int32_t OMMMetalBufferDownload(OMMMetalBufferRef buffer, size_t byteOffset,
                               void* destination, size_t bytes, uint8_t blocking,
                               OMMMetalErrorC* error);
int32_t OMMMetalBufferCopy(OMMMetalBufferRef source, size_t sourceOffset,
                           OMMMetalBufferRef destination, size_t destinationOffset,
                           size_t bytes, OMMMetalErrorC* error);

int32_t OMMMetalProgramCreateSource(OMMMetalQueueRef queue,
                                    const char* source, size_t sourceLength,
                                    uint8_t fastMath, uint16_t languageMajor,
                                    uint16_t languageMinor,
                                    OMMMetalProgramRef* program, OMMMetalErrorC* error);
int32_t OMMMetalProgramCreateLibrary(OMMMetalQueueRef queue,
                                     const void* libraryBytes, size_t libraryLength,
                                     OMMMetalProgramRef* program, OMMMetalErrorC* error);
void OMMMetalProgramRelease(OMMMetalProgramRef program);

int32_t OMMMetalKernelCreate(OMMMetalProgramRef program,
                             const char* name, size_t nameLength,
                             int32_t bindingMode, int32_t argumentBufferIndex,
                             OMMMetalKernelRef* kernel, OMMMetalErrorC* error);
void OMMMetalKernelRelease(OMMMetalKernelRef kernel);
int32_t OMMMetalKernelMaxBlockSize(OMMMetalKernelRef kernel, int32_t* result,
                                   OMMMetalErrorC* error);
int32_t OMMMetalKernelAddBuffer(OMMMetalKernelRef kernel, OMMMetalBufferRef buffer,
                                OMMMetalErrorC* error);
int32_t OMMMetalKernelAddBytes(OMMMetalKernelRef kernel, const void* value,
                               size_t size, OMMMetalErrorC* error);
int32_t OMMMetalKernelAddEmpty(OMMMetalKernelRef kernel, OMMMetalErrorC* error);
int32_t OMMMetalKernelSetBuffer(OMMMetalKernelRef kernel, int32_t index,
                                OMMMetalBufferRef buffer, OMMMetalErrorC* error);
int32_t OMMMetalKernelSetBytes(OMMMetalKernelRef kernel, int32_t index,
                               const void* value, size_t size, OMMMetalErrorC* error);
int32_t OMMMetalKernelExecute(OMMMetalKernelRef kernel, int32_t threads,
                              int32_t blockSize, OMMMetalErrorC* error);

/**
 * Create a Swift-owned NoCutoff nonbonded execution plan.  The plan retains
 * the two context buffers and owns its topology, typed parameter buffers,
 * pipeline, and energy reduction storage until OMMMetalNonbondedRelease().
 */
int32_t OMMMetalNonbondedCreate(
        OMMMetalBufferRef positions, OMMMetalBufferRef forces,
        const void* libraryBytes, size_t libraryLength,
        const OMMMetalNonbondedParticleC* particles, size_t particleCount,
        const OMMMetalNonbondedExceptionC* exceptions, size_t exceptionCount,
        OMMMetalNonbondedRef* plan, OMMMetalErrorC* error);
void OMMMetalNonbondedRelease(OMMMetalNonbondedRef plan);
int32_t OMMMetalNonbondedExecute(OMMMetalNonbondedRef plan,
                                 uint8_t includeForces, uint8_t includeEnergy,
                                 double* energy, OMMMetalErrorC* error);
int32_t OMMMetalNonbondedUpdate(
        OMMMetalNonbondedRef plan,
        const OMMMetalNonbondedParticleC* particles, size_t particleCount,
        const OMMMetalNonbondedExceptionC* exceptions, size_t exceptionCount,
        int32_t firstParticle, int32_t lastParticle,
        int32_t firstException, int32_t lastException,
        OMMMetalErrorC* error);

int32_t OMMMetalEventCreate(OMMMetalQueueRef queue, OMMMetalEventRef* event,
                            OMMMetalErrorC* error);
void OMMMetalEventRelease(OMMMetalEventRef event);
int32_t OMMMetalEventEnqueue(OMMMetalEventRef event, OMMMetalErrorC* error);
int32_t OMMMetalEventWait(OMMMetalEventRef event, OMMMetalErrorC* error);
int32_t OMMMetalEventQueueWait(OMMMetalEventRef event, OMMMetalQueueRef queue,
                               OMMMetalErrorC* error);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OPENMM_METALSWIFTBRIDGE_H_ */
