set(METAL_TEST_KERNEL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(METAL_TEST_KERNEL_SOURCE_CLASS MetalTestKernelSources)
set(METAL_TEST_KERNELS_CPP "${CMAKE_CURRENT_BINARY_DIR}/${METAL_TEST_KERNEL_SOURCE_CLASS}.cpp")
set(METAL_TEST_KERNELS_H "${CMAKE_CURRENT_BINARY_DIR}/${METAL_TEST_KERNEL_SOURCE_CLASS}.h")
file(GLOB METAL_TEST_KERNEL_FILES CONFIGURE_DEPENDS
    "${METAL_TEST_KERNEL_SOURCE_DIR}/kernels/*.metal")
add_custom_command(
    OUTPUT "${METAL_TEST_KERNELS_CPP}" "${METAL_TEST_KERNELS_H}"
    COMMAND "${CMAKE_COMMAND}"
        "-DKERNEL_SOURCE_DIR=${METAL_TEST_KERNEL_SOURCE_DIR}"
        "-DKERNELS_CPP=${METAL_TEST_KERNELS_CPP}"
        "-DKERNELS_H=${METAL_TEST_KERNELS_H}"
        "-DKERNEL_SOURCE_CLASS=${METAL_TEST_KERNEL_SOURCE_CLASS}"
        -DKERNEL_FILE_EXTENSION=metal
        -P "${CMAKE_SOURCE_DIR}/cmake_modules/EncodeKernelFiles.cmake"
    DEPENDS
        ${METAL_TEST_KERNEL_FILES}
        "${METAL_TEST_KERNEL_SOURCE_DIR}/${METAL_TEST_KERNEL_SOURCE_CLASS}.cpp.in"
        "${METAL_TEST_KERNEL_SOURCE_DIR}/${METAL_TEST_KERNEL_SOURCE_CLASS}.h.in"
        "${CMAKE_SOURCE_DIR}/cmake_modules/EncodeKernelFiles.cmake"
    VERBATIM)
set_source_files_properties("${METAL_TEST_KERNELS_CPP}" "${METAL_TEST_KERNELS_H}"
    PROPERTIES GENERATED TRUE)
add_custom_target(GenerateMetalTestKernelSources
    DEPENDS "${METAL_TEST_KERNELS_CPP}" "${METAL_TEST_KERNELS_H}")
include_directories(BEFORE "${CMAKE_CURRENT_BINARY_DIR}")

# The bit-exact oracle must exercise offline-compiled atomic helpers as well as
# runtime compilation.  Reuse the existing test entry points and production
# helpers in a separate library, embedded only in the ComputeContext tests.
if(OPENMM_METAL_USE_EMBEDDED_METALLIB)
    set(METAL_FIXED_POINT_TEST_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/fixed-point-kernels")
    set(METAL_FIXED_POINT_TEST_AIR "${METAL_FIXED_POINT_TEST_BUILD_DIR}/FixedPointTests.air")
    set(METAL_FIXED_POINT_TEST_METALLIB "${METAL_FIXED_POINT_TEST_BUILD_DIR}/FixedPointTests.metallib")
    set(METAL_FIXED_POINT_TEST_ASM "${CMAKE_CURRENT_BINARY_DIR}/EmbeddedFixedPointTests.S")
    add_custom_command(
        OUTPUT "${METAL_FIXED_POINT_TEST_AIR}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${METAL_FIXED_POINT_TEST_BUILD_DIR}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${METAL_FIXED_POINT_TEST_BUILD_DIR}/module-cache"
        COMMAND "${OPENMM_METAL_COMPILER}"
            -c -std=metal3.0 -ffast-math
            -mmacosx-version-min=13.0
            "-fmodules-cache-path=${METAL_FIXED_POINT_TEST_BUILD_DIR}/module-cache"
            -isysroot "${OPENMM_METAL_SDK}"
            "${CMAKE_CURRENT_SOURCE_DIR}/FixedPointTests.metal"
            -o "${METAL_FIXED_POINT_TEST_AIR}"
        DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/FixedPointTests.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/counterAtomics.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/fixedPointHelpers.metal"
            "${METAL_KERNEL_SOURCE_DIR}/kernels/counterAtomics.metal"
            "${METAL_KERNEL_SOURCE_DIR}/kernels/fixedPoint.metal"
        COMMENT "Compiling offline Metal fixed-point oracle kernels"
        VERBATIM)
    add_custom_command(
        OUTPUT "${METAL_FIXED_POINT_TEST_METALLIB}"
        COMMAND "${OPENMM_METALLIB_COMPILER}" "${METAL_FIXED_POINT_TEST_AIR}"
            -o "${METAL_FIXED_POINT_TEST_METALLIB}"
        DEPENDS "${METAL_FIXED_POINT_TEST_AIR}"
        COMMENT "Linking FixedPointTests.metallib"
        VERBATIM)
    add_custom_target(MetalFixedPointTestKernelLibrary
        DEPENDS "${METAL_FIXED_POINT_TEST_METALLIB}")
    configure_file(EmbeddedFixedPointTests.S.in "${METAL_FIXED_POINT_TEST_ASM}" @ONLY)
    set_source_files_properties("${METAL_FIXED_POINT_TEST_ASM}" PROPERTIES
        GENERATED TRUE
        OBJECT_DEPENDS "${METAL_FIXED_POINT_TEST_METALLIB}")
endif()

if(OPENMM_BUILD_SHARED_LIB)
    add_executable(TestMetalComputeContext TestMetalComputeContext.cpp
        "${METAL_TEST_KERNELS_CPP}" "${METAL_TEST_KERNELS_H}")
    add_dependencies(TestMetalComputeContext GenerateMetalTestKernelSources)
    if(OPENMM_METAL_USE_EMBEDDED_METALLIB)
        target_sources(TestMetalComputeContext PRIVATE "${METAL_FIXED_POINT_TEST_ASM}")
        add_dependencies(TestMetalComputeContext MetalFixedPointTestKernelLibrary)
        target_compile_definitions(TestMetalComputeContext PRIVATE
            OPENMM_METAL_TEST_USE_EMBEDDED_FIXED_POINT_METALLIB)
    endif()
    target_link_libraries(TestMetalComputeContext PRIVATE ${SHARED_TARGET})
    set_target_properties(TestMetalComputeContext PROPERTIES
        LINK_FLAGS "${EXTRA_LINK_FLAGS}"
        COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}"
        MACOSX_DEPLOYMENT_TARGET "13.0")
    add_test(NAME TestMetalComputeContext COMMAND TestMetalComputeContext)

    add_executable(TestMetalPlatform TestMetalPlatform.cpp)
    target_link_libraries(TestMetalPlatform PRIVATE ${SHARED_TARGET})
    set_target_properties(TestMetalPlatform PROPERTIES
        LINK_FLAGS "${EXTRA_LINK_FLAGS}"
        COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}"
        MACOSX_DEPLOYMENT_TARGET "13.0")
    add_test(NAME TestMetalPlatform COMMAND TestMetalPlatform)

    add_executable(TestMetalRuntime TestMetalRuntime.mm
        "${METAL_TEST_KERNELS_CPP}" "${METAL_TEST_KERNELS_H}")
    add_dependencies(TestMetalRuntime GenerateMetalTestKernelSources)
    target_link_libraries(TestMetalRuntime PRIVATE ${SHARED_TARGET} ${FOUNDATION_FRAMEWORK})
    target_compile_options(TestMetalRuntime PRIVATE -fobjc-arc)
    set_target_properties(TestMetalRuntime PROPERTIES
        LINK_FLAGS "${EXTRA_LINK_FLAGS}"
        COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}"
        MACOSX_DEPLOYMENT_TARGET "13.0")
    add_test(NAME TestMetalRuntime COMMAND TestMetalRuntime)

    add_executable(TestMetalVerticalSlice TestMetalVerticalSlice.cpp)
    target_link_libraries(TestMetalVerticalSlice PRIVATE ${SHARED_TARGET})
    set_target_properties(TestMetalVerticalSlice PROPERTIES
        LINK_FLAGS "${EXTRA_LINK_FLAGS}"
        COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}"
        MACOSX_DEPLOYMENT_TARGET "13.0")
    add_test(NAME TestMetalVerticalSlice COMMAND TestMetalVerticalSlice)

    add_executable(TestMetalNonbondedForce TestMetalNonbondedForce.cpp)
    target_link_libraries(TestMetalNonbondedForce PRIVATE ${SHARED_TARGET})
    set_target_properties(TestMetalNonbondedForce PROPERTIES
        LINK_FLAGS "${EXTRA_LINK_FLAGS}"
        COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}"
        MACOSX_DEPLOYMENT_TARGET "13.0")
    add_test(NAME TestMetalNonbondedForce COMMAND TestMetalNonbondedForce)
endif()

if(OPENMM_BUILD_STATIC_LIB)
    add_executable(TestMetalComputeContextStatic TestMetalComputeContext.cpp
        "${METAL_TEST_KERNELS_CPP}" "${METAL_TEST_KERNELS_H}")
    add_dependencies(TestMetalComputeContextStatic GenerateMetalTestKernelSources)
    if(OPENMM_METAL_USE_EMBEDDED_METALLIB)
        target_sources(TestMetalComputeContextStatic PRIVATE "${METAL_FIXED_POINT_TEST_ASM}")
        add_dependencies(TestMetalComputeContextStatic MetalFixedPointTestKernelLibrary)
        target_compile_definitions(TestMetalComputeContextStatic PRIVATE
            OPENMM_METAL_TEST_USE_EMBEDDED_FIXED_POINT_METALLIB)
    endif()
    target_link_libraries(TestMetalComputeContextStatic PRIVATE ${STATIC_TARGET})
    target_compile_definitions(TestMetalComputeContextStatic PRIVATE
        OPENMM_USE_STATIC_LIBRARIES
        OPENMM_METAL_USE_STATIC_LIBRARIES)
    set_target_properties(TestMetalComputeContextStatic PROPERTIES
        LINK_FLAGS "${EXTRA_LINK_FLAGS}"
        COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}"
        MACOSX_DEPLOYMENT_TARGET "13.0")
    add_test(NAME TestMetalComputeContextStatic COMMAND TestMetalComputeContextStatic)

    add_executable(TestMetalPlatformStatic TestMetalPlatform.cpp)
    target_link_libraries(TestMetalPlatformStatic PRIVATE ${STATIC_TARGET})
    target_compile_definitions(TestMetalPlatformStatic PRIVATE
        OPENMM_USE_STATIC_LIBRARIES
        OPENMM_METAL_USE_STATIC_LIBRARIES)
    set_target_properties(TestMetalPlatformStatic PROPERTIES
        LINK_FLAGS "${EXTRA_LINK_FLAGS}"
        COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}"
        MACOSX_DEPLOYMENT_TARGET "13.0")
    add_test(NAME TestMetalPlatformStatic COMMAND TestMetalPlatformStatic)

    add_executable(TestMetalRuntimeStatic TestMetalRuntime.mm
        "${METAL_TEST_KERNELS_CPP}" "${METAL_TEST_KERNELS_H}")
    add_dependencies(TestMetalRuntimeStatic GenerateMetalTestKernelSources)
    target_link_libraries(TestMetalRuntimeStatic PRIVATE ${STATIC_TARGET} ${FOUNDATION_FRAMEWORK})
    target_compile_definitions(TestMetalRuntimeStatic PRIVATE
        OPENMM_USE_STATIC_LIBRARIES
        OPENMM_METAL_USE_STATIC_LIBRARIES)
    target_compile_options(TestMetalRuntimeStatic PRIVATE -fobjc-arc)
    set_target_properties(TestMetalRuntimeStatic PROPERTIES
        LINK_FLAGS "${EXTRA_LINK_FLAGS}"
        COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}"
        MACOSX_DEPLOYMENT_TARGET "13.0")
    add_test(NAME TestMetalRuntimeStatic COMMAND TestMetalRuntimeStatic)

    add_executable(TestMetalVerticalSliceStatic TestMetalVerticalSlice.cpp)
    target_link_libraries(TestMetalVerticalSliceStatic PRIVATE ${STATIC_TARGET})
    target_compile_definitions(TestMetalVerticalSliceStatic PRIVATE
        OPENMM_USE_STATIC_LIBRARIES
        OPENMM_METAL_USE_STATIC_LIBRARIES)
    set_target_properties(TestMetalVerticalSliceStatic PROPERTIES
        LINK_FLAGS "${EXTRA_LINK_FLAGS}"
        COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}"
        MACOSX_DEPLOYMENT_TARGET "13.0")
    add_test(NAME TestMetalVerticalSliceStatic COMMAND TestMetalVerticalSliceStatic)

    add_executable(TestMetalNonbondedForceStatic TestMetalNonbondedForce.cpp)
    target_link_libraries(TestMetalNonbondedForceStatic PRIVATE ${STATIC_TARGET})
    target_compile_definitions(TestMetalNonbondedForceStatic PRIVATE
        OPENMM_USE_STATIC_LIBRARIES
        OPENMM_METAL_USE_STATIC_LIBRARIES)
    set_target_properties(TestMetalNonbondedForceStatic PROPERTIES
        LINK_FLAGS "${EXTRA_LINK_FLAGS}"
        COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}"
        MACOSX_DEPLOYMENT_TARGET "13.0")
    add_test(NAME TestMetalNonbondedForceStatic COMMAND TestMetalNonbondedForceStatic)
endif()
