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

if(OPENMM_BUILD_SHARED_LIB)
    add_executable(TestMetalComputeContext TestMetalComputeContext.cpp)
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
endif()

if(OPENMM_BUILD_STATIC_LIB)
    add_executable(TestMetalComputeContextStatic TestMetalComputeContext.cpp)
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
endif()
