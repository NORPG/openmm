if(OPENMM_BUILD_SHARED_LIB)
    add_executable(TestMetalPlatform TestMetalPlatform.cpp)
    target_link_libraries(TestMetalPlatform PRIVATE ${SHARED_TARGET})
    set_target_properties(TestMetalPlatform PROPERTIES
        COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}"
        MACOSX_DEPLOYMENT_TARGET "13.0")
    add_test(NAME TestMetalPlatform COMMAND TestMetalPlatform)

    add_executable(TestMetalRuntime TestMetalRuntime.cpp)
    target_link_libraries(TestMetalRuntime PRIVATE ${SHARED_TARGET})
    target_compile_definitions(TestMetalRuntime PRIVATE
        OPENMM_METAL_RUNTIME_TEST_SOURCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/kernels/RuntimeTestKernels.metal")
    set_target_properties(TestMetalRuntime PROPERTIES
        COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}"
        MACOSX_DEPLOYMENT_TARGET "13.0")
    add_test(NAME TestMetalRuntime COMMAND TestMetalRuntime)

    add_executable(TestMetalVerticalSlice TestMetalVerticalSlice.cpp)
    target_link_libraries(TestMetalVerticalSlice PRIVATE ${SHARED_TARGET})
    set_target_properties(TestMetalVerticalSlice PROPERTIES
        COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}"
        MACOSX_DEPLOYMENT_TARGET "13.0")
    add_test(NAME TestMetalVerticalSlice COMMAND TestMetalVerticalSlice)

    add_executable(TestMetalNonbondedForce TestMetalNonbondedForce.cpp)
    target_link_libraries(TestMetalNonbondedForce PRIVATE ${SHARED_TARGET})
    set_target_properties(TestMetalNonbondedForce PROPERTIES
        COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}"
        MACOSX_DEPLOYMENT_TARGET "13.0")
    add_test(NAME TestMetalNonbondedForce COMMAND TestMetalNonbondedForce)
endif()

if(OPENMM_BUILD_STATIC_LIB)
    add_executable(TestMetalPlatformStatic TestMetalPlatform.cpp)
    target_link_libraries(TestMetalPlatformStatic PRIVATE ${STATIC_TARGET})
    target_compile_definitions(TestMetalPlatformStatic PRIVATE
        OPENMM_USE_STATIC_LIBRARIES
        OPENMM_METAL_USE_STATIC_LIBRARIES)
    set_target_properties(TestMetalPlatformStatic PROPERTIES
        COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}"
        MACOSX_DEPLOYMENT_TARGET "13.0")
    add_test(NAME TestMetalPlatformStatic COMMAND TestMetalPlatformStatic)

    add_executable(TestMetalRuntimeStatic TestMetalRuntime.cpp)
    target_link_libraries(TestMetalRuntimeStatic PRIVATE ${STATIC_TARGET})
    target_compile_definitions(TestMetalRuntimeStatic PRIVATE
        OPENMM_USE_STATIC_LIBRARIES
        OPENMM_METAL_USE_STATIC_LIBRARIES
        OPENMM_METAL_RUNTIME_TEST_SOURCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/kernels/RuntimeTestKernels.metal")
    set_target_properties(TestMetalRuntimeStatic PROPERTIES
        COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}"
        MACOSX_DEPLOYMENT_TARGET "13.0")
    add_test(NAME TestMetalRuntimeStatic COMMAND TestMetalRuntimeStatic)

    add_executable(TestMetalVerticalSliceStatic TestMetalVerticalSlice.cpp)
    target_link_libraries(TestMetalVerticalSliceStatic PRIVATE ${STATIC_TARGET})
    target_compile_definitions(TestMetalVerticalSliceStatic PRIVATE
        OPENMM_USE_STATIC_LIBRARIES
        OPENMM_METAL_USE_STATIC_LIBRARIES)
    set_target_properties(TestMetalVerticalSliceStatic PROPERTIES
        COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}"
        MACOSX_DEPLOYMENT_TARGET "13.0")
    add_test(NAME TestMetalVerticalSliceStatic COMMAND TestMetalVerticalSliceStatic)

    add_executable(TestMetalNonbondedForceStatic TestMetalNonbondedForce.cpp)
    target_link_libraries(TestMetalNonbondedForceStatic PRIVATE ${STATIC_TARGET})
    target_compile_definitions(TestMetalNonbondedForceStatic PRIVATE
        OPENMM_USE_STATIC_LIBRARIES
        OPENMM_METAL_USE_STATIC_LIBRARIES)
    set_target_properties(TestMetalNonbondedForceStatic PROPERTIES
        COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}"
        MACOSX_DEPLOYMENT_TARGET "13.0")
    add_test(NAME TestMetalNonbondedForceStatic COMMAND TestMetalNonbondedForceStatic)
endif()
