if(OPENMM_BUILD_SHARED_LIB)
    add_executable(TestMetalPlatform TestMetalPlatform.cpp)
    target_link_libraries(TestMetalPlatform PRIVATE ${SHARED_TARGET})
    set_target_properties(TestMetalPlatform PROPERTIES
        LINK_FLAGS "${EXTRA_LINK_FLAGS}"
        COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}"
        MACOSX_DEPLOYMENT_TARGET "13.0")
    add_test(NAME TestMetalPlatform COMMAND TestMetalPlatform)

    add_executable(TestMetalRuntime TestMetalRuntime.mm)
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

    add_executable(TestMetalRuntimeStatic TestMetalRuntime.mm)
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
