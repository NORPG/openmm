if(OPENMM_METAL_USE_EMBEDDED_METALLIB)
    # Source properties are directory scoped.  Repeat the dependency so an
    # incremental .metal edit forces the .incbin object to be reassembled.
    set_source_files_properties("${OPENMM_METAL_EMBEDDED_ASM}" PROPERTIES
        GENERATED TRUE
        OBJECT_DEPENDS "${OPENMM_METAL_METALLIB}")
else()
    set_source_files_properties(${METAL_KERNELS_CPP} ${METAL_KERNELS_H}
        PROPERTIES GENERATED TRUE)
endif()
add_library(${STATIC_TARGET} STATIC
    ${SOURCE_FILES}
    ${SOURCE_INCLUDE_FILES}
    ${API_ABS_INCLUDE_FILES})
if(OPENMM_METAL_USE_EMBEDDED_METALLIB)
    add_dependencies(${STATIC_TARGET} OpenMMMetalKernelLibrary)
else()
    add_dependencies(${STATIC_TARGET} GenerateMetalKernelSources)
endif()

target_link_libraries(${STATIC_TARGET}
    PUBLIC
        ${OPENMM_LIBRARY_NAME}_static
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK})
target_compile_options(${STATIC_TARGET} PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX,OBJCXX>:SHELL:${EXTRA_COMPILE_FLAGS}>"
    "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
target_compile_definitions(${STATIC_TARGET} PRIVATE
    OPENMM_METAL_BUILDING_STATIC_LIBRARY)
if(OPENMM_METAL_USE_EMBEDDED_METALLIB)
    target_compile_definitions(${STATIC_TARGET} PRIVATE
        OPENMM_METAL_USE_EMBEDDED_METALLIB)
endif()
set_target_properties(${STATIC_TARGET} PROPERTIES
    LINK_FLAGS "${EXTRA_LINK_FLAGS}")

install(TARGETS ${STATIC_TARGET}
    RUNTIME DESTINATION lib/plugins
    LIBRARY DESTINATION lib/plugins
    ARCHIVE DESTINATION lib/plugins)
