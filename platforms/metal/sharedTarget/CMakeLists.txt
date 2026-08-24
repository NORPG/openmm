# Source-file properties are directory scoped.  Repeat the metallib dependency
# here so an incremental .metal edit forces this .incbin object to reassemble.
set_source_files_properties("${OPENMM_METAL_EMBEDDED_ASM}" PROPERTIES
    GENERATED TRUE
    OBJECT_DEPENDS "${OPENMM_METAL_METALLIB}")
# CMake's Swift Ninja rule does not emit a depfile for an imported C header and
# ignores OBJECT_DEPENDS on a Swift source.  A generated empty Swift input gives
# Ninja an explicit header -> Swift module edge without modifying source files.
set(OPENMM_METAL_SWIFT_BRIDGE_DEPENDENCY
    "${CMAKE_CURRENT_BINARY_DIR}/MetalSwiftBridgeDependency.swift")
add_custom_command(
    OUTPUT "${OPENMM_METAL_SWIFT_BRIDGE_DEPENDENCY}"
    COMMAND "${CMAKE_COMMAND}" -E touch "${OPENMM_METAL_SWIFT_BRIDGE_DEPENDENCY}"
    DEPENDS "${OPENMM_METAL_SWIFT_BRIDGE_HEADER}"
    COMMENT "Updating Swift bridge-header dependency"
    VERBATIM)
set_source_files_properties("${OPENMM_METAL_SWIFT_BRIDGE_DEPENDENCY}" PROPERTIES
    GENERATED TRUE)

add_library(${SHARED_TARGET} SHARED
    ${SOURCE_FILES}
    "${OPENMM_METAL_SWIFT_BRIDGE_DEPENDENCY}"
    ${SOURCE_INCLUDE_FILES}
    ${API_ABS_INCLUDE_FILES})
add_dependencies(${SHARED_TARGET} OpenMMMetalKernelLibrary)

target_link_libraries(${SHARED_TARGET}
    PUBLIC
        ${OPENMM_LIBRARY_NAME}
    PRIVATE
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK})
target_compile_options(${SHARED_TARGET} PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:SHELL:${EXTRA_COMPILE_FLAGS}>"
    "$<$<COMPILE_LANGUAGE:Swift>:-disable-bridging-pch>"
    "$<$<COMPILE_LANGUAGE:Swift>:-import-objc-header>"
    "$<$<COMPILE_LANGUAGE:Swift>:${OPENMM_METAL_SWIFT_BRIDGE_HEADER}>")
target_link_options(${SHARED_TARGET} PRIVATE
    "SHELL:-target ${OPENMM_METAL_SWIFT_TARGET}")
target_compile_definitions(${SHARED_TARGET} PRIVATE
    OPENMM_METAL_BUILDING_SHARED_LIBRARY)
set_target_properties(${SHARED_TARGET} PROPERTIES
    MACOSX_DEPLOYMENT_TARGET "13.0"
    Swift_LANGUAGE_VERSION 6
    Swift_MODULE_NAME OpenMMMetalSwift)

install(TARGETS ${SHARED_TARGET}
    RUNTIME DESTINATION lib/plugins
    LIBRARY DESTINATION lib/plugins
    ARCHIVE DESTINATION lib/plugins)
