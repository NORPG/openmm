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

add_library(${STATIC_TARGET} STATIC
    ${SOURCE_FILES}
    "${OPENMM_METAL_SWIFT_BRIDGE_DEPENDENCY}"
    ${SOURCE_INCLUDE_FILES}
    ${API_ABS_INCLUDE_FILES})
add_dependencies(${STATIC_TARGET} OpenMMMetalKernelLibrary)

target_link_libraries(${STATIC_TARGET}
    PUBLIC
        ${OPENMM_LIBRARY_NAME}_static
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK})
target_compile_options(${STATIC_TARGET} PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:SHELL:${EXTRA_COMPILE_FLAGS}>"
    "$<$<COMPILE_LANGUAGE:Swift>:-disable-bridging-pch>"
    "$<$<COMPILE_LANGUAGE:Swift>:-import-objc-header>"
    "$<$<COMPILE_LANGUAGE:Swift>:${OPENMM_METAL_SWIFT_BRIDGE_HEADER}>")
# A static archive has no link step of its own.  Propagate the deployment
# target so either Clang or the Swift driver can consume its autolink records
# when a downstream executable links OpenMMMetal_static.
target_link_options(${STATIC_TARGET} INTERFACE
    "SHELL:-target ${OPENMM_METAL_SWIFT_TARGET}")
target_compile_definitions(${STATIC_TARGET} PRIVATE
    OPENMM_METAL_BUILDING_STATIC_LIBRARY)
set_target_properties(${STATIC_TARGET} PROPERTIES
    MACOSX_DEPLOYMENT_TARGET "13.0"
    Swift_LANGUAGE_VERSION 6
    Swift_MODULE_NAME OpenMMMetalStaticSwift)

install(TARGETS ${STATIC_TARGET}
    RUNTIME DESTINATION lib/plugins
    LIBRARY DESTINATION lib/plugins
    ARCHIVE DESTINATION lib/plugins)
