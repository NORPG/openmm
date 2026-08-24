#---------------------------------------------------
# OpenMM Metal Platform
#
# Creates the experimental native Metal plugin.
#---------------------------------------------------

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    message(FATAL_ERROR "The OpenMM Metal platform is only available on macOS")
endif()
if(NOT ARM)
    message(FATAL_ERROR "The OpenMM Metal platform currently supports Apple Silicon only")
endif()
if(NOT CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
    message(FATAL_ERROR
        "The OpenMM Metal platform currently requires CMAKE_OSX_ARCHITECTURES=arm64; "
        "Swift does not support this target as a universal binary")
endif()
if(CMAKE_VERSION VERSION_LESS 3.29)
    message(FATAL_ERROR "The Swift-based OpenMM Metal platform requires CMake 3.29 or newer")
endif()

# Root OpenMM definitions include value-bearing C/C++ macros.  CMake otherwise
# forwards them as Swift conditional-compilation flags, which is both noisy and
# incompatible with Swift -warnings-as-errors.  Preserve every inherited value
# for C++ sources and omit them only from Swift/ASM compilation.
get_directory_property(OPENMM_METAL_INHERITED_DEFINITIONS COMPILE_DEFINITIONS)
set(OPENMM_METAL_LANGUAGE_SCOPED_DEFINITIONS)
foreach(definition IN LISTS OPENMM_METAL_INHERITED_DEFINITIONS)
    list(APPEND OPENMM_METAL_LANGUAGE_SCOPED_DEFINITIONS
        "$<$<COMPILE_LANGUAGE:CXX>:${definition}>")
endforeach()
set_property(DIRECTORY PROPERTY COMPILE_DEFINITIONS
    ${OPENMM_METAL_LANGUAGE_SCOPED_DEFINITIONS})

# CMake 3.29 teaches generators how to build Swift in an add_subdirectory().
# Keep the newer policy local to this optional plugin so it does not alter the
# established build of any other OpenMM platform.
cmake_policy(SET CMP0157 NEW)

# Swift cannot target the 10.7 deployment baseline used by the rest of OpenMM.
# This normal (non-cache) variable is scoped to platforms/metal and below.
set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0")
set(OPENMM_METAL_SWIFT_TARGET "arm64-apple-macosx13.0")
set(OPENMM_METAL_SWIFT_MODULE_CACHE "${CMAKE_CURRENT_BINARY_DIR}/swift-module-cache")
file(MAKE_DIRECTORY "${OPENMM_METAL_SWIFT_MODULE_CACHE}")
# These flags must be present during CMake's compiler probe, before there is a
# target on which target_compile_options() could be set.  The explicit cache
# also makes the build independent of a writable per-user Clang cache.
string(APPEND CMAKE_Swift_FLAGS
    " -target ${OPENMM_METAL_SWIFT_TARGET} -module-cache-path \"${OPENMM_METAL_SWIFT_MODULE_CACHE}\"")
enable_language(Swift)
enable_language(ASM)
if(CMAKE_Swift_COMPILER_VERSION VERSION_LESS 6.3)
    message(FATAL_ERROR "The OpenMM Metal platform requires Swift 6.3 or newer")
endif()

find_library(METAL_FRAMEWORK Metal)
find_library(FOUNDATION_FRAMEWORK Foundation)
if(NOT METAL_FRAMEWORK OR NOT FOUNDATION_FRAMEWORK)
    message(FATAL_ERROR "The OpenMM Metal platform requires the macOS Metal and Foundation frameworks")
endif()
mark_as_advanced(METAL_FRAMEWORK FOUNDATION_FRAMEWORK)

# Use Xcode's standalone Metal compiler for production kernels.  Newer Xcode
# releases can install it in a versioned Metal toolchain rather than /usr/bin,
# so locate metallib with xcrun and derive its sibling `metal` executable.
execute_process(
    COMMAND /usr/bin/xcrun --find metallib
    RESULT_VARIABLE OPENMM_METALLIB_FIND_STATUS
    OUTPUT_VARIABLE OPENMM_METALLIB_COMPILER
    ERROR_VARIABLE OPENMM_METALLIB_FIND_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT OPENMM_METALLIB_FIND_STATUS EQUAL 0 OR NOT EXISTS "${OPENMM_METALLIB_COMPILER}")
    message(FATAL_ERROR
        "Could not find Xcode's metallib compiler. Install the Metal Toolchain "
        "component in Xcode. xcrun reported: ${OPENMM_METALLIB_FIND_ERROR}")
endif()
get_filename_component(OPENMM_METAL_TOOLCHAIN_BIN "${OPENMM_METALLIB_COMPILER}" DIRECTORY)
set(OPENMM_METAL_COMPILER "${OPENMM_METAL_TOOLCHAIN_BIN}/metal")
if(NOT EXISTS "${OPENMM_METAL_COMPILER}")
    message(FATAL_ERROR "Could not find the metal compiler beside ${OPENMM_METALLIB_COMPILER}")
endif()

execute_process(
    COMMAND /usr/bin/xcrun --sdk macosx --show-sdk-path
    RESULT_VARIABLE OPENMM_METAL_SDK_STATUS
    OUTPUT_VARIABLE OPENMM_METAL_SDK
    ERROR_VARIABLE OPENMM_METAL_SDK_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT OPENMM_METAL_SDK_STATUS EQUAL 0 OR NOT IS_DIRECTORY "${OPENMM_METAL_SDK}")
    message(FATAL_ERROR "Could not locate the macOS SDK: ${OPENMM_METAL_SDK_ERROR}")
endif()
mark_as_advanced(OPENMM_METAL_COMPILER OPENMM_METALLIB_COMPILER OPENMM_METAL_SDK)

set(OPENMM_BUILD_METAL_TESTS TRUE CACHE BOOL "Whether to build Metal platform test cases")
mark_as_advanced(OPENMM_BUILD_METAL_TESTS)

set(OPENMMMETAL_LIBRARY_NAME OpenMMMetal)
set(SHARED_TARGET ${OPENMMMETAL_LIBRARY_NAME})
set(STATIC_TARGET ${OPENMMMETAL_LIBRARY_NAME}_static)
set(OPENMM_METAL_SWIFT_BRIDGE_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/src/MetalSwiftBridge.h")

# Compile every production MSL file offline, combine them into one metallib,
# then embed that binary in both shared and static libraries.  Runtime source
# compilation remains available through MetalProgram for generated kernels and
# is covered by a separate test-only .metal source.
set(OPENMM_METAL_KERNEL_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/HarmonicBond.metal"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/Verlet.metal")
set(OPENMM_METAL_KERNEL_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/kernels")
set(OPENMM_METAL_MODULE_CACHE "${CMAKE_CURRENT_BINARY_DIR}/metal-module-cache")
set(OPENMM_METAL_AIR_FILES)
foreach(kernel_source IN LISTS OPENMM_METAL_KERNEL_SOURCES)
    get_filename_component(kernel_name "${kernel_source}" NAME_WE)
    set(kernel_air "${OPENMM_METAL_KERNEL_BUILD_DIR}/${kernel_name}.air")
    add_custom_command(
        OUTPUT "${kernel_air}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${OPENMM_METAL_KERNEL_BUILD_DIR}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${OPENMM_METAL_MODULE_CACHE}"
        COMMAND "${OPENMM_METAL_COMPILER}"
                -c -std=metal3.0 -ffast-math
                -mmacosx-version-min=13.0
                "-fmodules-cache-path=${OPENMM_METAL_MODULE_CACHE}"
                -isysroot "${OPENMM_METAL_SDK}"
                "${kernel_source}" -o "${kernel_air}"
        DEPENDS "${kernel_source}"
        COMMENT "Compiling ${kernel_name}.metal"
        VERBATIM)
    list(APPEND OPENMM_METAL_AIR_FILES "${kernel_air}")
endforeach()

set(OPENMM_METAL_METALLIB "${OPENMM_METAL_KERNEL_BUILD_DIR}/OpenMMMetalKernels.metallib")
add_custom_command(
    OUTPUT "${OPENMM_METAL_METALLIB}"
    COMMAND "${OPENMM_METALLIB_COMPILER}" ${OPENMM_METAL_AIR_FILES}
            -o "${OPENMM_METAL_METALLIB}"
    DEPENDS ${OPENMM_METAL_AIR_FILES}
    COMMENT "Linking OpenMMMetalKernels.metallib"
    VERBATIM)
add_custom_target(OpenMMMetalKernelLibrary
    DEPENDS "${OPENMM_METAL_METALLIB}")

set(OPENMM_METAL_METALLIB_PATH "${OPENMM_METAL_METALLIB}")
set(OPENMM_METAL_EMBEDDED_ASM "${CMAKE_CURRENT_BINARY_DIR}/EmbeddedMetallib.S")
configure_file(src/EmbeddedMetallib.S.in "${OPENMM_METAL_EMBEDDED_ASM}" @ONLY)
set_source_files_properties("${OPENMM_METAL_EMBEDDED_ASM}" PROPERTIES
    GENERATED TRUE
    OBJECT_DEPENDS "${OPENMM_METAL_METALLIB}")

set(OPENMM_SOURCE_SUBDIRS .)
set(API_INCLUDE_DIRS)
foreach(subdir ${OPENMM_SOURCE_SUBDIRS})
    list(APPEND API_INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/include/internal)
endforeach()

set(API_ABS_INCLUDE_FILES)
foreach(dir ${API_INCLUDE_DIRS})
    file(GLOB fullpaths ${dir}/*.h)
    list(APPEND API_ABS_INCLUDE_FILES ${fullpaths})
endforeach()

set(SOURCE_FILES)
set(SOURCE_INCLUDE_FILES)
foreach(subdir ${OPENMM_SOURCE_SUBDIRS})
    file(GLOB_RECURSE src_files
        ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.swift)
    file(GLOB incl_files ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.h)
    list(APPEND SOURCE_FILES ${src_files})
    list(APPEND SOURCE_INCLUDE_FILES ${incl_files})
    include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/include)
endforeach()
list(APPEND SOURCE_FILES "${OPENMM_METAL_EMBEDDED_ASM}")
include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/../common/include)

file(GLOB CORE_HEADERS include/*.h)
install(FILES ${CORE_HEADERS} DESTINATION include/openmm/metal)
install(FILES ${OPENMM_METAL_KERNEL_SOURCES} DESTINATION share/openmm/metal/kernels)

# The public runtime wrappers derive from OpenMM's lightweight Common compute
# interfaces.  Metal can be built without the Common library, so make sure the
# headers needed by installed Metal headers are still present in that build.
set(METAL_COMMON_INTERFACE_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/include/openmm/common/ArrayInterface.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/include/openmm/common/ComputeEvent.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/include/openmm/common/ComputeKernel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/include/openmm/common/ComputeProgram.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/include/openmm/common/ComputeQueue.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/include/openmm/common/windowsExportCommon.h)
install(FILES ${METAL_COMMON_INTERFACE_HEADERS} DESTINATION include/openmm/common)

if(OPENMM_BUILD_SHARED_LIB)
    add_subdirectory(sharedTarget)
endif()
if(OPENMM_BUILD_STATIC_LIB)
    add_subdirectory(staticTarget)
endif()

if(BUILD_TESTING AND OPENMM_BUILD_METAL_TESTS)
    add_subdirectory(tests)
endif()
