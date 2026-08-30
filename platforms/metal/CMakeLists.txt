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

# Metal 3 and the binding-reflection API used by this backend require macOS
# 13.  Keep the override local to this plugin and set it before compiler probes
# so it applies consistently to Objective-C++, assembly, libraries, and tests.
set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0")

# Objective-C++ is confined to this plugin.  Public headers remain ordinary
# C++, so clients of OpenMMMetal do not need to enable Objective-C++.
enable_language(OBJCXX)

find_library(METAL_FRAMEWORK Metal)
find_library(FOUNDATION_FRAMEWORK Foundation)
if(NOT METAL_FRAMEWORK OR NOT FOUNDATION_FRAMEWORK)
    message(FATAL_ERROR "The OpenMM Metal platform requires the macOS Metal and Foundation frameworks")
endif()
mark_as_advanced(METAL_FRAMEWORK FOUNDATION_FRAMEWORK)

set(OPENMM_BUILD_METAL_TESTS TRUE CACHE BOOL "Whether to build Metal platform test cases")
mark_as_advanced(OPENMM_BUILD_METAL_TESTS)

# Prefer production kernels compiled by Xcode's standalone Metal toolchain,
# while retaining runtime source compilation for systems that only have the
# SDK/Command Line Tools installed.  ON makes the toolchain mandatory; OFF is
# an explicit compatibility fallback.
set(OPENMM_METAL_KERNEL_COMPILATION "AUTO" CACHE STRING
    "How to build production Metal kernels: AUTO, ON (offline metallib), or OFF (runtime MSL)")
set_property(CACHE OPENMM_METAL_KERNEL_COMPILATION PROPERTY STRINGS AUTO ON OFF)
string(TOUPPER "${OPENMM_METAL_KERNEL_COMPILATION}" OPENMM_METAL_KERNEL_COMPILATION)
if(NOT OPENMM_METAL_KERNEL_COMPILATION MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR
        "OPENMM_METAL_KERNEL_COMPILATION must be AUTO, ON, or OFF")
endif()

set(OPENMM_METAL_USE_EMBEDDED_METALLIB FALSE)
if(NOT OPENMM_METAL_KERNEL_COMPILATION STREQUAL "OFF")
    execute_process(
        COMMAND /usr/bin/xcrun --find metallib
        RESULT_VARIABLE OPENMM_METALLIB_FIND_STATUS
        OUTPUT_VARIABLE OPENMM_METALLIB_COMPILER
        ERROR_VARIABLE OPENMM_METALLIB_FIND_ERROR
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(OPENMM_METALLIB_FIND_STATUS EQUAL 0 AND EXISTS "${OPENMM_METALLIB_COMPILER}")
        get_filename_component(OPENMM_METAL_TOOLCHAIN_BIN "${OPENMM_METALLIB_COMPILER}" DIRECTORY)
        set(OPENMM_METAL_COMPILER "${OPENMM_METAL_TOOLCHAIN_BIN}/metal")
    endif()
    execute_process(
        COMMAND /usr/bin/xcrun --sdk macosx --show-sdk-path
        RESULT_VARIABLE OPENMM_METAL_SDK_STATUS
        OUTPUT_VARIABLE OPENMM_METAL_SDK
        ERROR_VARIABLE OPENMM_METAL_SDK_ERROR
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(OPENMM_METALLIB_FIND_STATUS EQUAL 0 AND
            EXISTS "${OPENMM_METALLIB_COMPILER}" AND
            EXISTS "${OPENMM_METAL_COMPILER}" AND
            OPENMM_METAL_SDK_STATUS EQUAL 0 AND
            IS_DIRECTORY "${OPENMM_METAL_SDK}")
        set(OPENMM_METAL_USE_EMBEDDED_METALLIB TRUE)
    elseif(OPENMM_METAL_KERNEL_COMPILATION STREQUAL "ON")
        message(FATAL_ERROR
            "Offline Metal kernel compilation was requested, but Xcode's metal/metallib "
            "toolchain or the macOS SDK could not be found. Install the Metal Toolchain "
            "component in Xcode. metallib: ${OPENMM_METALLIB_FIND_ERROR}; "
            "SDK: ${OPENMM_METAL_SDK_ERROR}")
    else()
        message(STATUS
            "Xcode's offline Metal toolchain is unavailable; production kernels will be "
            "compiled from embedded MSL at runtime")
    endif()
endif()
mark_as_advanced(OPENMM_METAL_COMPILER OPENMM_METALLIB_COMPILER OPENMM_METAL_SDK)

set(OPENMMMETAL_LIBRARY_NAME OpenMMMetal)
set(SHARED_TARGET ${OPENMMMETAL_LIBRARY_NAME})
set(STATIC_TARGET ${OPENMMMETAL_LIBRARY_NAME}_static)

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
        ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.mm)
    file(GLOB incl_files ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.h)
    list(APPEND SOURCE_FILES ${src_files})
    list(APPEND SOURCE_INCLUDE_FILES ${incl_files})
    include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/include)
endforeach()
include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/../common/include)

# MetalContext implements the ComputeContext interface, but the initial Metal
# backend deliberately does not compile the full Common kernel stack.  These
# three files are the minimal out-of-line implementation closure for the base
# context, ComputeArray wrappers, and force metadata.
set(METAL_COMMON_CONTEXT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/src/ComputeArray.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/src/ComputeContext.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/src/ComputeForceInfo.cpp)
list(APPEND SOURCE_FILES ${METAL_COMMON_CONTEXT_SOURCES})

# Keep hand-written MSL in standalone .metal files.  With the full Xcode Metal
# toolchain they are compiled offline and embedded as one metallib.  Otherwise
# they are encoded into C++ and compiled at runtime.  Both products are fully
# self-contained and never depend on locating the source tree at runtime.
set(METAL_KERNEL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
file(GLOB METAL_KERNEL_FILES CONFIGURE_DEPENDS
    "${METAL_KERNEL_SOURCE_DIR}/kernels/*.metal")
if(OPENMM_METAL_USE_EMBEDDED_METALLIB)
    enable_language(ASM)
    set(OPENMM_METAL_KERNEL_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/kernels")
    set(OPENMM_METAL_MODULE_CACHE "${CMAKE_CURRENT_BINARY_DIR}/metal-module-cache")
    set(OPENMM_METAL_AIR_FILES)
    foreach(kernel_source IN LISTS METAL_KERNEL_FILES)
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
    list(APPEND SOURCE_FILES "${OPENMM_METAL_EMBEDDED_ASM}")
else()
    set(METAL_KERNEL_SOURCE_CLASS MetalKernelSources)
    set(METAL_KERNELS_CPP "${CMAKE_CURRENT_BINARY_DIR}/src/${METAL_KERNEL_SOURCE_CLASS}.cpp")
    set(METAL_KERNELS_H "${CMAKE_CURRENT_BINARY_DIR}/src/${METAL_KERNEL_SOURCE_CLASS}.h")
    add_custom_command(
        OUTPUT "${METAL_KERNELS_CPP}" "${METAL_KERNELS_H}"
        COMMAND "${CMAKE_COMMAND}"
            "-DKERNEL_SOURCE_DIR=${METAL_KERNEL_SOURCE_DIR}"
            "-DKERNELS_CPP=${METAL_KERNELS_CPP}"
            "-DKERNELS_H=${METAL_KERNELS_H}"
            "-DKERNEL_SOURCE_CLASS=${METAL_KERNEL_SOURCE_CLASS}"
            -DKERNEL_FILE_EXTENSION=metal
            -P "${CMAKE_SOURCE_DIR}/cmake_modules/EncodeKernelFiles.cmake"
        DEPENDS
            ${METAL_KERNEL_FILES}
            "${METAL_KERNEL_SOURCE_DIR}/${METAL_KERNEL_SOURCE_CLASS}.cpp.in"
            "${METAL_KERNEL_SOURCE_DIR}/${METAL_KERNEL_SOURCE_CLASS}.h.in"
            "${CMAKE_SOURCE_DIR}/cmake_modules/EncodeKernelFiles.cmake"
        VERBATIM)
    set_source_files_properties("${METAL_KERNELS_CPP}" "${METAL_KERNELS_H}"
        PROPERTIES GENERATED TRUE)
    add_custom_target(GenerateMetalKernelSources
        DEPENDS "${METAL_KERNELS_CPP}" "${METAL_KERNELS_H}")
    list(APPEND SOURCE_FILES "${METAL_KERNELS_CPP}")
    list(APPEND SOURCE_INCLUDE_FILES "${METAL_KERNELS_H}")
    include_directories(BEFORE "${CMAKE_CURRENT_BINARY_DIR}/src")
endif()

file(GLOB CORE_HEADERS include/*.h)
install(FILES ${CORE_HEADERS} DESTINATION include/openmm/metal)
install(FILES ${METAL_KERNEL_FILES} DESTINATION share/openmm/metal/kernels)

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
