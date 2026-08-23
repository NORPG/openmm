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

file(GLOB CORE_HEADERS include/*.h)
install(FILES ${CORE_HEADERS} DESTINATION include/openmm/metal)

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
