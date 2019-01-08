# CMake 3.2 is the minimum version where the CUDA separable
# compilation issue was fixed:
# http://public.kitware.com/pipermail/cmake/2015-January/059482.html
#
# CMake 3.7.2 changes the treatment of host/device compilation flags
# https://gitlab.kitware.com/cmake/cmake/issues/16411
#
# Cmake 3.10.0 fixes incorrect handling of CUDA_USE_STATIC_CUDA_RUNTIME
# https://gitlab.kitware.com/cmake/cmake/merge_requests/1123
#
cmake_minimum_required(VERSION 3.10.0)

project(gpurt)

set(PROJECT_VERSION_MAJOR 0)
set(PROJECT_VERSION_MINOR 1)
set(PROJECT_VERSION_PATCH 0)

set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE INTERNAL "")
find_package(CUDA QUIET REQUIRED)
find_cuda_helper_libs(cuda)

# Search for OpenCL in the same paths where CUDA is located.
set(OpenCL_INCLUDE_DIR "${CUDA_TOOLKIT_INCLUDE}")
set(ENV{CUDA_PATH} "${CUDA_TOOLKIT_ROOT_DIR}")
find_package(OpenCL QUIET REQUIRED)

set(CUDA_SEPARABLE_COMPILATION ON)

# Manual build type selection (for debugging purposes)
#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)

string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_UPPER)

if (CMAKE_BUILD_TYPE_UPPER STREQUAL "RELEASE")
set(CUDA_NVCC_FLAGS_RELEASE "-DNDEBUG;-O3;--use_fast_math;-lineinfo;${CUDA_NVCC_FLAGS_RELEASE}")
if ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffast-math -march=native")
endif()
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -march=native")
endif()
elseif (CMAKE_BUILD_TYPE_UPPER STREQUAL "DEBUG")
set(CUDA_NVCC_FLAGS_DEBUG "-G;-g;-O0;${CUDA_NVCC_FLAGS_DEBUG}")
endif (CMAKE_BUILD_TYPE_UPPER STREQUAL "RELEASE")

set(CUDA_NVCC_FLAGS "-gencode=arch=compute_60,code=sm_60;${CUDA_NVCC_FLAGS}")
set(CUDA_NVCC_FLAGS "-gencode=arch=compute_61,code=sm_61;${CUDA_NVCC_FLAGS}")
set(CUDA_NVCC_FLAGS "-gencode=arch=compute_70,code=sm_70;${CUDA_NVCC_FLAGS}")
set(CUDA_NVCC_FLAGS "--expt-relaxed-constexpr;${CUDA_NVCC_FLAGS}")
set(CUDA_NVCC_FLAGS "--maxrregcount=32;${CUDA_NVCC_FLAGS}")
set(CUDA_NVCC_FLAGS "-keep;${CUDA_NVCC_FLAGS}")
set(CUDA_NVCC_FLAGS "-Xptxas=-v;${CUDA_NVCC_FLAGS}")

# Enable C++ 11 support
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA is unfortunately ignoring C++11 settings above, except on Windows
if (NOT WIN32)
set(CUDA_NVCC_FLAGS "-std=c++11;${CUDA_NVCC_FLAGS}")
endif()

# Search path for CMake include files.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

file(GLOB_RECURSE LIBRARY_SRC "src/*.cpp" "src/*.cu" "src/*.c")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${OpenCL_INCLUDE_DIR})

add_executable(${PROJECT_NAME}_xxd ${CMAKE_CURRENT_SOURCE_DIR}/xxd/xxd.c)

# Embed OpenCL sources and add runtime compilation hooks.
file(GLOB CL_SRC "src/*.cl")
set(CL_EMBED ${PROJECT_NAME}_xxd)
foreach(CL_FILE ${CL_SRC})
	cuda_compute_build_path("${CL_FILE}" CL_BUILD_PATH)
    set(CL_INTERMEDIATE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${PROJECT_NAME}.dir/${CL_BUILD_PATH}") 
	# Translate OpenCL source into comma-separated byte codes.
	get_filename_component(CL_FILE_BASE ${CL_FILE} NAME)
	set(CL_HEX_FILE "${CL_INTERMEDIATE_DIRECTORY}/${CL_FILE_BASE}.hex")
    add_custom_command(
        OUTPUT ${CL_HEX_FILE}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CL_INTERMEDIATE_DIRECTORY}"
        COMMAND ${CL_EMBED} -i < ${CL_FILE} > ${CL_HEX_FILE}
        COMMENT "Generating hex representation for OpenCL source file ${CL_FILE}"
        DEPENDS ${CL_FILE} ${PROJECT_NAME}_xxd)
	set_source_files_properties("${CL_HEX_FILE}" PROPERTIES GENERATED TRUE) 
	set(CL_EMBED_FILE "${CL_INTERMEDIATE_DIRECTORY}/${CL_FILE_BASE}.cpp")
	add_custom_command(
		OUTPUT ${CL_EMBED_FILE}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CL_INTERMEDIATE_DIRECTORY}"
		COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR} -DCL_HEX_FILE=${CL_HEX_FILE} -DCL_EMBED_FILE=${CL_EMBED_FILE} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/GenerateOpenCL.cmake
		COMMENT "Embedding OpenCL source file ${CL_FILE}"
		DEPENDS ${CL_HEX_FILE} "${CMAKE_CURRENT_SOURCE_DIR}/src/OpenCL.cpp.in")
	set_source_files_properties("${CL_EMBED_FILE}" PROPERTIES GENERATED TRUE) 
	# Submit the resulting source file for compilation
    LIST(APPEND LIBRARY_SRC ${CL_EMBED_FILE})
endforeach()

if (WIN32)
cuda_add_library(${PROJECT_NAME} SHARED ${LIBRARY_SRC})
else()
cuda_add_library(${PROJECT_NAME} SHARED ${LIBRARY_SRC} OPTIONS --compiler-options "-fPIC")
endif()
target_link_libraries(${PROJECT_NAME} ${CUDA_cuda_LIBRARY})
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${OpenCL_LIBRARIES})
if (WIN32)
target_link_libraries(${PROJECT_NAME} Ws2_32)
else()
target_link_libraries(${PROJECT_NAME} dl)
endif()

