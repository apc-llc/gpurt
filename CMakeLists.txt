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

# Search path for CMake include files.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE INTERNAL "")
find_package(CUDA QUIET REQUIRED)
find_cuda_helper_libs(cuda)

# Search for OpenCL in the same paths where CUDA is located.
set(OpenCL_INCLUDE_DIR "${CUDA_TOOLKIT_INCLUDE}")
set(ENV{CUDA_PATH} "${CUDA_TOOLKIT_ROOT_DIR}")
find_package(OpenCL QUIET REQUIRED)

find_package(GPURT QUIET REQUIRED)

set(CUDA_SEPARABLE_COMPILATION ON)

set(CUDA_NVCC_FLAGS_RELEASE "-DNDEBUG;-O3;--use_fast_math;-lineinfo;${CUDA_NVCC_FLAGS_RELEASE}")
set(CUDA_NVCC_FLAGS_DEBUG "-G;-g;-O0;${CUDA_NVCC_FLAGS_DEBUG}")

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

add_executable(${PROJECT_NAME}_xxd ${CMAKE_CURRENT_SOURCE_DIR}/xxd/xxd.c)

file(GLOB GPURT_SRC "src/*.cpp" "src/*.cu" "src/*.c")

# Add internal kernels for:
# 1) cl_mem pointer address retrieval
# 2) maximum number of concurrent blocks
add_opencl_kernel("${CMAKE_CURRENT_SOURCE_DIR}/src/getRawPointer.cl" GPURT_SRC)
add_opencl_kernel("${CMAKE_CURRENT_SOURCE_DIR}/src/maxConcurrentBlockEval.cl" GPURT_SRC)

# Note the library must be static - to properly override 
# calls to __cudaRegisterFunction function.
if (WIN32)
cuda_add_library(${PROJECT_NAME} STATIC ${GPURT_SRC})
else()
# Needed to workaround the issue described here:
# https://stackoverflow.com/questions/30642229/fail-to-build-shared-library-using-cmake-and-cuda
cuda_add_library(${PROJECT_NAME} STATIC ${GPURT_SRC} OPTIONS --compiler-options "-fPIC")
endif()
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCL_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} ${CUDA_cuda_LIBRARY})
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${OpenCL_LIBRARIES})
if (WIN32)
target_link_libraries(${PROJECT_NAME} Ws2_32)
else()
target_link_libraries(${PROJECT_NAME} dl)
endif()

# Test program
cuda_add_executable(${PROJECT_NAME}_test "src/test/${PROJECT_NAME}.cpp")
target_link_libraries(${PROJECT_NAME}_test ${PROJECT_NAME})

