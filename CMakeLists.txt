# CMake 3.18 is required for enable_language(CUDA)
# https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html
#
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(gpurt)

set(PROJECT_VERSION_MAJOR 0)
set(PROJECT_VERSION_MINOR 1)
set(PROJECT_VERSION_PATCH 0)

# Search path for CMake include files.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

enable_language(CUDA)

# Search for OpenCL in the same paths where CUDA is located.
set(OpenCL_INCLUDE_DIR ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
find_package(OpenCL QUIET REQUIRED)

# Enable C++ 11 support
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(ThirdParty/res_embed)

include(ResEmbed)

file(GLOB GPURT_SRC "src/*.cpp" "src/*.cu" "src/*.c")

# Note the library must be static - in order to properly override 
# calls to __cudaRegisterFunction function.
add_library(${PROJECT_NAME} STATIC ${GPURT_SRC})

# Add internal kernels for:
# 1) cl_mem pointer address retrieval
# 2) maximum number of concurrent blocks
res_embed(${PROJECT_NAME} "getRawPointer" "${CMAKE_CURRENT_SOURCE_DIR}/src/getRawPointer.cl")
res_embed(${PROJECT_NAME} "maxConcurrentBlockEval" "${CMAKE_CURRENT_SOURCE_DIR}/src/maxConcurrentBlockEval.cl")

# Note the linked CUDA runtime library must be shared - in order to properly override
# calls to __cudaRegisterFunction function.
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_RUNTIME_LIBRARY Shared)

# Add support for Pascal and Volta GPU.
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 60 61 70)

# An idiomatic way to specify -fPIC flag
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCL_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} cuda ${OpenCL_LIBRARIES})
if (WIN32)
target_link_libraries(${PROJECT_NAME} Ws2_32)
else()
target_link_libraries(${PROJECT_NAME} dl)
endif()

# Test program
add_executable(${PROJECT_NAME}_test "src/test/${PROJECT_NAME}.cpp")
target_link_libraries(${PROJECT_NAME}_test ${PROJECT_NAME})
set_property(TARGET ${PROJECT_NAME}_test PROPERTY CUDA_RUNTIME_LIBRARY Shared)

