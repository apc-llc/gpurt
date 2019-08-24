# LIBGPURT - a library to unify CUDA and OpenCL APIs in a CUDA-alike way

Need to offer both CUDA and OpenCL backends in a client application? Prefer CUDA runtime over OpenCL excessive verbosity? This library is an elegant solution: let it take over CUDA and OpenCL from a single API!

## Usage

Integrate the library into your CMake project by adding it as a submodule and as a CMake subdirectory:

```
git submodule add https://github.com/apc-llc/gpurt
```

Application's CMakeLists.txt:

```cmake
...
# Search path for CMake include files.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} gpurt/cmake)
...
file(GLOB SRC "src/*.cpp")

find_package(GPURT REQUIRED)
add_subdirectory(gpurt)

# Add embedded OpenCL kernels into the list of project sources
add_opencl_kernel("${CMAKE_CURRENT_SOURCE_DIR}/src/opencl_kernel.cl" SRC)

add_executable(YOUR_PROJECT ${SRC})
...
# Add GPURT include paths and libray dependency to your project
target_include_directories(YOUR_PROJECT PRIVATE "${GPURT_INCLUDE_DIRS}")
target_link_libraries(YOUR_PROJECT ${GPURT_LIBRARIES})
...
```

## Complete example

See https://github.com/dmikushin/libretracker for complete working example.

