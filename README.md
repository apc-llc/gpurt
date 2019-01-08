# LIBGPURT - a library to unify CUDA and OpenCL APIs in a CUDA-alike way

Need to offer both CUDA and OpenCL backends in a client application? Prefer CUDA runtime over OpenCL excessive verbosity? This library is an elegant solution: let it take over CUDA and OpenCL from a single API!

## Testing

```
git clone https://github.com/apc-llc/libgpurt
cd libgpurt
mkdir build
cd build
cmake ..
make -j12
```

## Deployment

Integrate the library into your CMake project by adding it as a submodule andi as a CMake subdirectory:

```
git submodule add https://github.com/apc-llc/libgpurt
```

Application's CMakeLists.txt:

```
...
add_subdirectory(libgpurt)
include_directories(libgpurt/include)
...
target_link_libaries(${TARGET} libgpurt)
...
```

