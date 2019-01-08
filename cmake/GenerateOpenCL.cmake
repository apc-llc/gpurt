# CMake 3.2 is the minimum version where the CUDA separable
# compilation issue was fixed:
# http://public.kitware.com/pipermail/cmake/2015-January/059482.html
#
# CMake 3.7.2 changes the treatment of host/device compilation flags
# https://gitlab.kitware.com/cmake/cmake/issues/16411
#
cmake_minimum_required(VERSION 3.7.2)

# Read the resulting HEX content into variable
file(READ ${CL_HEX_FILE} CL_HEX)

get_filename_component(CL_NAME ${CL_EMBED_FILE} NAME_WE)

# Substitute encoded HEX content into template source file
configure_file("${SOURCE_DIR}/src/OpenCL.cpp.in" ${CL_EMBED_FILE})

