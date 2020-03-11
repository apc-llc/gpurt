if (NOT GPURT_FOUND)
set(GPURT_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/../include" CACHE STRING "" FORCE)
set(GPURT_LIBRARIES gpurt CACHE STRING "" FORCE)
set(GPURT_FOUND TRUE)
endif ()

# Convert OpenCL source into a binary blob and add it to the list of SOURCES.
# Preprocess OpenCL sources with non-trivial include tree structure.
macro(add_opencl_kernel CL_FILE_SRC SOURCES)
	set(CL_EMBED gpurt_xxd)
	cuda_compute_build_path("${CL_FILE_SRC}" CL_BUILD_PATH)
	set(CL_INTERMEDIATE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${PROJECT_NAME}.dir/${CL_BUILD_PATH}") 

	# Preprocess source file into self-contained OpenCL source file.
	get_filename_component(CL_FILE_BASE ${CL_FILE_SRC} NAME)
	set(CL_FILE "${CL_INTERMEDIATE_DIRECTORY}/${CL_FILE_BASE}.cl")
	get_property(INCLUDE_DIRS DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
	list(APPEND INCLUDE_DIRS ${OpenCL_INCLUDE_DIR})
	set(INCLUDE_FLAGS "")
	foreach(INCLUDE_DIR ${INCLUDE_DIRS})
		list(APPEND INCLUDE_FLAGS "-I${INCLUDE_DIR}")
	endforeach()
	list(APPEND INCLUDE_FLAGS "-I${GPURT_INCLUDE_DIRS}")
	add_custom_commaind(
        	OUTPUT ${CL_FILE}
	        COMMAND ${CMAKE_COMMAND} -E make_directory "${CL_INTERMEDIATE_DIRECTORY}"
		COMMAND "${CUDA_NVCC_EXECUTABLE}" -x c++ -E -D__OPENCL_VERSION__=1 ${INCLUDE_FLAGS} ${CL_FILE_SRC} -o ${CL_FILE}
        	COMMENT "Preprocessing source file ${CL_FILE_SRC} for OpenCL"
	        DEPENDS ${CL_FILE_SRC})
	set_source_files_properties("${CL_FILE}" PROPERTIES GENERATED TRUE) 

	# Translate OpenCL source into comma-separated byte codes.
	set(CL_HEX_FILE "${CL_INTERMEDIATE_DIRECTORY}/${CL_FILE_BASE}.hex")
	add_custom_command(
		OUTPUT ${CL_HEX_FILE}
		COMMAND ${CMAKE_COMMAND} -E make_directory "${CL_INTERMEDIATE_DIRECTORY}"
		COMMAND ${CL_EMBED} -i < ${CL_FILE} > ${CL_HEX_FILE}
		COMMENT "Generating hex representation for OpenCL source file ${CL_FILE}"
		DEPENDS ${CL_FILE} ${CL_EMBED})
	set_source_files_properties("${CL_HEX_FILE}" PROPERTIES GENERATED TRUE) 
	set(CL_EMBED_FILE "${CL_INTERMEDIATE_DIRECTORY}/${CL_FILE_BASE}.cpp")
	add_custom_command(
		OUTPUT ${CL_EMBED_FILE}
		COMMAND ${CMAKE_COMMAND} -E make_directory "${CL_INTERMEDIATE_DIRECTORY}"
		COMMAND ${CMAKE_COMMAND} -DGPURT_INCLUDE_DIRS=${GPURT_INCLUDE_DIRS} -DCL_HEX_FILE=${CL_HEX_FILE} -DCL_EMBED_FILE=${CL_EMBED_FILE} -P ${GPURT_INCLUDE_DIRS}/../cmake/GenerateOpenCL.cmake
		COMMENT "Embedding OpenCL source file ${CL_FILE}"
		DEPENDS ${CL_HEX_FILE} "${GPURT_INCLUDE_DIRS}/OpenCL.cpp.in")
	set_source_files_properties("${CL_EMBED_FILE}" PROPERTIES GENERATED TRUE) 
	# Submit the resulting source file for compilation
	list(APPEND ${SOURCES} ${CL_EMBED_FILE})
	message(STATUS "OpenCL kernel ${CL_FILE} shall be added to ${SOURCES}")
endmacro()

