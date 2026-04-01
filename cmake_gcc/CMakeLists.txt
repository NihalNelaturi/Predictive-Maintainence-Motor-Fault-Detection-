# Define minimal required version of CMake.
cmake_minimum_required(VERSION "3.25")

# Project definition
project(
	Final2o
	VERSION 1.0
	LANGUAGES C CXX ASM
)

# Include the definition of the slc_Final2o target,
# which contains the content of the SLC project
include(Final2o.cmake)

add_executable(Final2o
    # Add additional sources here
)

target_include_directories(Final2o PUBLIC
    # Add additional include paths here
)

target_compile_definitions(Final2o PUBLIC
    # Add additional macros here
)

target_compile_options(Final2o PUBLIC
    # Set additional compiler flags here
)

target_link_options(Final2o PUBLIC
    # Set additional linker flags here
)

# Link with the content defined in the SLC project
target_link_libraries(Final2o PRIVATE
    slc
)

# Include managed project content if available
include(Final2o_project.cmake OPTIONAL RESULT_VARIABLE managed_project)
if(managed_project)
    message(STATUS "Using managed project content from ${managed_project}")
endif()

# Force the gcc linker command
set_target_properties(Final2o PROPERTIES LINKER_LANGUAGE C)

# Create .bin, .hex and .s37 artifacts after building the project
add_custom_command(TARGET Final2o
    POST_BUILD
    COMMAND ${CMAKE_OBJCOPY} ${OBJCOPY_SREC_CMD} "$<TARGET_FILE:Final2o>" "$<TARGET_FILE_DIR:Final2o>/$<TARGET_FILE_BASE_NAME:Final2o>.s37"
    COMMAND ${CMAKE_OBJCOPY} ${OBJCOPY_IHEX_CMD} "$<TARGET_FILE:Final2o>" "$<TARGET_FILE_DIR:Final2o>/$<TARGET_FILE_BASE_NAME:Final2o>.hex"
    COMMAND ${CMAKE_OBJCOPY} ${OBJCOPY_BIN_CMD}  "$<TARGET_FILE:Final2o>" "$<TARGET_FILE_DIR:Final2o>/$<TARGET_FILE_BASE_NAME:Final2o>.bin" 
)

# Run post-build pipeline to perform additional post-processing
if(post_build_command)
add_custom_command(TARGET Final2o
    POST_BUILD
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/..
    COMMAND ${post_build_command}
)
endif()