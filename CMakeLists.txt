cmake_minimum_required(VERSION 3.10)

set(OpenGL_GL_PREFERENCE GLVND) # Prefer opengl 3.0 vs legacy

project(gputest)

# Find OpenCL
find_package(OpenCL REQUIRED)

# Find GLFW - try the modern CMake way first, fallback to manual search
find_package(glfw3 QUIET)
if(NOT glfw3_FOUND)
    # Manual search for GLFW (common on Windows)
    find_path(GLFW_INCLUDE_DIR 
        NAMES GLFW/glfw3.h
        PATHS 
            ${CMAKE_SOURCE_DIR}/third_party/glfw/include
            ${CMAKE_SOURCE_DIR}/glfw/include
            "C:/Program Files/GLFW/include"
            "C:/glfw/include"
            ENV GLFW_DIR
        PATH_SUFFIXES include
    )
    
    find_library(GLFW_LIBRARY
        NAMES glfw3 glfw
        PATHS
            ${CMAKE_SOURCE_DIR}/third_party/glfw/lib
            ${CMAKE_SOURCE_DIR}/glfw/lib
            "C:/Program Files/GLFW/lib"
            "C:/glfw/lib"
            ENV GLFW_DIR
        PATH_SUFFIXES lib lib-vc2019 lib-vc2017 lib-mingw-w64
    )
    
    if(GLFW_INCLUDE_DIR AND GLFW_LIBRARY)
        set(GLFW_FOUND TRUE)
        set(GLFW_LIBRARIES ${GLFW_LIBRARY})
        set(GLFW_INCLUDE_DIRS ${GLFW_INCLUDE_DIR})
        message(STATUS "Found GLFW: ${GLFW_LIBRARY}")
    else()
        message(FATAL_ERROR "GLFW not found! Please install GLFW or set GLFW_DIR environment variable.")
    endif()
else()
    # Modern CMake target found
    set(GLFW_FOUND TRUE)
    set(GLFW_LIBRARIES glfw)
    get_target_property(GLFW_INCLUDE_DIRS glfw INTERFACE_INCLUDE_DIRECTORIES)
endif()

# Find OpenGL
find_package(OpenGL REQUIRED)

# Add the executable
add_executable(${PROJECT_NAME} src/main.cpp)

# Include OpenCL headers
include_directories(${OpenCL_INCLUDE_DIRS})

# Include GLFW headers
include_directories(${GLFW_INCLUDE_DIRS})

# Link libraries
if(TARGET glfw)
    # Use modern CMake target if available
    target_link_libraries(${PROJECT_NAME} ${OpenCL_LIBRARIES} glfw ${OPENGL_LIBRARIES})
else()
    # Use traditional linking
    target_link_libraries(${PROJECT_NAME} ${OpenCL_LIBRARIES} ${GLFW_LIBRARIES} ${OPENGL_LIBRARIES})
endif()

# Debug mode
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")