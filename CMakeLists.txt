cmake_minimum_required(VERSION 3.10)

set(OpenGL_GL_PREFERENCE GLVND)

project(gputest)

# Find OpenCL
find_package(OpenCL REQUIRED)

# Find GLFW using pkg-config
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLFW REQUIRED glfw3)

# Find OpenGL
find_package(OpenGL REQUIRED)

# Add the executable
add_executable(${PROJECT_NAME} src/main.cpp)

# Include OpenCL headers
include_directories(${OpenCL_INCLUDE_DIRS})

# Include GLFW headers from pkg-config
include_directories(${GLFW_INCLUDE_DIRS})

# Link libraries
target_link_libraries(${PROJECT_NAME} ${OpenCL_LIBRARIES} ${GLFW_LIBRARIES} ${OPENGL_LIBRARIES})

# Specify that the executable requires OpenCL and GLFW
target_compile_options(${PROJECT_NAME} PRIVATE ${GLFW_CFLAGS_OTHER})

# debug mode
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")