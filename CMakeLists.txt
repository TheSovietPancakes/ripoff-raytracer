cmake_minimum_required(VERSION 3.10)

set(OpenGL_GL_PREFERENCE GLVND)

project(gputest)

# Find OpenCL from cmake/FindOpenCL.cmake
find_package(OpenCL)
if(NOT OpenCL_FOUND)
  # if windows
  message(STATUS "OpenCL not found via find_package, attempting manual search...")
  if(WIN32)
    # Test the path env:SystemRoot\System32\OpenCL.dll
    # (Which simplifies to C:/Windows/System32/OpenCL.dll)
    if(EXISTS "$ENV{SystemRoot}/System32/OpenCL.dll")
      message(STATUS "Found OpenCL.dll in $ENV{SystemRoot}/System32")
      set(OpenCL_LIBRARY "$ENV{SystemRoot}/System32/OpenCL.dll")

      set(OpenCL_INCLUDE_DIR "C:/Program Files/OpenCL-SDK/include") # Include is different; It gets compiled into the exe. THe DLL is dynamic.
      set(OpenCL_LIBRARIES ${OpenCL_LIBRARY})
      set(OpenCL_INCLUDE_DIRS ${OpenCL_INCLUDE_DIR})
    endif()

    # Try to find common OpenCL.dll locations
    # C:/Program Files/OpenCL-SDK/<...>/
    if(NOT EXISTS "C:/Program Files/OpenCL-SDK" AND NOT OpenCL_FOUND)
      message(FATAL_ERROR "OpenCL SDK not found in C:/Program Files/OpenCL-SDK. Please install OpenCL SDK.")
    endif()

    if(EXISTS "C:/Program Files/OpenCL-SDK/include/CL/cl.h")
      message(STATUS "Found OpenCL headers in C:/Program Files/OpenCL-SDK/include")
      set(OpenCL_FOUND TRUE)
      set(OpenCL_INCLUDE_DIR "C:/Program Files/OpenCL-SDK/include")
      set(OpenCL_LIBRARY "C:/Program Files/OpenCL-SDK/lib/OpenCL.lib")

      set(OpenCL_LIBRARIES ${OpenCL_LIBRARY})
      set(OpenCL_INCLUDE_DIRS ${OpenCL_INCLUDE_DIR})
    endif()
    # Go through each directory of OpenCL-SDK
    # On my system, there was a nested folder, so maybe
    # it's like this for you, too.
    if(NOT OpenCL_FOUND)
      file(GLOB OPENCL_SDK_DIRS "C:/Program Files/OpenCL-SDK/*")

      foreach(OPENCL_SDK_DIR ${OPENCL_SDK_DIRS})
        message(STATUS "Checking OpenCL SDK directory: ${OPENCL_SDK_DIR}")
        if(EXISTS "${OPENCL_SDK_DIR}/include/CL/cl.h")
          message(STATUS "Found OpenCL headers in ${OPENCL_SDK_DIR}/include")
          set(OpenCL_FOUND TRUE)
          set(OpenCL_INCLUDE_DIR "${OPENCL_SDK_DIR}/include")
          set(OpenCL_LIBRARY "${OPENCL_SDK_DIR}/lib/OpenCL.lib")

          set(OpenCL_LIBRARIES ${OpenCL_LIBRARY})
          set(OpenCL_INCLUDE_DIRS ${OpenCL_INCLUDE_DIR})
          break()
        endif()
      endforeach()
    endif()

    if(NOT OpenCL_FOUND)
      message(FATAL_ERROR "OpenCL not found. Please install the OpenCL SDK.")
    endif()
  else()
    message(FATAL_ERROR "OpenCL could not be found for your platform. Please install the OpenCL SDK.")
  endif()
endif()

set(TRACE_CL_SOURCE "src/Trace.cl")
set(TRACE_CL_OUT "src/kernelsource.hpp")
file(READ ${TRACE_CL_SOURCE} TRACE_CL_CONTENTS)
string(REPLACE "\\" "\\\\" TRACE_CL_CONTENTS "${TRACE_CL_CONTENTS}")
string(REPLACE "\n" "\\n" TRACE_CL_CONTENTS "${TRACE_CL_CONTENTS}")
string(REPLACE "\"" "\\\"" TRACE_CL_CONTENTS "${TRACE_CL_CONTENTS}")

file(WRITE ${TRACE_CL_OUT} "// This file is auto-generated from ${TRACE_CL_SOURCE}. Do not edit directly; Changes will be overwritten!\n")
file(APPEND ${TRACE_CL_OUT} "const char* kernel_source = \"${TRACE_CL_CONTENTS}\";\n")

# Add the executable
add_executable(${PROJECT_NAME} src/main.cpp)

# Include GLFW headers
# find_package(PkgConfig REQUIRED)
# pkg_check_modules(GLFW REQUIRED glfw3)
# target_link_libraries(${PROJECT_NAME} ${GLFW_LIBRARIES} ${OPENGL_LIBRARIES})
# include_directories(${GLFW_INCLUDE_DIRS})
# target_compile_options(${PROJECT_NAME} PRIVATE ${GLFW_CFLAGS_OTHER})

# Include OpenCL
include_directories(${OpenCL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCL_LIBRARIES})
target_link_libraries(${PROJECT_NAME} -static) # Most normal people don't have C++ libraries pre-installed on their computer. Who knew!

# # ImGui sources
# set(IMGUI_SOURCES external/imgui/imgui.cpp
#                   external/imgui/imgui_draw.cpp
#                   external/imgui/imgui_tables.cpp
#                   external/imgui/imgui_widgets.cpp
#                   external/imgui/backends/imgui_impl_glfw.cpp
#                   external/imgui/backends/imgui_impl_opengl3.cpp)
# add_library(imgui STATIC ${IMGUI_SOURCES})
# # Include path
# include_directories(external/imgui)

# debug mode
if(CMAKE_COMPILER_IS_GNUCXX)
  if(MSVC OR MINGW)
    # If using MinGW, don't use `-lc++`, use default libstdc++
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -g -std=c++17")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -std=c++17")
  else()
    # For non-Windows GNU compilers (Linux), apply `-lc++` (only if using Clang/GCC with libc++)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -g -std=c++17 -lc++")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -std=c++17 -lc++")
  endif()
else() # Added these arguments for Clang (-stdlib=libc++ is not supported by g++)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -g -stdlib=libc++ -std=c++17")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -stdlib=libc++ -std=c++17")
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)