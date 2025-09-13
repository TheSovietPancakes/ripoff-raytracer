cmake_minimum_required(VERSION 3.10)

set(OpenGL_GL_PREFERENCE GLVND)

project(gputest)

# Find OpenCL from cmake/FindOpenCL.cmake
find_package(OpenCL)
if(NOT OpenCL_FOUND)
  # if windows
  message(STATUS "OpenCL not found via find_package, attempting manual search...")
  if(WIN32)
    # Try to find common OpenCL.dll locations
    # C:/Program Files/OpenCL-SDK/<...>/
    if(NOT EXISTS "C:/Program Files/OpenCL-SDK")
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
      message(FATAL_ERROR "OpenCL not found. Please install OpenCL SDK.")
    endif()
  endif()
endif()

# # Find GLFW using pkg-config
# find_package(PkgConfig REQUIRED)
# pkg_check_modules(GLFW REQUIRED glfw3)

# # Find OpenGL
# find_package(OpenGL REQUIRED)

# Add the executable
add_executable(${PROJECT_NAME} src/main.cpp)

# Include OpenCL headers
include_directories(${OpenCL_INCLUDE_DIRS})

# # Include GLFW headers from pkg-config
# include_directories(${GLFW_INCLUDE_DIRS})

# Link libraries
# target_link_libraries(${PROJECT_NAME} ${GLFW_LIBRARIES} ${OPENGL_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${OpenCL_LIBRARIES})

# # Specify that the executable requires OpenCL and GLFW
# target_compile_options(${PROJECT_NAME} PRIVATE ${GLFW_CFLAGS_OTHER})

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
