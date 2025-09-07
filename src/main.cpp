// VISITORS BEWARE:
// This program is super inefficient. Don't trust it as a reference for anything.
// Additionally, this has caused GPU freezes and crashes on my PC many times.
// As a precaution, please save everything in every program everywhere before you run this.
///////////////////////////////////////////////////////////////////////////////////////////

// Helps with clearing some warnings.
#define CL_TARGET_OPENCL_VERSION 300

// Program settings

// The program renders once with the camera at the start position and saves to output.bmp, then exits. No window is opened.
// #define RENDER_AND_GET_OUT
// The start position of the camera, if RENDER_AND_GET_OUT is defined.
#define CAMERA_START_X 0.0f
#define CAMERA_START_Y 50.0f
#define CAMERA_START_Z 200.0f
// On each frame, the pRNG seed is different, and all frames are averaged together to produce a less noisy image.
// Obviously, the higher this is, the longer yet higher quality the render will be.
#define FRAME_TOTAL 300
// Functionally equivalent to FRAME_TOTAL, but when RENDER_AND_GET_OUT is NOT defined,
// this is how many times pixels are averaged before 1 frame is sent to the window.
#define RAYS_PER_PIXEL 1
// The resolution of the output image and size of the window, if a window is created.
#define WIDTH 500
#define HEIGHT 500
// The path, absolute or relative (to the cwd), to the .obj file to load.
#define OBJECT_PATH "/home/sovietpancakes/Desktop/Code/gputest/knight.obj"

// imports
#include <CL/cl.h>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include "readobj.hpp"

#ifndef RENDER_AND_GET_OUT
#include <GLFW/glfw3.h>
#endif

std::string loadKernelSource(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    exit(1);
  }
  return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

void placeImageDataIntoBMP(const std::vector<unsigned char>& pixels, int width, int height, const std::string& filename) {
  std::ofstream file(filename, std::ios::binary);
  if (!file)
    return;

  int padSize = (4 - (width * 3) % 4) % 4;
  int rowSize = 3 * width + padSize;
  int dataSize = rowSize * height;
  int fileSize = 54 + dataSize;

  // BMP Header
  unsigned char header[54] = {0};
  header[0] = 'B';
  header[1] = 'M';
  header[2] = fileSize & 0xFF;
  header[3] = (fileSize >> 8) & 0xFF;
  header[4] = (fileSize >> 16) & 0xFF;
  header[5] = (fileSize >> 24) & 0xFF;
  header[10] = 54;
  header[14] = 40;
  header[18] = width & 0xFF;
  header[19] = (width >> 8) & 0xFF;
  header[20] = (width >> 16) & 0xFF;
  header[21] = (width >> 24) & 0xFF;
  header[22] = height & 0xFF;
  header[23] = (height >> 8) & 0xFF;
  header[24] = (height >> 16) & 0xFF;
  header[25] = (height >> 24) & 0xFF;
  header[26] = 1;
  header[28] = 24;
  file.write(reinterpret_cast<char*>(header), 54);

  std::vector<unsigned char> padding(padSize, 0);

  for (int y = height - 1; y >= 0; --y) {
    for (int x = 0; x < width; ++x) {
      size_t i = (y * width + x) * 4;
      unsigned char R = pixels[i + 0];                  // .x
      unsigned char G = pixels[i + 1];                  // .y
      unsigned char B = pixels[i + 2];                  // .z
      std::array<unsigned char, 3> triplet = {B, G, R}; // BMP wants BGR
      file.write(reinterpret_cast<char*>(triplet.data()), 3);
    }
    file.write(reinterpret_cast<char*>(padding.data()), padSize);
  }
  file.close();
}

bool windowIsFocused = true;

int main() {
  cl_int err;
#ifndef RENDER_AND_GET_OUT
  err = glfwInit();
  if (err != GLFW_TRUE) {
    std::cerr << "Failed to initialize GLFW\n";
    return 1;
  }
  GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "GPU Test", nullptr, nullptr);
  glfwMakeContextCurrent(window);
#endif
  cl_uint numPlatforms;
  err = clGetPlatformIDs(0, nullptr, &numPlatforms);
  if (err != CL_SUCCESS || numPlatforms == 0) {
    std::cerr << "Failed to find any OpenCL platforms\n";
    return 1;
  }
  std::vector<cl_platform_id> platforms(numPlatforms);
  err = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to get OpenCL platform IDs\n";
    return 1;
  }
  cl_platform_id platform = platforms[0]; // Pick one lol

  cl_uint numDevices;
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to get number of OpenCL devices\n";
    return 1;
  }
  std::vector<cl_device_id> devices(numDevices);
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to get OpenCL device IDs\n";
    return 1;
  }
  cl_device_id device = devices[0]; // Pick one lol

  // --- get device info
  cl_uint compUnits = 0;
  cl_ulong globalMem = 0;
  err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &compUnits, nullptr);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to get device info\n";
    return 1;
  }
  err = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &globalMem, nullptr);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to get device info\n";
    return 1;
  }

  cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to create context\n";
    return 1;
  }
  cl_command_queue queue = clCreateCommandQueueWithProperties(ctx, device, nullptr, &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to create command queue\n";
    return 1;
  }
  if (!std::filesystem::exists(OBJECT_PATH)) {
    std::cerr << "OBJ file does not exist: " << OBJECT_PATH << std::endl;
    return 1;
  }
  std::string kernelSource = loadKernelSource("src/Trace.cl");
  const char* data = kernelSource.data();
  cl_program program = clCreateProgramWithSource(ctx, 1, &data, nullptr, &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to create program\n";
    return 1;
  }
  err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to build program\n";
    size_t logSize;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
    std::vector<char> log(logSize);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
    std::cerr << "Build log:\n" << log.data() << "\n";
    return 1;
  }

  cl_kernel kernel = clCreateKernel(program, "raytrace", &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to create kernel\n";
    return 1;
  }
  std::chrono::time_point triangleStart = std::chrono::high_resolution_clock::now();
  std::cout << "Loading triangles from mesh..." << std::flush;
  MeshInfo mesh = loadMeshFromOBJFile(OBJECT_PATH);
  std::cout << "bounding box for mesh: min " << mesh.boundsMin << ", max " << mesh.boundsMax << std::endl;
  // Add a light-emitting triangle underneath the dragon
  MeshInfo triangleMesh = {.firstTriangleIdx = (cl_uint)triangleList.size(),
                           .numTriangles = 1,
                           .boundsMin = {-10000.0f, 8000.0f, -10000.0f},
                           .boundsMax = {10000.0f, 12000.0f, 10000.0f},
                           .material = {.color = {0.0f, 0.0f, 0.0f}, .emissionColor = {1.0f, 0.9f, 0.7f}, .emissionStrength = 1.0f}};
  meshList.push_back(triangleMesh);
  triangleList.push_back({.posA = {10000.0f, 10000.0f, -10000.0f},
                          .posB = {-10000.0f, 10000.0f, -10000.0f},
                          .posC = {0.0f, 10000.0f, 10000.0f},
                          .normalA = {0.0f, -1.0f, 0.0f},
                          .normalB = {0.0f, -1.0f, 0.0f},
                          .normalC = {0.0f, -1.0f, 0.0f}});
  cl_mem triangleBuffer =
      clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, triangleList.size() * sizeof(Triangle), triangleList.data(), &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to create triangle buffer\n";
    return 1;
  }
  std::chrono::time_point triangleEndTime = std::chrono::high_resolution_clock::now();
  std::cout << " done in " << std::chrono::duration_cast<std::chrono::milliseconds>(triangleEndTime - triangleStart).count() << " ms ("
            << triangleList.size() << ").";
  std::cout << "\nLoading mesh info..." << std::flush;
  mesh.material.color = {1.0f, 0.5f, 0.5f};
  // mesh.material = {.color = {0.8f, 0.8f, 0.8f}, .emissionColor = {1.0f, 1.0f, 1.0f}, .emissionStrength = 10.0f};
  meshList.push_back(mesh);
  std::chrono::time_point meshEndTime = std::chrono::high_resolution_clock::now();
  std::cout << " done in " << std::chrono::duration_cast<std::chrono::milliseconds>(meshEndTime - triangleEndTime).count() << " ms ("
            << meshList.size() << ")." << std::endl;
  cl_mem meshBuffer = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, meshList.size() * sizeof(MeshInfo), meshList.data(), &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to create mesh buffer\n";
    return 1;
  }
  cl_mem imageBuffer = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, WIDTH * HEIGHT * sizeof(cl_uchar4), nullptr, &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to create image buffer\n";
    return 1;
  }

// --- Simple GL setup (after creating context)
#ifndef RENDER_AND_GET_OUT
  glfwSwapInterval(1); // vsync = 1 (avoid hogging)
  glEnable(GL_TEXTURE_2D);
  GLuint texture;
  glGenTextures(1, &texture);

  // Setup a default viewport and identity matrices for legacy immediate-mode draw
  glViewport(0, 0, WIDTH, HEIGHT);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // Create the texture (bind first)
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  // Pixel storage alignment so each row is tightly packed (4 byte RGBA)
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
#endif

  std::vector<unsigned char> pixels(WIDTH * HEIGHT * 4, 0);

  // args (same order as kernel)
  cl_int meshCount = (cl_int)meshList.size();
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &meshBuffer);
  if (err != CL_SUCCESS) {
    std::cerr << "failed to set kernel arg" << std::endl;
    return 1;
  }
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &triangleBuffer);
  if (err != CL_SUCCESS) {
    std::cerr << "failed to set kernel arg" << std::endl;
    return 1;
  }
  err = clSetKernelArg(kernel, 2, sizeof(cl_int), &meshCount);
  if (err != CL_SUCCESS) {
    std::cerr << "failed to set kernel arg" << std::endl;
    return 1;
  }
  err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &imageBuffer);
  if (err != CL_SUCCESS) {
    std::cerr << "failed to set kernel arg" << std::endl;
    return 1;
  }
  const cl_int width = WIDTH; // you must pass a pointer into clSetKernelArg, meaning you have to pass an lvalue
  const cl_int height = HEIGHT;
  err = clSetKernelArg(kernel, 4, sizeof(cl_int), &width);
  if (err != CL_SUCCESS) {
    std::cerr << "failed to set kernel arg" << std::endl;
    return 1;
  }
  err = clSetKernelArg(kernel, 5, sizeof(cl_int), &height);
  if (err != CL_SUCCESS) {
    std::cerr << "failed to set kernel arg" << std::endl;
    return 1;
  }
  CameraInformation camInfo = {.position = {CAMERA_START_X, CAMERA_START_Y, CAMERA_START_Z},
                               .pitch = 0.0f,
                               .yaw = 0.0f,
                               .roll = 0.0f,
                               .fov = 90.0f,
                               .aspectRatio = (float)WIDTH / (float)HEIGHT};
  err = clSetKernelArg(kernel, 6, sizeof(CameraInformation), &camInfo);
  if (err != CL_SUCCESS) {
    std::cerr << "failed to set kernel arg" << std::endl;
    return 1;
  }
  cl_uint numFrames = 0;
  err = clSetKernelArg(kernel, 7, sizeof(cl_int), &numFrames);
  if (err != CL_SUCCESS) {
    std::cerr << "failed to set kernel arg" << std::endl;
    return 1;
  }

  size_t global[2] = {(size_t)WIDTH, (size_t)HEIGHT};

  clFinish(queue);
#ifndef RENDER_AND_GET_OUT
  std::chrono::time_point lastRecordedTime = std::chrono::high_resolution_clock::now();
  glfwSetWindowFocusCallback(window, [](GLFWwindow* win, int focused) { windowIsFocused = focused != 0; });
  bool isRendering = false;
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    // Camera movement
    // Arrow keys will rotate the camera (pitch, yaw) while
    // QE will move the camera up/down along Y axis
    // WASD will function like a game (W forward, S back, A left, D right)
    // Note: no roll implemented
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS && !isRendering) {
      // "R"ender by averaging frames together
      isRendering = true;
      numFrames = 1;
    }

    if (isRendering) {
      static std::vector<uint32_t> intBuffer;
      if (intBuffer.size() != pixels.size()) {
        intBuffer.assign(pixels.size(), 0u);
        numFrames = 1; // no frames accumulated yet
      }

      if (numFrames >= FRAME_TOTAL) {
        // finished: output final image
        std::vector<unsigned char> finalImg(pixels.size());
        for (size_t i = 0; i < pixels.size(); ++i) {
          finalImg[i] = static_cast<unsigned char>(std::min<uint32_t>(255u, intBuffer[i] / numFrames));
        }
        std::cout << "\rRendering complete, saving output.bmp" << std::endl;
        placeImageDataIntoBMP(finalImg, WIDTH, HEIGHT, "output.bmp");
        isRendering = false;
        continue;
      }

      // --- Run kernel with a seed derived from numFrames
      err = clSetKernelArg(kernel, 6, sizeof(CameraInformation), &camInfo);
      if (err != CL_SUCCESS) {
        std::cerr << "Failed to set kernel arg: " << err << "\n";
        break;
      }
      err = clSetKernelArg(kernel, 7, sizeof(cl_int), &numFrames);
      if (err != CL_SUCCESS) {
        std::cerr << "Failed to set kernel arg: " << err << "\n";
        break;
      }
      err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr);
      if (err != CL_SUCCESS) {
        std::cerr << "Failed to enqueue kernel: " << err << "\n";
        break;
      }
      err = clFinish(queue); // safer than flush when reading back
      if (err != CL_SUCCESS) {
        std::cerr << "Failed to finish command queue: " << err << "\n";
        break;
      }
      err = clEnqueueReadBuffer(queue, imageBuffer, CL_TRUE, 0, pixels.size(), pixels.data(), 0, nullptr, nullptr);
      if (err != CL_SUCCESS) {
        std::cerr << "Failed to read buffer: " << err << "\n";
        break;
      }
      for (size_t i = 0; i < pixels.size(); ++i) {
        intBuffer[i] += pixels[i];
      }

      // Progressive display
      std::vector<unsigned char> accumBuffer(pixels.size());
      for (size_t i = 0; i < pixels.size(); ++i) {
        accumBuffer[i] = static_cast<unsigned char>(intBuffer[i] / numFrames);
      }
      glBindTexture(GL_TEXTURE_2D, texture);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, accumBuffer.data());
      numFrames++;
      std::cout << "\rRendering frame " << numFrames << " of " << FRAME_TOTAL << "..." << std::flush;
    }

    std::chrono::time_point nowTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration deltaTime = nowTime - lastRecordedTime;
    unsigned long millisecondsPassed = std::chrono::duration_cast<std::chrono::milliseconds>(deltaTime).count();
    lastRecordedTime = nowTime;
    if (millisecondsPassed > 0)
      std::cout << "\rfps: " << 1000 / millisecondsPassed << std::flush;
    if (!windowIsFocused) {
      std::this_thread::sleep_for(std::chrono::milliseconds(17));
      continue;
    }
    if (!isRendering) {
      numFrames++; // change the seed even if not rendering
      float deltaSeconds = (float)millisecondsPassed / 1000;
      const float moveSpeed = 60.0f;
      const float rotSpeed = 6.0f;
      if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        camInfo.position.x += moveSpeed * std::sin(camInfo.yaw) * deltaSeconds;
        camInfo.position.z += moveSpeed * std::cos(camInfo.yaw) * deltaSeconds;
      }
      if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        camInfo.position.x -= moveSpeed * std::sin(camInfo.yaw) * deltaSeconds;
        camInfo.position.z -= moveSpeed * std::cos(camInfo.yaw) * deltaSeconds;
      }
      if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        camInfo.position.x -= moveSpeed * std::cos(camInfo.yaw) * deltaSeconds;
        camInfo.position.z += moveSpeed * std::sin(camInfo.yaw) * deltaSeconds;
      }
      if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        camInfo.position.x += moveSpeed * std::cos(camInfo.yaw) * deltaSeconds;
        camInfo.position.z -= moveSpeed * std::sin(camInfo.yaw) * deltaSeconds;
      }
      if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
        camInfo.position.y -= moveSpeed * deltaSeconds;
      }
      if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
        camInfo.position.y += moveSpeed * deltaSeconds;
      }
      if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
        camInfo.pitch -= rotSpeed * deltaSeconds;
      }
      if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
        camInfo.pitch += rotSpeed * deltaSeconds;
      }
      if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
        camInfo.yaw -= rotSpeed * deltaSeconds;
      }
      if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
        camInfo.yaw += rotSpeed * deltaSeconds;
      }
      // Update camera info arg
      err = clSetKernelArg(kernel, 6, sizeof(CameraInformation), &camInfo);
      if (err != CL_SUCCESS) {
        std::cerr << "Failed to set kernel arg: " << err << "\n";
        break;
      }
      err = clSetKernelArg(kernel, 7, sizeof(cl_int), &numFrames);
      if (err != CL_SUCCESS) {
        std::cerr << "Failed to set kernel arg: " << err << "\n";
        break;
      }
      // Enqueue kernel
      err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr);
      if (err != CL_SUCCESS) {
        std::cerr << "Failed to enqueue kernel: " << err << "\n";
        break;
      }

      // Make sure commands have been submitted to the device
      err = clFlush(queue);
      if (err != CL_SUCCESS) {
        std::cerr << "Failed to flush command queue: " << err << "\n";
        break;
      }

      // Blocking read: make sure we read the exact number of bytes
      size_t bytesToRead = pixels.size() * sizeof(unsigned char);
      // Optional: glFinish() to avoid driver-side GL <-> CL race (if GL interop later)
      glFinish();

      err = clEnqueueReadBuffer(queue, imageBuffer, CL_TRUE, 0, bytesToRead, pixels.data(), 0, nullptr, nullptr);
      if (err != CL_SUCCESS) {
        std::cerr << "Failed to read buffer: " << err << "\n";
        break;
      }
      // Upload to the bound OpenGL texture (bind to be explicit)
      glBindTexture(GL_TEXTURE_2D, texture);
      // Use GL_RGBA8 internal format explicitly
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    }

    // Render quad
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // Ensure correct matrices (in case some GL call changed them)
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Draw fullscreen quad. bind texture (already bound above).
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texture);
    glBegin(GL_QUADS);
    // bottom-left
    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(-1.0f, -1.0f);
    // bottom-right
    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(1.0f, -1.0f);
    // top-right
    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(1.0f, 1.0f);
    // top-left
    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(-1.0f, 1.0f);
    glEnd();
    glDisable(GL_TEXTURE_2D);

    glfwSwapBuffers(window);

    err = clFinish(queue);
    if (err != CL_SUCCESS) {
      std::cerr << "Failed to finish command queue: " << err << "\n";
      return 1;
    }
  }
  // Cleanup
  glDeleteTextures(1, &texture);
#else
  // Render the image. Enqueue the kernel 'FRAME_TOTAL' times and average the results.
  static std::vector<uint32_t> intBuffer;
  std::chrono::time_point startTime = std::chrono::high_resolution_clock::now();
  intBuffer.reserve(pixels.size());
  while (numFrames < FRAME_TOTAL) {
    std::cout << "\rRendering frame " << (numFrames + 1) << " of " << FRAME_TOTAL << "..." << std::flush;
    err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
      std::cerr << "Failed to enqueue kernel: " << err << "\n";
      return 1;
    }
    err = clFinish(queue); // safer than flush when reading back
    if (err != CL_SUCCESS) {
      std::cerr << "Failed to finish command queue: " << err << "\n";
      return 1;
    }
    err = clEnqueueReadBuffer(queue, imageBuffer, CL_TRUE, 0, pixels.size(), pixels.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
      std::cerr << "Failed to read buffer: " << err << "\n";
      return 1;
    }
    if (intBuffer.size() != pixels.size()) {
      intBuffer.assign(pixels.size(), 0u);
    }
    for (size_t i = 0; i < pixels.size(); ++i) {
      intBuffer[i] += pixels[i];
    }
    // Set the parameter for the frame number
    err = clSetKernelArg(kernel, 7, sizeof(cl_int), &numFrames);
    if (err != CL_SUCCESS) {
      std::cerr << "Failed to set frame num: " << err << "\n";
      return 1;
    }
    err = clFinish(queue);
    if (err != CL_SUCCESS) {
      std::cerr << "Failed to finish command queue: " << err << "\n";
      return 1;
    }
    numFrames++;
  }
  std::chrono::time_point endTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration deltaTime = endTime - startTime;
  unsigned long millisecondsPassed = std::chrono::duration_cast<std::chrono::milliseconds>(deltaTime).count();
  if (millisecondsPassed > 0)
    std::cout << " done in " << (millisecondsPassed / 1000.0) << " seconds" << std::endl;
  // Average the results (divide)
  std::vector<unsigned char> finalImg(pixels.size());
  for (size_t i = 0; i < pixels.size(); ++i) {
    finalImg[i] = static_cast<unsigned char>(std::min<uint32_t>(255u, intBuffer[i] / FRAME_TOTAL));
  }
  placeImageDataIntoBMP(finalImg, WIDTH, HEIGHT, "output.bmp");
  std::cout << "Wrote output.bmp with " << WIDTH << "x" << HEIGHT << " resolution." << std::endl;
  // We're done!
#endif
  err = clReleaseMemObject(meshBuffer);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to release mesh buffer: " << err << "\n";
    return 1;
  }
  err = clReleaseMemObject(triangleBuffer);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to release triangle buffer: " << err << "\n";
    return 1;
  }
  err = clReleaseMemObject(imageBuffer);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to release image buffer: " << err << "\n";
    return 1;
  }
  err = clReleaseKernel(kernel);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to release kernel: " << err << "\n";
    return 1;
  }
  err = clReleaseProgram(program);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to release program: " << err << "\n";
    return 1;
  }
  err = clReleaseCommandQueue(queue);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to release command queue: " << err << "\n";
    return 1;
  }
  err = clReleaseContext(ctx);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to release context: " << err << "\n";
    return 1;
  }
#ifndef RENDER_AND_GET_OUT
  glfwDestroyWindow(window);
  glfwTerminate();
#endif
  return 0;
}