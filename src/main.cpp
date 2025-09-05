#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

// glfw
#include <GLFW/glfw3.h>

using float3 = cl_float3;
using float4 = cl_float4;

typedef struct {
  float3 position;
  float pitch, yaw, roll;
  float fov;
  float aspectRatio;
} CameraInformation;

typedef struct {
  float3 color;
  float3 emissionColor = {0.0f, 0.0f, 0.0f};
  float emissionStrength = 0.0f;
} RayTracingMaterial;

typedef struct {
  float3 center = {0.0f, 0.0f, 0.0f};
  float radius = 1.0f;
  RayTracingMaterial material;
} Sphere;

std::string loadKernelSource(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open kernel file: " + filename);
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

int main() {
  const int width = 1920;
  const int height = 1080;
  // 720p
  // const int width = 1280;
  // const int height = 720;

  glfwInit();
  GLFWwindow* window = glfwCreateWindow(width, height, "GPU Test", nullptr, nullptr);
  glfwMakeContextCurrent(window);

  cl_int err;
  cl_uint numPlatforms;
  clGetPlatformIDs(0, nullptr, &numPlatforms);
  std::vector<cl_platform_id> platforms(numPlatforms);
  clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
  cl_platform_id platform = platforms[0]; // Pick one lol

  cl_uint numDevices;
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
  std::vector<cl_device_id> devices(numDevices);
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);
  cl_device_id device = devices[0]; // Pick one lol

  // --- get device info
  cl_uint compUnits = 0;
  cl_ulong globalMem = 0;
  clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &compUnits, nullptr);
  clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &globalMem, nullptr);
  std::cout << "Compute Units: " << compUnits << "\n";
  std::cout << "Global Memory: " << (globalMem / (1024 * 1024)) << " MB\n";

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
  std::string kernelSource = loadKernelSource("/home/sovietpancakes/Desktop/Code/gputest/src/Trace.cl");
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
    std::cout << "error creating kernel\n" << std::endl;
    return 1;
  }

  std::vector<Sphere> spheres = {Sphere{.center = {2000.0f, 0.0f, 0.0f},
                                        .radius = 999.0f,
                                        .material = {.color = {0.0f, 0.0f, 0.0f}, .emissionColor = {1.0f, 1.0f, 1.0f}, .emissionStrength = 2.0f}},
                                 Sphere{.center = {0.0f, 0.0f, -5.0f},
                                        .radius = 1.0f,
                                        .material = {.color = {1.0f, 0.5f, 0.0f}, .emissionColor = {0.0f, 0.0f, 0.0f}, .emissionStrength = 0.0f}},
                                 Sphere{.center = {0.0f, 500.0f, -5.0f},
                                        .radius = 499.0f,
                                        .material = {.color = {0.1f, 0.7f, 0.5f}, .emissionColor = {0.0f, 0.0f, 0.0f}, .emissionStrength = 0.0f}},
                                 Sphere{.center = {-5.0f, 0.0f, 0.0f},
                                        .radius = 1.0f,
                                        .material = {.color = {1.0f, 0.5f, 0.0f}, .emissionColor = {0.0f, 0.0f, 0.0f}, .emissionStrength = 0.0f}},
                                 Sphere{.center = {5.0f, 0.0f, 0.0f},
                                        .radius = 1.0f,
                                        .material = {.color = {1.0f, 0.0f, 0.0f}, .emissionColor = {0.0f, 0.0f, 0.0f}, .emissionStrength = 1.0f}},
                                 Sphere{.center = {0.0f, 0.0f, 5.0f},
                                        .radius = 1.0f,
                                        .material = {.color = {1.0f, 0.5f, 0.0f}, .emissionColor = {0.0f, 0.0f, 0.0f}, .emissionStrength = 0.0f}}};
  cl_mem sphereBuffer = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, spheres.size() * sizeof(Sphere), spheres.data(), &err);
  cl_mem imageBuffer = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, width * height * sizeof(cl_uchar4), nullptr, &err);

  // --- Simple GL setup (after creating context)
  glfwSwapInterval(1); // vsync = 1 (avoid hogging)
  glEnable(GL_TEXTURE_2D);
  GLuint texture;
  glGenTextures(1, &texture);

  // Setup a default viewport and identity matrices for legacy immediate-mode draw
  glViewport(0, 0, width, height);
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

  std::vector<unsigned char> pixels(width * height * 4, 0);

  // args (same order as kernel)
  cl_int sphereCount = (cl_int)spheres.size();
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &sphereBuffer);
  clSetKernelArg(kernel, 1, sizeof(cl_int), &sphereCount);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &imageBuffer);
  clSetKernelArg(kernel, 3, sizeof(cl_int), &width);
  clSetKernelArg(kernel, 4, sizeof(cl_int), &height);
  CameraInformation camInfo = {
      .position = {0.0f, 0.0f, 0.0f}, .pitch = 0.0f, .yaw = 0.0f, .roll = 0.0f, .fov = 90.0f, .aspectRatio = (float)width / (float)height};
  clSetKernelArg(kernel, 5, sizeof(CameraInformation), &camInfo);
  cl_uint numFrames = 1;
  clSetKernelArg(kernel, 6, sizeof(cl_int), &numFrames);

  size_t global[2] = {(size_t)width, (size_t)height};

  // Debug tip: use a smaller size while debugging (uncomment to test).
  // const int dbgW = 640, dbgH = 360;
  // size_t global[2] = { (size_t)dbgW, (size_t)dbgH };
  std::chrono::time_point lastRecordedTime = std::chrono::high_resolution_clock::now();
  cl_uint targetAvgNum = 100;
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
      std::cout << "Beginning rendering 0/" << targetAvgNum << " frames...\n";
    }

    if (isRendering) {
      static std::vector<uint32_t> intBuffer;
      if (intBuffer.size() != pixels.size()) {
        intBuffer.assign(pixels.size(), 0u);
        numFrames = 0; // no frames accumulated yet
      }

      if (numFrames >= targetAvgNum) {
        // finished: output final image
        std::vector<unsigned char> finalImg(pixels.size());
        for (size_t i = 0; i < pixels.size(); ++i) {
          finalImg[i] = static_cast<unsigned char>(std::min<uint32_t>(255u, intBuffer[i] / numFrames));
        }
        std::cout << "\nFinished rendering " << numFrames << " frames, saving to output.bmp\n";
        placeImageDataIntoBMP(finalImg, width, height, "output.bmp");
        isRendering = false;
        continue;
      }

      // --- Run kernel with a seed derived from numFrames
      clSetKernelArg(kernel, 6, sizeof(cl_int), &numFrames);
      clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr);
      clFinish(queue); // safer than flush when reading back

      clEnqueueReadBuffer(queue, imageBuffer, CL_TRUE, 0, pixels.size(), pixels.data(), 0, nullptr, nullptr);

      // Accumulate this frame
      for (size_t i = 0; i < pixels.size(); ++i) {
        intBuffer[i] += pixels[i];
      }
      numFrames++; // NOW increment, since we’ve added a frame

      // Progressive display
      std::vector<unsigned char> accumBuffer(pixels.size());
      for (size_t i = 0; i < pixels.size(); ++i) {
        accumBuffer[i] = static_cast<unsigned char>(intBuffer[i] / numFrames);
      }
      glBindTexture(GL_TEXTURE_2D, texture);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, accumBuffer.data());

      std::cout << "\rAccumulating frames: " << numFrames << "/" << targetAvgNum << std::flush;
    }

    std::chrono::time_point nowTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration deltaTime = nowTime - lastRecordedTime;
    unsigned long millisecondsPassed = std::chrono::duration_cast<std::chrono::milliseconds>(deltaTime).count();
    if (millisecondsPassed > 0)
      std::cout << "\rfps: " << std::to_string(1000 / (millisecondsPassed)) << std::flush;
    if (!isRendering) {
      float deltaSeconds = (float)millisecondsPassed / 1000;
      lastRecordedTime = nowTime;
      const float moveSpeed = 6.0f;
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
        camInfo.pitch += rotSpeed * deltaSeconds;
      }
      if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
        camInfo.pitch -= rotSpeed * deltaSeconds;
      }
      if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
        camInfo.yaw -= rotSpeed * deltaSeconds;
      }
      if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
        camInfo.yaw += rotSpeed * deltaSeconds;
      }
      // Update camera info arg
      clSetKernelArg(kernel, 5, sizeof(CameraInformation), &camInfo);

      // Enqueue kernel
      cl_int qerr = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr);
      if (qerr != CL_SUCCESS) {
        std::cerr << "Failed to enqueue kernel: " << qerr << "\n";
        break;
      }

      // Make sure commands have been submitted to the device
      cl_int flushErr = clFlush(queue);
      if (flushErr != CL_SUCCESS) {
        std::cerr << "Failed to flush command queue: " << flushErr << "\n";
        break;
      }

      // Blocking read: make sure we read the exact number of bytes
      size_t bytesToRead = pixels.size() * sizeof(unsigned char);
      // Optional: glFinish() to avoid driver-side GL <-> CL race (if using GL interop later)
      glFinish();

      cl_int readErr = clEnqueueReadBuffer(queue, imageBuffer, CL_TRUE, 0, bytesToRead, pixels.data(), 0, nullptr, nullptr);
      if (readErr != CL_SUCCESS) {
        std::cerr << "Failed to read buffer: " << readErr << "\n";
        break;
      }
      // Upload to the bound OpenGL texture (bind to be explicit)
      glBindTexture(GL_TEXTURE_2D, texture);
      // Use GL_RGBA8 internal format explicitly
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
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
  }

  // Cleanup
  glDeleteTextures(1, &texture);
  clFinish(queue);
  clReleaseMemObject(sphereBuffer);
  clReleaseMemObject(imageBuffer);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);
  glfwTerminate();

  return 0;
}