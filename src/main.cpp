#include "settings.hpp"
#include "image.hpp"

#ifndef RENDER_AND_GET_OUT
#include <GLFW/glfw3.h>
#endif
#include <numeric>

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
#else
  // Do this step first, before we allocate anything important:
  // Check if the output directory for video frames exists
  if (VIDEO_FRAME_COUNT > 1) {
    if (!std::filesystem::exists(VIDEO_FRAME_OUTPUT_DIR)) {
      std::cerr << "Output directory for video frames does not exist: " << VIDEO_FRAME_OUTPUT_DIR << std::endl;
      return 1;
    } else {
      // Check if it has any contents
      if (!std::filesystem::is_empty(VIDEO_FRAME_OUTPUT_DIR)) {
        std::cerr << "Output directory for video frames is not empty: " << VIDEO_FRAME_OUTPUT_DIR << std::endl;
        return 1;
      }
    }
  }
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
  char version[128];
  clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(version), version, nullptr);
  std::cout << "OpenCL Version: " << version << std::endl;

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
#if !defined(NODEBUG) && !defined(_NODEBUG)
  {
    char deviceName[128];
    char deviceVendor[128];
    cl_device_type deviceType;
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
    if (err != CL_SUCCESS) {
      std::cerr << "Failed to get device info\n";
      return 1;
    }
    err = clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, nullptr);
    if (err != CL_SUCCESS) {
      std::cerr << "Failed to get device info\n";
      return 1;
    }
    err = clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(deviceVendor), deviceVendor, nullptr);
    if (err != CL_SUCCESS) {
      std::cerr << "Failed to get device info\n";
      return 1;
    }
    std::cout << "Using device: " << deviceName << "\n";
    std::cout << "Compute Units: " << compUnits << "\n";
    std::cout << "Global Memory: " << globalMem / (1024 * 1024) << " MB\n";
    std::cout << "Device Type: " << (deviceType == CL_DEVICE_TYPE_GPU ? "GPU" : deviceType == CL_DEVICE_TYPE_CPU ? "CPU" : "Other") << "\n";
    std::cout << "Device Vendor: " << deviceVendor << "\n";
    std::cout << "-----------------------------------\n";
  }
#endif

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
  std::string kernelSource = loadKernelSource();
  const char* data = kernelSource.data();
  cl_program program = clCreateProgramWithSource(ctx, 1, &data, nullptr, &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to create program\n";
    return 1;
  }
  const char* buildOptions = "-cl-fast-relaxed-math -cl-mad-enable"; // Fast math yay
  err = clBuildProgram(program, 1, &device, buildOptions, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to build program\n";
    size_t logSize;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
    std::vector<char> log(logSize);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
    std::cerr << "Build log:\n" << log.data() << std::endl;
    return 1;
  }

  cl_kernel kernel = clCreateKernel(program, "raytrace", &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to create kernel\n";
    return 1;
  }
  MeshInfo mesh = loadMeshFromOBJFile(OBJECT_PATH);
  mesh.material = {
      .type = MaterialType_Solid,
      .color = {0.8, 0.8, 0.8},
      .emissionColor = {0.0f, 0.0f, 0.0f},
      .emissionStrength = 0.0f,
      .reflectiveness = 1.0f,
      .specularProbability = 0.0f,
  };
  mesh.yaw = 1.5f;
  // KNIGHT
  mesh.scale = 0.5f;
  // DRAGON
  // mesh.pos.y += 60.0f;
  // mesh.scale = 200.0f;

  addCornellBoxToScene(mesh);

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
  meshList.emplace_back(mesh);
  CameraInformation camInfo = {.position = {CAMERA_START_X, CAMERA_START_Y, CAMERA_START_Z},
                               .pitch = CAMERA_START_PITCH,
                               .yaw = CAMERA_START_YAW,
                               .roll = CAMERA_START_ROLL,
                               .fov = 90.0f,
                               .aspectRatio = (float)WIDTH / (float)HEIGHT};
  err = clSetKernelArg(kernel, 6, sizeof(CameraInformation), &camInfo);
  if (err != CL_SUCCESS) {
    std::cerr << "failed to set kernel arg camera information: " << err << std::endl;
    return 1;
  }
  cl_uint numFrames = 0;
  err = clSetKernelArg(kernel, 7, sizeof(cl_int), &numFrames);
  if (err != CL_SUCCESS) {
    std::cerr << "failed to set kernel arg num frames: " << err << std::endl;
    return 1;
  }

  err = clFinish(queue);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to finish command queue: " << err << std::endl;
    return 1;
  }
#ifndef RENDER_AND_GET_OUT
  std::chrono::high_resolution_clock::time_point lastRecordedTime = std::chrono::high_resolution_clock::now();
  glfwSetWindowFocusCallback(window, [](GLFWwindow* win, int focused) { windowIsFocused = focused != 0; });
  bool isRendering = false;
  size_t global[2] = {WIDTH, HEIGHT};
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
        std::cerr << "Failed to set kernel arg camera information: " << err << std::endl;
        break;
      }
      err = clSetKernelArg(kernel, 7, sizeof(cl_int), &numFrames);
      if (err != CL_SUCCESS) {
        std::cerr << "Failed to set kernel arg num frames: " << err << std::endl;
        break;
      }
      err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr);
      if (err != CL_SUCCESS) {
        std::cerr << "Failed to enqueue kernel: " << err << std::endl;
        break;
      }
      err = clFinish(queue); // safer than flush when reading back
      if (err != CL_SUCCESS) {
        std::cerr << "Failed to finish command queue: " << err << std::endl;
        break;
      }
      err = clEnqueueReadBuffer(queue, imageBuffer, CL_TRUE, 0, pixels.size(), pixels.data(), 0, nullptr, nullptr);
      if (err != CL_SUCCESS) {
        std::cerr << "Failed to read buffer: " << err << std::endl;
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

    std::chrono::high_resolution_clock::time_point nowTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration deltaTime = nowTime - lastRecordedTime;
    unsigned long millisecondsPassed = std::chrono::duration_cast<std::chrono::milliseconds>(deltaTime).count();
    lastRecordedTime = nowTime;
    if (millisecondsPassed > 0)
      // add spaces to overwrite previous numbers, if they were printed there
      std::cout << "\rfps: " << 1000 / millisecondsPassed << "   " << std::flush;
    if (!windowIsFocused) {
      std::this_thread::sleep_for(std::chrono::milliseconds(17));
      continue;
    }
    if (!isRendering) {
      numFrames++; // change the seed even if not rendering
      float deltaSeconds = (float)millisecondsPassed / 1000.0f;
      const float moveSpeed = 60.0f;
      const float rotSpeed = 1.5f; // Reduced from 6.0f for more reasonable rotation speed

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
        std::cerr << "Failed to set kernel arg camera information: " << err << std::endl;
        break;
      }
      err = clSetKernelArg(kernel, 7, sizeof(cl_int), &numFrames);
      if (err != CL_SUCCESS) {
        std::cerr << "Failed to set kernel arg num frames: " << err << std::endl;
        break;
      }
      // Enqueue kernel
      err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr);
      if (err != CL_SUCCESS) {
        std::cerr << "Failed to enqueue kernel: " << err << std::endl;
        break;
      }

      // Make sure commands have been submitted to the device
      err = clFlush(queue);
      if (err != CL_SUCCESS) {
        std::cerr << "Failed to flush command queue: " << err << std::endl;
        break;
      }

      // Blocking read: make sure we read the exact number of bytes
      size_t bytesToRead = pixels.size() * sizeof(unsigned char);
      // Optional: glFinish() to avoid driver-side GL <-> CL race (if GL interop later)
      glFinish();

      err = clEnqueueReadBuffer(queue, imageBuffer, CL_TRUE, 0, bytesToRead, pixels.data(), 0, nullptr, nullptr);
      if (err != CL_SUCCESS) {
        std::cerr << "Failed to read buffer: " << err << std::endl;
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
      std::cerr << "Failed to finish command queue: " << err << std::endl;
      return 1;
    }
  }
  // Cleanup
  glDeleteTextures(1, &texture);
#else
  Buffers buffers;
  if (VIDEO_FRAME_COUNT > 1) {
    // Loop it as if this were a video!
    int videoFrameIdx = 0;
    while (videoFrameIdx < VIDEO_FRAME_COUNT) {
      setupNextVideoFrame(camInfo, videoFrameIdx++);
      buffers = generateBuffers(triangleList, meshList, nodeList, ctx, kernel);
      std::cout << "Rendering video frame " << (videoFrameIdx) << " of " << VIDEO_FRAME_COUNT << std::endl;
      // In this context, "numFrames" is still the pRNG seed for the current averaged-together image.
      std::string path = std::string(VIDEO_FRAME_OUTPUT_DIR) + "/output_" + std::to_string(videoFrameIdx) + ".bmp";
      accumulateAndRenderFrame(pixels, numFrames, err, kernel, camInfo, queue, buffers, path, videoFrameIdx);
      releaseBuffers(buffers);
    }
    // We're done!
  } else {
    setupNextVideoFrame(camInfo, 0);
    buffers = generateBuffers(triangleList, meshList, nodeList, ctx, kernel);
    accumulateAndRenderFrame(pixels, numFrames, err, kernel, camInfo, queue, buffers, "output.bmp");
  }
// We're done!
#endif
  err = releaseBuffers(buffers);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to release buffers: " << err << std::endl;
    return 1;
  }
  err = clReleaseKernel(kernel);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to release kernel: " << err << std::endl;
    return 1;
  }
  err = clReleaseProgram(program);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to release program: " << err << std::endl;
    return 1;
  }
  err = clReleaseCommandQueue(queue);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to release command queue: " << err << std::endl;
    return 1;
  }
  err = clReleaseContext(ctx);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to release context: " << err << std::endl;
    return 1;
  }
#ifndef RENDER_AND_GET_OUT
  glfwDestroyWindow(window);
  glfwTerminate();
#endif
  return 0;
}