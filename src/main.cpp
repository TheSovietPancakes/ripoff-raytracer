#include "image.hpp"
#include "settings.hpp"

#ifndef RENDER_AND_GET_OUT
#include <GLFW/glfw3.h>
#endif
// #include "backends/imgui_impl_glfw.h"
// #include "backends/imgui_impl_opengl3.h"
// #include "imgui.h"
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
  ImGui::CreateContext();
  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 130");
#else
  // Do this step first, before we allocate anything important:
  // Check if the output directory for video frames exists
  if (VIDEO_FRAME_COUNT > 1) {
    if (!std::filesystem::exists(VIDEO_FRAME_OUTPUT_DIR)) {
      std::cout << "Output directory for video frames does not exist: " << VIDEO_FRAME_OUTPUT_DIR << std::endl;
      std::cout << "Should one be made automatically in the current working directory? (y/N)\n> " << std::flush;
      char response = 'n';
      std::cin >> response;
      if (response == 'y' || response == 'Y') {
        std::filesystem::create_directory(VIDEO_FRAME_OUTPUT_DIR);
      } else {
        std::cout << "Exiting..." << std::endl;
        return 1;
      }
    } else {
      // Check if it has any contents
      if (!std::filesystem::is_empty(VIDEO_FRAME_OUTPUT_DIR)) {
        std::cout << "Output directory for video frames is not empty: " << VIDEO_FRAME_OUTPUT_DIR << std::endl;
        std::cout << "Files will not be overwritten, just in case you have something important in there.\n";
        std::cout << "Please empty it and try again.\nExiting..." << std::endl;
        return 1;
      }
    }
  }
#endif
  std::cout << "Please enter a width, in pixels. For example, 1920, 3840, ...\n> " << std::flush;
  std::cin >> WIDTH;
  std::cout << "Please enter a height, in pixels. For example, 1080, 2160, ...\n> " << std::flush;
  std::cin >> HEIGHT;
  std::cout << "Please enter how many rays per pixel to shoot. Higher = better quality, but slower. 100-500 is a good trade-off.\n> " << std::flush;
  std::cin >> RAYS_PER_PIXEL;
  std::cout << "Please enter the maximum number of bounces per ray. Higher = better quality, but slower, with diminishing returns. 50+ is a good trade-off.\n> " << std::flush;
  std::cin >> MAX_BOUNCE_COUNT;
  std::cout << "Please enter a tile size. 512-4096 is a good trade-off. Powers of 2 are preferred.\n"
            << "Higher = less stable on some GPUs, with more infrequent progress updates.\n"
            << "Higher values are also more likely to freeze or hang, as there is a larger workload at once.\n"
            << "> " << std::flush;
  std::cin >> TILE_SIZE;
  if (TILE_SIZE > WIDTH && TILE_SIZE > HEIGHT) {
    TILE_SIZE = std::min(WIDTH, HEIGHT);
    std::cout << "Invalid tile size, using " << TILE_SIZE << " instead.\n";
  }
  
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
      .emissionColor = {1.0f, 1.0f, 1.0f},
      .emissionStrength = 0.0f,
      .reflectiveness = 0.7f,
      .specularProbability = 1.0f,
  };
  mesh.yaw = 1.5f;
  // KNIGHT
  // mesh.scale = 0.5f;
  // DRAGON
  mesh.pos.s[1] += 60.0f;
  mesh.scale = 200.0f;
  CAMERA_START_Y -= 60.0f;

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
  cl_uint bounceCount = MAX_BOUNCE_COUNT;
  cl_uint raysPerPixel = RAYS_PER_PIXEL;
  err = clSetKernelArg(kernel, 10, sizeof(cl_int), &raysPerPixel);
  if (err != CL_SUCCESS) {
    std::cerr << "failed to set kernel arg rays per pixel: " << err << std::endl;
    return 1;
  }

  err = clFinish(queue);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to finish command queue: " << err << std::endl;
    return 1;
  }
  Buffers buffers;
#ifndef RENDER_AND_GET_OUT
  std::chrono::high_resolution_clock::time_point lastRecordedTime = std::chrono::high_resolution_clock::now();
  glfwSetWindowFocusCallback(window, [](GLFWwindow* win, int focused) { windowIsFocused = focused != 0; });
  size_t global[2] = {WIDTH, HEIGHT};
  buffers = generateBuffers(triangleList, meshList, nodeList, ctx, kernel);
  bool shouldRefreshBuffers = false;
  // COMPLETELY IGNORE the buffers.imageBuffer here, as we will
  // average frames together in here :)
  int selectedMeshIdx = -1;
  std::vector<float3> intBuffer(WIDTH * HEIGHT, {0.0f, 0.0f, 0.0f});
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGui::Begin("Settings");
    // Camera movement
    // Arrow keys will rotate the camera (pitch, yaw) while
    // QE will move the camera up/down along Y axis
    // WASD will function like a game (W forward, S back, A left, D right)

    if (shouldRefreshBuffers) {
      releaseBuffers(buffers);
      buffers = generateBuffers(triangleList, meshList, nodeList, ctx, kernel);
      shouldRefreshBuffers = false;
      numFrames = 0;
      intBuffer.assign(intBuffer.size(), {0.0f, 0.0f, 0.0f});
      // Map the mesh buffer to host memory
      if (selectedMeshIdx >= 0 && selectedMeshIdx < (int)meshList.size()) {

        MeshInfo* hostMeshBuffer = (MeshInfo*)clEnqueueMapBuffer(queue, buffers.meshBuffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0,
                                                                 sizeof(MeshInfo) * meshList.size(), 0, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) {
          std::cerr << "Failed to map mesh buffer: " << err << std::endl;
          return 1;
        }

        // Modify the selected mesh's material color
        hostMeshBuffer[selectedMeshIdx].material.color = {1.0f, 0.0f, 0.0f};

        // Unmap the buffer
        err = clEnqueueUnmapMemObject(queue, buffers.meshBuffer, hostMeshBuffer, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
          std::cerr << "Failed to unmap mesh buffer: " << err << std::endl;
          return 1;
        }
        err = clFinish(queue);
        if (err != CL_SUCCESS) {
          std::cerr << "Failed to finish command queue after unmapping: " << err << std::endl;
          return 1;
        }
      }
    }

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS && windowIsFocused) {
      // Get mouse position, relative to window and WIDTH/HEIGHT
      double mouseX, mouseY;
      glfwGetCursorPos(window, &mouseX, &mouseY);
      // Spawn the test kernel
      cl_kernel testKernel = clCreateKernel(program, "checkIntersectingRay", &err);
      if (err != CL_SUCCESS) {
        std::cerr << "Failed to create test kernel\n";
        return 1;
      }
      /*
      __kernel void checkIntersectingRay(__global const MeshInfo *meshList,
                                   int meshCount,
                                   __global const Triangle *triangleList,
                                   __global const Node *nodeList,
                                   CameraInformation camInfo, float2 uv,
                                   __global int *outMeshIdx)
      */
      err = clSetKernelArg(testKernel, 0, sizeof(cl_mem), &buffers.meshBuffer);
      if (err != CL_SUCCESS) {
        std::cerr << "failed to set test kernel arg mesh buffer: " << err << std::endl;
        return 1;
      }
      cl_int meshCount = (cl_int)meshList.size();
      err = clSetKernelArg(testKernel, 1, sizeof(cl_int), &meshCount);
      if (err != CL_SUCCESS) {
        std::cerr << "failed to set test kernel arg mesh count: " << err << std::endl;
        return 1;
      }
      err = clSetKernelArg(testKernel, 2, sizeof(cl_mem), &buffers.triangleBuffer);
      if (err != CL_SUCCESS) {
        std::cerr << "failed to set test kernel arg triangle buffer: " << err << std::endl;
        return 1;
      }
      err = clSetKernelArg(testKernel, 3, sizeof(cl_mem), &buffers.nodeBuffer);
      if (err != CL_SUCCESS) {
        std::cerr << "failed to set test kernel arg node buffer: " << err << std::endl;
        return 1;
      }
      err = clSetKernelArg(testKernel, 4, sizeof(CameraInformation), &camInfo);
      if (err != CL_SUCCESS) {
        std::cerr << "failed to set test kernel arg camera information: " << err << std::endl;
        return 1;
      }
      cl_float2 uv = {(float)mouseX / (float)WIDTH, (float)mouseY / (float)HEIGHT};
      err = clSetKernelArg(testKernel, 5, sizeof(cl_float2), &uv);
      if (err != CL_SUCCESS) {
        std::cerr << "failed to set test kernel arg UV coordinates: " << err << std::endl;
        return 1;
      }
      cl_int outMeshIdx = -1;
      cl_mem outMeshIdxBuffer = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int), &outMeshIdx, &err);
      if (err != CL_SUCCESS) {
        std::cerr << "failed to create output mesh index buffer: " << err << std::endl;
        return 1;
      }
      err = clSetKernelArg(testKernel, 6, sizeof(cl_mem), &outMeshIdxBuffer);
      if (err != CL_SUCCESS) {
        std::cerr << "failed to set test kernel arg output mesh index buffer: " << err << std::endl;
        return 1;
      }
      size_t onePixel[2] = {1, 1};
      err = clEnqueueNDRangeKernel(queue, testKernel, 2, nullptr, onePixel, nullptr, 0, nullptr, nullptr);
      if (err != CL_SUCCESS) {
        std::cerr << "Failed to enqueue test kernel: " << err << std::endl;
        return 1;
      }
      err = clEnqueueReadBuffer(queue, outMeshIdxBuffer, CL_TRUE, 0, sizeof(cl_int), &outMeshIdx, 0, nullptr, nullptr);
      if (err != CL_SUCCESS) {
        std::cerr << "Failed to read output mesh index buffer: " << err << std::endl;
        return 1;
      }
      clReleaseMemObject(outMeshIdxBuffer);
      clReleaseKernel(testKernel);
      if (selectedMeshIdx != outMeshIdx) {
        selectedMeshIdx = outMeshIdx;
        shouldRefreshBuffers = true;
      }
      // Make sure commands have been submitted to the device
      err = clFinish(queue);
      if (err != CL_SUCCESS) {
        std::cerr << "Failed to finish command queue after test kernel: " << err << std::endl;
        return 1;
      }
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
    numFrames++; // change the seed even if not rendering
    float deltaSeconds = (float)millisecondsPassed / 1000.0f;
    const float moveSpeed = 100.0f;
    const float rotSpeed = 1.5f; // Reduced from 6.0f for more reasonable rotation speed

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
      shouldRefreshBuffers = true;
      camInfo.position.s[0] += moveSpeed * std::sin(camInfo.yaw) * deltaSeconds;
      camInfo.position.s[2] += moveSpeed * std::cos(camInfo.yaw) * deltaSeconds;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
      shouldRefreshBuffers = true;
      camInfo.position.s[0] -= moveSpeed * std::sin(camInfo.yaw) * deltaSeconds;
      camInfo.position.s[2] -= moveSpeed * std::cos(camInfo.yaw) * deltaSeconds;
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
      shouldRefreshBuffers = true;
      camInfo.position.s[0] -= moveSpeed * std::cos(camInfo.yaw) * deltaSeconds;
      camInfo.position.s[2] += moveSpeed * std::sin(camInfo.yaw) * deltaSeconds;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
      shouldRefreshBuffers = true;
      camInfo.position.s[0] += moveSpeed * std::cos(camInfo.yaw) * deltaSeconds;
      camInfo.position.s[2] -= moveSpeed * std::sin(camInfo.yaw) * deltaSeconds;
    }
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
      shouldRefreshBuffers = true;
      camInfo.position.s[1] -= moveSpeed * deltaSeconds;
    }
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
      shouldRefreshBuffers = true;
      camInfo.position.s[1] += moveSpeed * deltaSeconds;
    }
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
      shouldRefreshBuffers = true;
      camInfo.pitch -= rotSpeed * deltaSeconds;
    }
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
      shouldRefreshBuffers = true;
      camInfo.pitch += rotSpeed * deltaSeconds;
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
      shouldRefreshBuffers = true;
      camInfo.yaw -= rotSpeed * deltaSeconds;
    }
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
      shouldRefreshBuffers = true;
      camInfo.yaw += rotSpeed * deltaSeconds;
    }
    // Update camera info arg
    err = clSetKernelArg(kernel, 6, sizeof(CameraInformation), &camInfo);
    if (err != CL_SUCCESS) {
      std::cerr << "Failed to set kernel arg camera information: " << err << std::endl;
    }
    err = clSetKernelArg(kernel, 7, sizeof(cl_int), &numFrames);
    if (err != CL_SUCCESS) {
      std::cerr << "Failed to set kernel arg num frames: " << err << std::endl;
    }
    err = clSetKernelArg(kernel, 9, sizeof(cl_int), &bounceCount);
    if (err != CL_SUCCESS) {
      std::cerr << "failed to set kernel arg bounce count: " << err << std::endl;
      return 1;
    }
    err = clSetKernelArg(kernel, 10, sizeof(cl_int), &raysPerPixel);
    if (err != CL_SUCCESS) {
      std::cerr << "failed to set kernel arg rays per pixel: " << err << std::endl;
      return 1;
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

    err = clEnqueueReadBuffer(queue, buffers.imageBuffer, CL_TRUE, 0, bytesToRead, pixels.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
      std::cerr << "Failed to read buffer: " << err << std::endl;
      break;
    }
    // ADD!!! buffers.imageBuffer to intCLBuffer
    for (size_t i = 0; i < WIDTH * HEIGHT; i++) {
      intBuffer[i].s[0] += pixels[i * 4 + 0];
      intBuffer[i].s[1] += pixels[i * 4 + 1];
      intBuffer[i].s[2] += pixels[i * 4 + 2];

      pixels[i * 4 + 0] = (unsigned char)(intBuffer[i].s[0] / numFrames);
      pixels[i * 4 + 1] = (unsigned char)(intBuffer[i].s[1] / numFrames);
      pixels[i * 4 + 2] = (unsigned char)(intBuffer[i].s[2] / numFrames);
    }
    // Upload to the bound OpenGL texture (bind to be explicit)
    glBindTexture(GL_TEXTURE_2D, texture);
    // Use GL_RGBA8 internal format explicitly
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

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

    // ImGUI stuff
    ImGui::Text("FPS: %ld", millisecondsPassed > 0 ? 1000 / millisecondsPassed : 0);
    ImGui::Text("Frames averaged: %d", numFrames);
    ImGui::Text("Camera Position: (%.1f, %.1f, %.1f)", camInfo.position.s[0], camInfo.position.s[1], camInfo.position.s[2]);
    ImGui::Text("Camera Pitch: %.2f", camInfo.pitch);
    ImGui::Text("Camera Yaw: %.2f", camInfo.yaw);
    ImGui::Text("Camera Roll: %.2f", camInfo.roll);
    ImGui::Text("Move: WASD, QE (up/down), Rotate: Arrows");
    ImGui::SliderInt("Rays per pixel", (int*)&raysPerPixel, 1, 64);
    ImGui::SliderInt("Max bounce count", (int*)&bounceCount, 1, 20);
    if (ImGui::Button("Reset View")) {
      camInfo.position = {CAMERA_START_X, CAMERA_START_Y, CAMERA_START_Z};
      camInfo.pitch = CAMERA_START_PITCH;
      camInfo.yaw = CAMERA_START_YAW;
      camInfo.roll = CAMERA_START_ROLL;
      shouldRefreshBuffers = true;
    }
    if (ImGui::Button("Refresh Buffers")) {
      shouldRefreshBuffers = true;
    }
    ImGui::Text("Triangles: %zu", triangleList.size());
    ImGui::Text("Meshes: %zu", meshList.size());
    ImGui::Text("Top-level nodes: %zu", nodeList.size());
    ImGui::End();
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

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
  err = clSetKernelArg(kernel, 6, sizeof(CameraInformation), &camInfo);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to set kernel arg camera information: " << err << std::endl;
    return 1;
  }
  err = clSetKernelArg(kernel, 7, sizeof(cl_int), &numFrames);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to set kernel arg num frames: " << err << std::endl;
    return 1;
  }
  err = clSetKernelArg(kernel, 9, sizeof(cl_int), &bounceCount);
  if (err != CL_SUCCESS) {
    std::cerr << "failed to set kernel arg bounce count: " << err << std::endl;
    return 1;
  }
  err = clSetKernelArg(kernel, 10, sizeof(cl_int), &raysPerPixel);
  if (err != CL_SUCCESS) {
    std::cerr << "failed to set kernel arg rays per pixel: " << err << std::endl;
    return 1;
  }
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
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();
#endif
  return 0;
}