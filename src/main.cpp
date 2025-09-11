// VISITORS BEWARE:
// This program is super inefficient. Don't trust it as a reference for anything.
// Additionally, this has caused GPU freezes and crashes on my PC many times.
// As a precaution, please save everything in every program everywhere before you run this.
// For your own computer's safety, this program is totally OK to interrupt with ^C.
///////////////////////////////////////////////////////////////////////////////////////////

// Helps with clearing some warnings.
#define CL_TARGET_OPENCL_VERSION 300

// Program settings

// The program renders once with the camera at the start position and saves to output.bmp, then exits. No window is opened.
// If commented or removed, a window is created and a movable (WASD+QE+arrows) camera is used instead.
#define RENDER_AND_GET_OUT
// Give the GPU a break (~250ms) between frames to avoid crashing.
// Additionally, every 10 frames, a preview.bmp image is outputted to show current render progress.
// (since it's better to find out you mesed up after 10 frames rather than after 1,000.)
#define RELAX_GPU
// The start position of the camera, if RENDER_AND_GET_OUT is defined.
#define CAMERA_START_X 0.0f
#define CAMERA_START_Y 100.0f
#define CAMERA_START_Z 300.0f
#define CAMERA_START_PITCH 0.0f
#define CAMERA_START_YAW 3.14f
#define CAMERA_START_ROLL 0.0f
// On each frame, the pRNG's seed is different, and all frames are averaged together to produce a less noisy image.
// Obviously, the higher this is, the longer yet higher quality the render will be.
#define FRAME_TOTAL 1
// Functionally equivalent to FRAME_TOTAL, but when RENDER_AND_GET_OUT is NOT defined,
// this is how many times pixels are averaged before 1 frame is sent to the window.
// #define RAYS_PER_PIXEL 1 // unimplemented - open Trace.cl
// The resolution of the output image and size of the window, if a window is created.
// #define WIDTH 1080
// #define HEIGHT 1920
// These are the dimensions of an iPhone 16, the phone that I have lol
// #define WIDTH 1179
// #define HEIGHT 2556

#define WIDTH 500
#define HEIGHT WIDTH
// The path, absolute or relative (to the cwd), to the .obj file to load.
#define OBJECT_PATH "/home/sovietpancakes/Desktop/Code/gputest/dragon.obj"
// How much space there is inside the Cornell box between the model and the walls
#define CORNELL_BREATHING_ROOM 200.0f

#include "readobj.hpp"

#ifndef RENDER_AND_GET_OUT
#include <GLFW/glfw3.h>
#endif
#include <numeric>

std::string loadKernelSource(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cout << "File not found or not opened: " << filename << std::endl;
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
  const char* buildOptions = "-cl-fast-relaxed-math -cl-mad-enable"; // Fast math yay
  err = clBuildProgram(program, 1, &device, buildOptions, nullptr, nullptr);
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
  std::chrono::high_resolution_clock::time_point triangleStart = std::chrono::high_resolution_clock::now();
  std::cout << "Loading triangles from mesh..." << std::flush;
  MeshInfo mesh = loadMeshFromOBJFile(OBJECT_PATH);
  mesh.material = {
      .type = MaterialType_Solid,
      .color = {0.7f, 0.7f, 0.7f},
      .emissionColor = {0.0f, 0.0f, 0.0f},
      .emissionStrength = 0.0f,
      .reflectiveness = 0.0f,
      .specularProbability = 0.0f,
  };
  mesh.pos.y += 60.0f;
  mesh.scale = 200.0f;
  // mesh.scale = 0.5f;
  mesh.yaw = 1.5f;
  // mesh.material = {.color = {0.8f, 0.8f, 0.8f}, .emissionColor = {1.0f, 1.0f, 1.0f}, .emissionStrength = 10.0f};
  // mesh.yaw = 0.5f;
  // mesh.scale = 0.5f;

  // Add a light-emitting triangle underneath the dragon
  auto addQuad = [&](cl_float3 a, cl_float3 b, cl_float3 c, cl_float3 d, cl_float3 normal, cl_float3 color) {
    Node n = {
        .bounds =
            {
                .min = {fminf(fminf(a.x, b.x), fminf(c.x, d.x)), fminf(fminf(a.y, b.y), fminf(c.y, d.y)), fminf(fminf(a.z, b.z), fminf(c.z, d.z))},
                .max = {fmaxf(fmaxf(a.x, b.x), fmaxf(c.x, d.x)), fmaxf(fmaxf(a.y, b.y), fmaxf(c.y, d.y)), fmaxf(fmaxf(a.z, b.z), fmaxf(c.z, d.z))},
            },
        .childIndex = 0,
        .firstTriangleIdx = (cl_uint)triangleList.size(),
        .numTriangles = 2,
    };
    nodeList.push_back(n);
    // SplitBVH(nodeList.size() - 1);
    SplitBVH(nodeList.back());
    MeshInfo quadMesh = {.nodeIdx = nodeList.size() - 1, // will be correct after SplitBVH
                         .material = {
                             .type = MaterialType_Solid,
                             .color = color,
                             .emissionColor = {0, 0, 0},
                             .emissionStrength = 0.0f,
                             .reflectiveness = 0.0f,
                             .specularProbability = 1.0f,
                         }};

    // two triangles
    triangleList.push_back({a, b, c, normal, normal, normal});
    triangleList.push_back({a, c, d, normal, normal, normal});
    meshList.push_back(quadMesh);
  };

  float minX = nodeList[mesh.nodeIdx].bounds.min.x - CORNELL_BREATHING_ROOM, maxX = nodeList[mesh.nodeIdx].bounds.max.x + CORNELL_BREATHING_ROOM;
  float minY = nodeList[mesh.nodeIdx].bounds.min.y,
        maxY = nodeList[mesh.nodeIdx].bounds.max.y + CORNELL_BREATHING_ROOM; // do not sub so the model touches the floor
  float minZ = nodeList[mesh.nodeIdx].bounds.min.z - CORNELL_BREATHING_ROOM, maxZ = nodeList[mesh.nodeIdx].bounds.max.z + CORNELL_BREATHING_ROOM;

  // Floor (Y = minY)
  addQuad({minX, minY, minZ}, {maxX, minY, minZ}, {maxX, minY, maxZ}, {minX, minY, maxZ}, {0, 1, 0}, {0.0f, 0.8f, 0.0f});
  meshList.back().material = {
      .type = MaterialType_Checker,
      .color = {0.1, 0.1, 0.1},
      .emissionColor = {0.8, 0.8, 0.8},
      .emissionStrength = 40.0f,
      .reflectiveness = 1.0f,
      .specularProbability = 0.0f,
  };

  // Ceiling (Y = maxY)
  addQuad({minX, maxY, minZ}, {maxX, maxY, minZ}, {maxX, maxY, maxZ}, {minX, maxY, maxZ}, {0, -1, 0}, {0.8f, 0.8f, 0.8f});

  // Back wall (Z = maxZ)
  addQuad({minX, minY, maxZ}, {maxX, minY, maxZ}, {maxX, maxY, maxZ}, {minX, maxY, maxZ}, {0, 0, -1}, {0.8f, 0.8f, 0.8f});

  // Front wall (Z = minZ)
  addQuad({minX, minY, minZ}, {maxX, minY, minZ}, {maxX, maxY, minZ}, {minX, maxY, minZ}, {0, 0, 1}, {0.8, 0.8, 0.8});
  // meshList.back().material.reflectiveness = 0.9; // slightly less than a mirror

  // Left wall (X = minX)
  addQuad({minX, minY, minZ}, {minX, minY, maxZ}, {minX, maxY, maxZ}, {minX, maxY, minZ}, {1, 0, 0}, {0.2f, 0.2f, 0.4});

  // Right wall (X = maxX)
  addQuad({maxX, minY, minZ}, {maxX, minY, maxZ}, {maxX, maxY, maxZ}, {maxX, maxY, minZ}, {-1, 0, 0}, {0.4f, 0.2f, 0.2f});

  // Light quad on ceiling
  float lx = 50, lz = 50, ly = maxY - 1; // just below ceiling
  addQuad({-lx, ly, -lz}, {lx, ly, -lz}, {lx, ly, lz}, {-lx, ly, lz}, {0, -1, 0}, {0.0f, 0.0f, 0.0f});
  meshList.back().material = {.type = MaterialType_Solid,
                              .color = {0.0f, 0.0f, 0.0f},
                              .emissionColor = {1.0f, 1.0f, 1.0f},
                              .emissionStrength = 25.0f,
                              .reflectiveness = 0.0f,
                              .specularProbability = 1};

  cl_mem triangleBuffer =
      clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, triangleList.size() * sizeof(Triangle), triangleList.data(), &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to create triangle buffer\n";
    return 1;
  }
  std::chrono::high_resolution_clock::time_point triangleEndTime = std::chrono::high_resolution_clock::now();
  std::cout << " done in " << std::chrono::duration_cast<std::chrono::milliseconds>(triangleEndTime - triangleStart).count() << " ms ("
            << triangleList.size() << ").";
  std::cout << "\nLoading mesh info..." << std::flush;
  meshList.push_back(mesh);
  std::chrono::high_resolution_clock::time_point meshEndTime = std::chrono::high_resolution_clock::now();
  std::cout << " done in " << std::chrono::duration_cast<std::chrono::milliseconds>(meshEndTime - triangleEndTime).count() << " ms ("
            << meshList.size() << ")." << std::endl;
  cl_mem meshBuffer = clCreateBuffer(ctx, CL_MEM_COPY_HOST_PTR, meshList.size() * sizeof(MeshInfo), meshList.data(), &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to create mesh buffer\n";
    return 1;
  }
  cl_mem imageBuffer = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, WIDTH * HEIGHT * sizeof(cl_uchar4), nullptr, &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to create image buffer\n";
    return 1;
  }
  std::vector<GPUNode> gpuNodes;
  gpuNodes.resize(nodeList.size());
  for (size_t i = 0; i < nodeList.size(); ++i) {
    GPUNode n = {
        .bounds = nodeList[i].bounds,
        .index = nodeList[i].childIndex == 0 ? nodeList[i].firstTriangleIdx : nodeList[i].childIndex,
        .numTriangles = nodeList[i].childIndex == 0 ? nodeList[i].numTriangles : 0,
    };
    gpuNodes[i] = n;
  }
  std::cout << "memory gained: " << (sizeof(Node) * nodeList.size() - sizeof(GPUNode) * gpuNodes.size()) / 1024 << " KB\n";

  cl_mem nodeBuffer = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, gpuNodes.size() * sizeof(GPUNode), gpuNodes.data(), &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to create node buffer\n";
    return 1;
  }
  err = clSetKernelArg(kernel, 8, sizeof(cl_mem), &nodeBuffer);
  if (err != CL_SUCCESS) {
    std::cerr << "failed to set kernel arg" << std::endl;
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
                               .pitch = CAMERA_START_PITCH,
                               .yaw = CAMERA_START_YAW,
                               .roll = CAMERA_START_ROLL,
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

  clFinish(queue);
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
  std::chrono::high_resolution_clock::time_point startTime = std::chrono::high_resolution_clock::now();
  intBuffer.reserve(pixels.size());
  size_t tileSize = std::min<size_t>(std::min<size_t>(WIDTH, HEIGHT), 128);
  size_t totalTilesX = (WIDTH + tileSize - 1) / tileSize;
  size_t totalTilesY = (HEIGHT + tileSize - 1) / tileSize;
  size_t totalTiles = totalTilesX * totalTilesY;
  float msPerFrame = -0.0f; // for a more accurate "time remaining" estimate

  while (numFrames < FRAME_TOTAL) {
    size_t tileIndex = 0;
    for (size_t x = 0; x < WIDTH; x += tileSize) {
      for (size_t y = 0; y < HEIGHT; y += tileSize) {
        int w = std::min(tileSize, WIDTH - x);
        int h = std::min(tileSize, HEIGHT - y);

        size_t globalSize[2] = {(size_t)w, (size_t)h};
        size_t globalOffset[2] = {(size_t)x, (size_t)y};

        float percentDoneThisFrame = ((float)tileIndex / (float)totalTiles);
        float percentDone = ((float)numFrames / FRAME_TOTAL);
        percentDone += percentDoneThisFrame / FRAME_TOTAL;
        std::cout << "\rRendering frame " << (numFrames + 1) << " of " << FRAME_TOTAL << " (" << std::fixed << std::setprecision(2)
                  << (percentDone * 100.0f) << "%) ...";
        if (numFrames > 0 || x > 0 || y > 0) { // Ensure we are not on the first tile of the first frame
          // Calculate overall progress based on frames and tiles
          float timeRemaining;
          if (msPerFrame == -0.0f) {
            std::chrono::high_resolution_clock::time_point nowTime = std::chrono::high_resolution_clock::now();
            auto deltaTime = std::chrono::duration_cast<std::chrono::milliseconds>(nowTime - startTime);
            // no need to calculate msPerFrame since the entire purpose of that variable is to be used once a full
            // frame has actually been rendered
            timeRemaining = ((float)deltaTime.count() / (float)(numFrames + percentDoneThisFrame)) *
                            ((float)(FRAME_TOTAL - numFrames) - percentDoneThisFrame) / 1000.0f;
          } else {
            // We have an average! Yay!
            float totalFramesLeft = (FRAME_TOTAL - 2) - numFrames; // -2 because off-by-1 and because we are mid render
            float currentFrameLeft = 1.0f - percentDoneThisFrame;
            timeRemaining = ((totalFramesLeft + currentFrameLeft) * msPerFrame) / 1000.0f;
          }
          if (timeRemaining > 60.0f) {
            unsigned long minutes = (unsigned long)(timeRemaining / 60.0f);
            unsigned long seconds = (unsigned long)(timeRemaining) % 60;
            std::cout << " (time remaining: " << minutes << "m" << seconds << "s) ";
          } else {
            std::cout << " (time remaining: " << std::fixed << std::setprecision(2) << timeRemaining << "s) ";
          }
        }
        std::cout << std::flush;
        err = clSetKernelArg(kernel, 6, sizeof(CameraInformation), &camInfo);
        if (err != CL_SUCCESS) {
          std::cerr << "Failed to set kernel arg: " << err << "\n";
          return 1;
        }
        err = clSetKernelArg(kernel, 7, sizeof(cl_int), &numFrames);
        if (err != CL_SUCCESS) {
          std::cerr << "Failed to set kernel arg: " << err << "\n";
          return 1;
        }
        err = clEnqueueNDRangeKernel(queue, kernel, 2, globalOffset, globalSize, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
          std::cerr << "Failed to enqueue kernel: " << err << "\n";
          return 1;
        }
        tileIndex++;
        err = clFinish(queue); // safer than flush when reading back
        if (err != CL_SUCCESS) {
          std::cerr << "Failed to finish command queue: " << err << "\n";
          return 1;
        }
      }
    }
    err = clEnqueueReadBuffer(queue, imageBuffer, CL_TRUE, 0, pixels.size(), pixels.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
      std::cerr << "Failed to read buffer: " << err << "\n";
      return 1;
    }

    for (size_t i = 0; i < pixels.size(); ++i) {
      intBuffer[i] += pixels[i];
    }

    numFrames++;

    // Set the time per frame
    std::chrono::high_resolution_clock::time_point nowTime = std::chrono::high_resolution_clock::now();
    auto deltaTime = std::chrono::duration_cast<std::chrono::milliseconds>(nowTime - startTime);
    msPerFrame = (deltaTime.count()) / (float)(numFrames);

#ifdef RELAX_GPU
    if (numFrames == FRAME_TOTAL - 1)
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    if (FRAME_TOTAL > 10 && numFrames % 10 == 0 && numFrames < FRAME_TOTAL) {
      // Every 10 frames, output a quick preview to the screen
      std::cout << "\nWriting preview file..." << std::flush;
      std::vector<unsigned char> accumBuffer(pixels.size());
      for (size_t i = 0; i < pixels.size(); ++i) {
        accumBuffer[i] = static_cast<unsigned char>(intBuffer[i] / numFrames);
      }
      placeImageDataIntoBMP(accumBuffer, WIDTH, HEIGHT, "preview.bmp");
      std::cout << " done (" << numFrames / 10 << "/" << (FRAME_TOTAL / 10) << ")" << std::flush;
    }
#endif
  }
  std::chrono::high_resolution_clock::time_point endTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> deltaTime = endTime - startTime;
  unsigned long millisecondsPassed = std::chrono::duration_cast<std::chrono::milliseconds>(deltaTime).count();
  if (millisecondsPassed > 0)
    std::cout << "\rRendering frame " << FRAME_TOTAL << " of " << FRAME_TOTAL << " (100.00%) ... done in " << (millisecondsPassed / 1000.0)
              << " seconds" << std::endl;
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
