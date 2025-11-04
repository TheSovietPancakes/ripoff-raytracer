#pragma once

#include "clErrorCodes.hpp"
#include "math.hpp"
#include "readobj.hpp"
#include "settings.hpp"

#include <mutex>
#include <queue>

std::string loadKernelSource(/* const std::string& filename */) {
#include "kernelsource.hpp"
  return kernel_source;
}

struct Buffers {
  cl_mem triangleBuffer = nullptr;
  cl_mem meshBuffer = nullptr;
  cl_mem imageBuffer = nullptr;
  cl_mem nodeBuffer = nullptr;
};

struct KernelContext {
  cl_kernel kernel = nullptr;
  cl_context ctx = nullptr;
  cl_command_queue queue = nullptr;
  cl_program program = nullptr;
};

KernelContext generateKernelForDevice(cl_device_id device) {
  cl_int err;
  cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to create context: " << getCLErrorString(err) << "\n";
    exit(1);
  }
  cl_command_queue queue = clCreateCommandQueueWithProperties(ctx, device, nullptr, &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to create command queue: " << getCLErrorString(err) << "\n";
    exit(1);
  }
  std::string kernelSource = loadKernelSource();
  const char* data = kernelSource.data();
  cl_program program = clCreateProgramWithSource(ctx, 1, &data, nullptr, &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to create program: " << getCLErrorString(err) << "\n";
    exit(1);
  }
  const char* buildOptions = "-cl-fast-relaxed-math -cl-mad-enable"; // Fast math yay
  err = clBuildProgram(program, 1, &device, buildOptions, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to build program: " << getCLErrorString(err) << "\n";
    size_t logSize;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
    std::vector<char> log(logSize);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
    std::cerr << "Build log:\n" << log.data() << std::endl;
    exit(1);
  }
  cl_kernel kernel = clCreateKernel(program, "raytrace", &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to create kernel: " << getCLErrorString(err) << "\n";
    exit(1);
  }
  return KernelContext{
      .kernel = kernel,
      .ctx = ctx,
      .queue = queue,
      .program = program,
  };
}

void releaseKernelContext(KernelContext& kernel) {
  cl_int err;
  err = clReleaseKernel(kernel.kernel);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to release kernel: " << getCLErrorString(err) << std::endl;
    exit(1);
  }
  err = clReleaseProgram(kernel.program);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to release program: " << getCLErrorString(err) << std::endl;
    exit(1);
  }
  err = clReleaseCommandQueue(kernel.queue);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to release command queue: " << getCLErrorString(err) << std::endl;
    exit(1);
  }
  err = clReleaseContext(kernel.ctx);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to release context: " << getCLErrorString(err) << std::endl;
    exit(1);
  }
}

Buffers generateBuffers(std::vector<Triangle>& triangleList, std::vector<MeshInfo>& meshList, std::vector<Node>& nodeList, cl_context& ctx,
                        cl_kernel& kernel) {
  cl_int err;
  cl_mem triangleBuffer =
      clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, triangleList.size() * sizeof(Triangle), triangleList.data(), &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to create triangle buffer\n";
    exit(1);
  }
  cl_mem meshBuffer = clCreateBuffer(ctx, CL_MEM_COPY_HOST_PTR, meshList.size() * sizeof(MeshInfo), meshList.data(), &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to create mesh buffer\n";
    exit(1);
  }
  cl_mem imageBuffer = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, WIDTH * HEIGHT * sizeof(cl_uchar4), nullptr, &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to create image buffer\n";
    exit(1);
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
    exit(1);
  }
  err = clSetKernelArg(kernel, 8, sizeof(cl_mem), &nodeBuffer);
  if (err != CL_SUCCESS) {
    std::cerr << "failed to set kernel arg node buffer: " << getCLErrorString(err) << std::endl;
    exit(1);
  }

  // args (same order as kernel)
  cl_int meshCount = (cl_int)meshList.size();
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &meshBuffer);
  if (err != CL_SUCCESS) {
    std::cerr << "failed to set kernel arg mesh buffer: " << getCLErrorString(err) << std::endl;
    exit(1);
  }
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &triangleBuffer);
  if (err != CL_SUCCESS) {
    std::cerr << "failed to set kernel arg triangle buffer: " << getCLErrorString(err) << std::endl;
    exit(1);
  }
  err = clSetKernelArg(kernel, 2, sizeof(cl_int), &meshCount);
  if (err != CL_SUCCESS) {
    std::cerr << "failed to set kernel arg mesh count: " << getCLErrorString(err) << std::endl;
    exit(1);
  }
  err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &imageBuffer);
  if (err != CL_SUCCESS) {
    std::cerr << "failed to set kernel arg image buffer: " << getCLErrorString(err) << std::endl;
    exit(1);
  }
  const cl_int width = WIDTH; // you must pass a pointer into clSetKernelArg, meaning you have to pass an lvalue
  const cl_int height = HEIGHT;
  err = clSetKernelArg(kernel, 4, sizeof(cl_int), &width);
  if (err != CL_SUCCESS) {
    std::cerr << "failed to set kernel arg image width: " << getCLErrorString(err) << std::endl;
    exit(1);
  }
  err = clSetKernelArg(kernel, 5, sizeof(cl_int), &height);
  if (err != CL_SUCCESS) {
    std::cerr << "failed to set kernel arg image height: " << getCLErrorString(err) << std::endl;
    exit(1);
  }

  return {triangleBuffer, meshBuffer, imageBuffer, nodeBuffer};
}

cl_int tryFreeMemoryObject(cl_mem& memobj) {
  if (memobj) {
    cl_int err = clReleaseMemObject(memobj);
    memobj = nullptr;
    return err;
  }
  return CL_SUCCESS;
}

cl_int releaseBuffers(Buffers& buffers) {
  cl_int err = CL_SUCCESS;
  err = tryFreeMemoryObject(buffers.triangleBuffer);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to release triangle buffer: " << getCLErrorString(err) << "\n";
    return err;
  }
  err = tryFreeMemoryObject(buffers.meshBuffer);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to release mesh buffer: " << getCLErrorString(err) << "\n";
    return err;
  }
  err = tryFreeMemoryObject(buffers.imageBuffer);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to release image buffer: " << getCLErrorString(err) << "\n";
    return err;
  }
  err = tryFreeMemoryObject(buffers.nodeBuffer);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to release node buffer: " << getCLErrorString(err) << "\n";
    return err;
  }
  return err;
}

struct TileInformation {
  size_t tileSize;
  size_t totalTilesX;
  size_t totalTilesY;
  size_t totalTiles;
};

void renderTile(std::vector<unsigned char>& pixels, cl_kernel kernel, cl_command_queue& queue, Buffers& buffers, size_t tileX, size_t tileY,
                size_t tileSize, cl_uint frameIndex, size_t seed, std::mutex* pixelsMutex = nullptr) {
  // Render a single tile
  cl_int err;
  int w = std::min(tileSize, WIDTH - tileX);
  int h = std::min(tileSize, HEIGHT - tileY);

  size_t globalSize[2] = {(size_t)w, (size_t)h};
  size_t globalOffset[2] = {tileX, tileY};

  cl_int pRNGseed = (cl_int)(((frameIndex << 3) * (tileY * ((WIDTH + tileSize - 1) / tileSize) + tileX) * 0x857379583) * (seed << 8));
  err = clSetKernelArg(kernel, 7, sizeof(cl_int), &pRNGseed);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to set kernel arg PRNG seed: " << getCLErrorString(err) << std::endl;
    exit(1);
  }

  err = clEnqueueNDRangeKernel(queue, kernel, 2, globalOffset, globalSize, nullptr, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to enqueue kernel: " << getCLErrorString(err) << std::endl;
    exit(1);
  }

  err = clFinish(queue);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to finish command queue: " << getCLErrorString(err) << std::endl;
    exit(1);
  }

  // Read back the rendered tile
  std::vector<unsigned char> readIntoPixels(WIDTH * HEIGHT * 4, 0u);
  err = clEnqueueReadBuffer(queue, buffers.imageBuffer, CL_TRUE, 0, readIntoPixels.size(), readIntoPixels.data(), 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to read buffer: " << getCLErrorString(err) << std::endl;
    exit(1);
  }

  // Copy tile pixels to main pixel buffer
  if (pixelsMutex) {
    pixelsMutex->lock();
  }

  for (size_t pixelY = 0; pixelY < h; pixelY++) {
    for (size_t pixelX = 0; pixelX < w; pixelX++) {
      size_t globalX = tileX + pixelX;
      size_t globalY = tileY + pixelY;
      if (globalX >= WIDTH || globalY >= HEIGHT)
        continue;

      size_t pixelIndex = (globalY * WIDTH + globalX) * 4;
      pixels[pixelIndex + 0] = readIntoPixels[pixelIndex + 0]; // R
      pixels[pixelIndex + 1] = readIntoPixels[pixelIndex + 1]; // G
      pixels[pixelIndex + 2] = readIntoPixels[pixelIndex + 2]; // B
      pixels[pixelIndex + 3] = 255;                            // A
    }
  }

  if (pixelsMutex) {
    pixelsMutex->unlock();
  }
}

void multiThreadedCompute(size_t tileSize, std::vector<KernelContext>& deviceKernels, std::vector<unsigned char>& pixels,
                          std::vector<Buffers>& buffersList) {
  cl_int err;
  std::chrono::high_resolution_clock::time_point frameStartTime = std::chrono::high_resolution_clock::now();
  std::mutex pixelsMutex;
  std::mutex queueMutex;
  std::queue<std::pair<size_t, size_t>> tileQueue; // Queue of (tileX, tileY) pairs

  // Populate the tile queue with all tile coordinates
  for (size_t tileY = 0; tileY < HEIGHT; tileY += tileSize) {
    for (size_t tileX = 0; tileX < WIDTH; tileX += tileSize) {
      tileQueue.push({tileX, tileY});
    }
  }

  size_t tilesTotal = tileQueue.size();

  std::vector<std::thread> threads;

  // Launch threads for each device
  for (size_t i = 0; i < deviceKernels.size(); ++i) {
    threads.emplace_back([&, i]() {
      KernelContext& kernel = deviceKernels[i];
      Buffers& buffers = buffersList[i];
      size_t seed = i * 12345; // Different seed per device

      while (true) {
        std::pair<size_t, size_t> currentTile;
        {
          std::lock_guard<std::mutex> lock(queueMutex);
          if (tileQueue.empty()) {
            break;
          }
          currentTile = tileQueue.front();
          tileQueue.pop();
          // Each time a tile is popped off, we will print out the progress to the console
          std::chrono::high_resolution_clock::time_point nowTime = std::chrono::high_resolution_clock::now();
          std::chrono::duration timeElapsed = nowTime - frameStartTime;
          double percentCompleted = ((double)(tilesTotal - tileQueue.size()) / (double)tilesTotal) * 100.0;
          unsigned long millisecondsPassed = std::chrono::duration_cast<std::chrono::milliseconds>(timeElapsed).count();
          // Remaining Time = Time Passed * (1 / Percent Completed - 1)
          std::cout << "\033[2K\rRendering tile " << (tilesTotal - tileQueue.size()) << " of " << tilesTotal << " (" << percentCompleted << "%) "
                    << millisecondsPassed << "ms elapsed; " << (unsigned long)(millisecondsPassed * ((100.0 / percentCompleted) - 1.0))
                    << "ms remaining" << std::flush;
        }

        size_t tileX = currentTile.first;
        size_t tileY = currentTile.second;

        renderTile(pixels, kernel.kernel, kernel.queue, buffers, tileX, tileY, tileSize, 0, seed, &pixelsMutex);
        seed++; // Update seed for next tile
      }
    });
  }

  // Wait for all threads to complete
  for (auto& t : threads) {
    t.join();
  }

  std::chrono::high_resolution_clock::time_point frameEndTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration frameDuration = frameEndTime - frameStartTime;
  unsigned long frameMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(frameDuration).count();
  std::cout << "\rRendering tile " << tilesTotal << " of " << tilesTotal << " (100%) " << frameMilliseconds << " ms elapsed; 0 ms remaining"
            << std::endl;

  // Release buffers for each device
  for (size_t i = 0; i < deviceKernels.size(); ++i) {
    releaseBuffers(buffersList[i]);
  }
}

void singleThreadedCompute(size_t tileSize, KernelContext& deviceKernel, std::vector<unsigned char>& pixels, Buffers& buffers) {
  cl_int err;
  std::chrono::high_resolution_clock::time_point frameStartTime = std::chrono::high_resolution_clock::now();
  size_t tilesTotal = ((WIDTH + tileSize - 1) / tileSize) * ((HEIGHT + tileSize - 1) / tileSize);
  size_t tilesCompleted = 0;

  for (size_t tileY = 0; tileY < HEIGHT; tileY += tileSize) {
    for (size_t tileX = 0; tileX < WIDTH; tileX += tileSize) {
      renderTile(pixels, deviceKernel.kernel, deviceKernel.queue, buffers, tileX, tileY, tileSize, 0, 0);

      tilesCompleted++;
      std::chrono::high_resolution_clock::time_point nowTime = std::chrono::high_resolution_clock::now();
      std::chrono::duration timeElapsed = nowTime - frameStartTime;
      double percentCompleted = ((double)tilesCompleted / (double)tilesTotal) * 100.0;
      unsigned long millisecondsPassed = std::chrono::duration_cast<std::chrono::milliseconds>(timeElapsed).count();
      // Remaining Time = Time Passed * (1 / Percent Completed - 1)
      std::cout << "\033[2K\rRendering tile " << tilesCompleted << " of " << tilesTotal << " (" << percentCompleted << "%) " << millisecondsPassed
                << "ms elapsed; " << (unsigned long)(millisecondsPassed * ((100.0 / percentCompleted) - 1.0)) << "ms remaining" << std::flush;
    }
  }

  std::chrono::high_resolution_clock::time_point frameEndTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration frameDuration = frameEndTime - frameStartTime;
  unsigned long frameMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(frameDuration).count();
  std::cout << "\rRendering tile " << tilesTotal << " of " << tilesTotal << " (100%) " << frameMilliseconds << " ms elapsed; 0 ms remaining"
            << std::endl;

  // Release buffers
  releaseBuffers(buffers);
}

// This function is called before ever video frame starts rendering,
// allowing you to modify the scene (e.g. moving the model smoothly around the scene.)
void setupNextVideoFrame(CameraInformation& camInfo, int frameIndex) {
  // Simple scene: Increase 3d model's yaw
  const float anglePerFrame = (3.14159265359f * 2.0f) / (float)VIDEO_FRAME_COUNT;
  const float currentRotation = anglePerFrame * (float)frameIndex;
  MeshInfo& mesh = meshList.back();
  mesh.yaw = currentRotation + 5.5f; // Add 1.5f so if only rendering 1 frame, it starts out cool

  // HSV hsv = {
  //   .h = lerp(0, 360, (float)frameIndex / (float)VIDEO_FRAME_COUNT),
  //   .s = 0.75, // Not fully 1, so that colors aren't blown out (no "1-color" frames with no shadows)
  //   .v = 0.75
  // };
  // auto rgb = hsv2rgb(hsv);
  // mesh.material.color = {(float)rgb.r, (float)rgb.g, (float)rgb.b};
}

void addCornellBoxToScene(const MeshInfo& mesh) {
  // Add a light-emitting triangle underneath the dragon
  float minX = (nodeList[mesh.nodeIdx].bounds.min.s[0] * mesh.scale) - CORNELL_BREATHING_ROOM,
        maxX = (nodeList[mesh.nodeIdx].bounds.max.s[0] * mesh.scale) + CORNELL_BREATHING_ROOM;
  float minY = (nodeList[mesh.nodeIdx].bounds.min.s[1] * mesh.scale),
        maxY = (nodeList[mesh.nodeIdx].bounds.max.s[1] * mesh.scale) + CORNELL_BREATHING_ROOM; // do not sub so the model touches the floor
  float minZ = (nodeList[mesh.nodeIdx].bounds.min.s[2] * mesh.scale) - CORNELL_BREATHING_ROOM,
        maxZ = (nodeList[mesh.nodeIdx].bounds.max.s[2] * mesh.scale) + CORNELL_BREATHING_ROOM;

  // Floor (Y = minY)
  addQuad(cl_float3{minX, minY, minZ}, cl_float3{maxX, minY, minZ}, cl_float3{maxX, minY, maxZ}, cl_float3{minX, minY, maxZ}, cl_float3{0, 1, 0},
          {0});
  meshList.back().material = {
      .type = MaterialType_Solid,
      .ior = 1.0f,
      .color = {0.1, 0.1, 0.1},
      .emissionColor = {0, 0, 0},
      .emissionStrength = 0.0f,
      .reflectiveness = 0.0f,
      .specularProbability = 1.0f,
  };

  // Ceiling (Y = maxY)
  addQuad({minX, maxY, minZ}, {maxX, maxY, minZ}, {maxX, maxY, maxZ}, {minX, maxY, maxZ}, {0, -1, 0}, {1, 1, 1});

  // Front wall (Z = maxZ)
  addQuad({minX, minY, maxZ}, {maxX, minY, maxZ}, {maxX, maxY, maxZ}, {minX, maxY, maxZ}, {0, 0, -1}, {1.0f, 1.0f, 1.0f});
  meshList.back().material.type = MaterialType_OneSided; // invisible wall from back

  // Back wall (Z = minZ)
  // addQuad({minX, minY, minZ}, {maxX, minY, minZ}, {maxX, maxY, minZ}, {minX, maxY, minZ}, {0, 0, 1}, {0.1f, 0.1f, 0.1f});
  addQuad({minX, minY, minZ}, {maxX, minY, minZ}, {maxX, maxY, minZ}, {minX, maxY, minZ}, {0, 0, 1}, {0.1, 0.8, 0.1});

  // Left wall (X = minX) Blue
  addQuad({minX, minY, minZ}, {minX, minY, maxZ}, {minX, maxY, maxZ}, {minX, maxY, minZ}, {1, 0, 0}, {0.1f, 0.1f, 1.f});

  // Right wall (X = maxX) Red
  addQuad({maxX, minY, minZ}, {maxX, minY, maxZ}, {maxX, maxY, maxZ}, {maxX, maxY, minZ}, {-1, 0, 0}, {1.f, 0.2f, 0.2f});

  // Light quad on ceiling
  float lx = 50, lz = 50, ly = maxY - 1; // just below ceiling
  addQuad({-lx, ly, -lz}, {lx, ly, -lz}, {lx, ly, lz}, {-lx, ly, lz}, {0, -1, 0}, {0.0f, 0.0f, 0.0f});
  meshList.back().material = {.type = MaterialType_Solid,
                              .color = {1, 1, 1},
                              .emissionColor = {1.0f, 1.0f, 1.0f},
                              .emissionStrength = 8.0f,
                              .reflectiveness = 0.0f,
                              .specularProbability = 1};
}