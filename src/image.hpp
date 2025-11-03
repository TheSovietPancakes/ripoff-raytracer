#pragma once

#include "clErrorCodes.hpp"
#include "math.hpp"
#include "readobj.hpp"
#include "settings.hpp"

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

void accumulateAndRenderFrame(std::vector<unsigned char>& pixels, cl_uint& numFrames, cl_int& err, cl_kernel kernel, CameraInformation& camInfo,
                              cl_command_queue& queue, Buffers& buffers, TileInformation info, size_t tilesToCompute, size_t tileStartPos = 0,
                              size_t seed = 0) {
  // Render the image. Enqueue the kernel 'FRAME_TOTAL' times and average the results.
  std::vector<unsigned char> intBuffer(pixels.size(), 0u);
  std::vector<unsigned char> readIntoPixels(pixels.size(), 0u);
  while (numFrames < FRAME_TOTAL) {
    size_t tileIndex = 0;
    for (size_t x = 0; x < WIDTH; x += info.tileSize) {
      for (size_t y = 0; y < HEIGHT; y += info.tileSize) {
        tileIndex++;
        if (tileIndex < tileStartPos || tileIndex > tileStartPos + tilesToCompute)
          continue;
        int w = std::min(info.tileSize, WIDTH - x);
        int h = std::min(info.tileSize, HEIGHT - y);

        size_t globalSize[2] = {(size_t)w, (size_t)h};
        size_t globalOffset[2] = {(size_t)x, (size_t)y};

        std::cout << std::flush;
        err = clSetKernelArg(kernel, 6, sizeof(CameraInformation), &camInfo);
        if (err != CL_SUCCESS) {
          std::cerr << "Failed to set kernel arg camera information: " << getCLErrorString(err) << std::endl;
          exit(1);
        }
        size_t pRNGseed =
            ((numFrames << 3) * (info.totalTiles << 1) + (tileIndex << 4) * 0x857379583) * (seed << 8); // A bunch of random nonsense to it
        err = clSetKernelArg(kernel, 7, sizeof(cl_int), &pRNGseed);
        if (err != CL_SUCCESS) {
          std::cerr << "Failed to set kernel arg num frames: " << getCLErrorString(err) << std::endl;
          exit(1);
        }
        err = clEnqueueNDRangeKernel(queue, kernel, 2, globalOffset, globalSize, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
          std::cerr << "Failed to enqueue kernel: " << getCLErrorString(err) << std::endl;
          exit(1);
        }
        err = clFinish(queue); // safer than flush when reading back
        if (err != CL_SUCCESS) {
          std::cerr << "Failed to finish command queue: " << getCLErrorString(err) << std::endl;
          exit(1);
        }
        // Move modified pixels into the buffer
        err = clEnqueueReadBuffer(queue, buffers.imageBuffer, CL_TRUE, 0, readIntoPixels.size(), readIntoPixels.data(), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
          std::cerr << "Failed to read buffer: " << getCLErrorString(err) << std::endl;
          exit(1);
        }
        // Use w, h, x, y, to find the pixels within this tile.
        for (size_t pixelX = 0; pixelX < w; pixelX++) {
          for (size_t pixelY = 0; pixelY < h; pixelY++) {
            // Find the X and Y position within the entire image based on the X and Y position of the current tile
            size_t globalX = (x * info.tileSize) + pixelX;
            size_t globalY = (y * info.tileSize) + pixelY;
            size_t pixelIndex = (globalY * WIDTH + globalX) * 4;
            if (globalX >= WIDTH || globalY >= HEIGHT)
              continue;
            // Accumulate into intBuffer
            intBuffer[pixelIndex + 0] = (unsigned char)(readIntoPixels[pixelIndex + 0]);
            intBuffer[pixelIndex + 1] = (unsigned char)(readIntoPixels[pixelIndex + 1]);
            intBuffer[pixelIndex + 2] = (unsigned char)(readIntoPixels[pixelIndex + 2]);
            // Alpha channel
            intBuffer[pixelIndex + 3] = 255;
          }
        }
      }
    }

    numFrames++;
  }
  // Average into pixels
  for (size_t i = 0; i < WIDTH * HEIGHT; i++) {
    pixels[i * 4 + 0] = (unsigned char)(intBuffer[i * 4 + 0] / numFrames);
    pixels[i * 4 + 1] = (unsigned char)(intBuffer[i * 4 + 1] / numFrames);
    pixels[i * 4 + 2] = (unsigned char)(intBuffer[i * 4 + 2] / numFrames);
    pixels[i * 4 + 3] = 255;
  }
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
          cl_float3{0.0f, 0.8f, 0.0f});
  meshList.back().material = {
      .type = MaterialType_Checker,
      .ior = 1.0f,
      .color = {0.05, 0.05, 0.05},
      .emissionColor = {0.1, 0.1, 0.1},
      .emissionStrength = 40.0f,
      .reflectiveness = 1.0f,
      .specularProbability = 1.0f,
  };

  // Ceiling (Y = maxY)
  addQuad({minX, maxY, minZ}, {maxX, maxY, minZ}, {maxX, maxY, maxZ}, {minX, maxY, maxZ}, {0, -1, 0}, {1, 1, 1});

  // Front wall (Z = maxZ)
  addQuad({minX, minY, maxZ}, {maxX, minY, maxZ}, {maxX, maxY, maxZ}, {minX, maxY, maxZ}, {0, 0, -1}, {1.0f, 1.0f, 1.0f});
  meshList.back().material.type = MaterialType_OneSided; // invisible wall from back

  // Back wall (Z = minZ)
  // addQuad({minX, minY, minZ}, {maxX, minY, minZ}, {maxX, maxY, minZ}, {minX, maxY, minZ}, {0, 0, 1}, {0.1f, 0.1f, 0.1f});
  addQuad({minX, minY, minZ}, {maxX, minY, minZ}, {maxX, maxY, minZ}, {minX, maxY, minZ}, {0, 0, 1}, {1.0f, 1.0f, 1.0f});
  meshList.back().material.reflectiveness = 0.9; // slightly less than a mirror

  // Left wall (X = minX) Blue
  addQuad({minX, minY, minZ}, {minX, minY, maxZ}, {minX, maxY, maxZ}, {minX, maxY, minZ}, {1, 0, 0}, {0.2f, 0.2f, 0.8f});

  // Right wall (X = maxX) Red
  addQuad({maxX, minY, minZ}, {maxX, minY, maxZ}, {maxX, maxY, maxZ}, {maxX, maxY, minZ}, {-1, 0, 0}, {0.8f, 0.2f, 0.2f});

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