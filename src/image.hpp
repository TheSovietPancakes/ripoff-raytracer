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

void accumulateAndRenderFrame(std::vector<unsigned char>& pixels, cl_uint& numFrames, cl_int& err, cl_kernel kernel, CameraInformation& camInfo,
                              cl_command_queue& queue, Buffers& buffers, const std::string& outputLocation, size_t seed = 0) {
  // Render the image. Enqueue the kernel 'FRAME_TOTAL' times and average the results.
  static std::vector<uint32_t> intBuffer(pixels.size(), 0u);
  std::chrono::high_resolution_clock::time_point startTime = std::chrono::high_resolution_clock::now();
  size_t tileSize = std::min<size_t>(std::min<size_t>(WIDTH, HEIGHT), TILE_SIZE);
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
          std::cerr << "Failed to set kernel arg camera information: " << getCLErrorString(err) << std::endl;
          exit(1);
        }
        size_t pRNGseed = ((numFrames << 3) * (totalTiles << 1) + (tileIndex << 4) * 0x857379583) * (seed << 8); // A bunch of random nonsense to it
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
        tileIndex++;
        err = clFinish(queue); // safer than flush when reading back
        if (err != CL_SUCCESS) {
          std::cerr << "Failed to finish command queue: " << getCLErrorString(err) << std::endl;
          exit(1);
        }
      }
    }
    err = clEnqueueReadBuffer(queue, buffers.imageBuffer, CL_TRUE, 0, pixels.size(), pixels.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
      std::cerr << "Failed to read buffer: " << getCLErrorString(err) << std::endl;
      exit(1);
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
  placeImageDataIntoBMP(finalImg, WIDTH, HEIGHT, outputLocation);
  std::cout << "Wrote " << outputLocation << " with " << WIDTH << "x" << HEIGHT << " resolution." << std::endl;
  finalImg.clear();
  finalImg.resize(WIDTH * HEIGHT * 4, 0);
  intBuffer.clear();
  intBuffer.resize(pixels.size(), 0u);
  pixels.clear();
  pixels.resize(WIDTH * HEIGHT * 4, 0);
  numFrames = 0;
}

// This function is called before ever video frame starts rendering,
// allowing you to modify the scene (e.g. moving the model smoothly around the scene.)
void setupNextVideoFrame(CameraInformation& camInfo, int frameIndex) {
  // Simple scene: Increase 3d model's yaw
  const float anglePerFrame = (3.14159265359f * 2.0f) / (float)VIDEO_FRAME_COUNT;
  const float currentRotation = anglePerFrame * (float)frameIndex;
  MeshInfo& mesh = meshList.back();
  mesh.yaw = currentRotation + 1.5f; // Add 1.5f so if only rendering 1 frame, it starts out cool

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
      // .color = {0.05, 0.22, 0.05},
      // .emissionColor = {0.075, 0.4, 0.075},
      .color = {0.07, 0.07, 0.07},
      .emissionColor = {0.1, 0.1, 0.1},
      .emissionStrength = 40.0f,
      .reflectiveness = 0.7f,
      .specularProbability = 1.0f,
  };

  // Ceiling (Y = maxY)
  addQuad({minX, maxY, minZ}, {maxX, maxY, minZ}, {maxX, maxY, maxZ}, {minX, maxY, maxZ}, {0, -1, 0}, {1, 1, 1});

  // Back wall (Z = maxZ)
  addQuad({minX, minY, maxZ}, {maxX, minY, maxZ}, {maxX, maxY, maxZ}, {minX, maxY, maxZ}, {0, 0, -1}, {1, 1, 1});

  // Front wall (Z = minZ)
  addQuad({minX, minY, minZ}, {maxX, minY, minZ}, {maxX, maxY, minZ}, {minX, maxY, minZ}, {0, 0, 1}, {1, 1, 1});
  // meshList.back().material.reflectiveness = 0.9; // slightly less than a mirror

  // Left wall (X = minX)
  addQuad({minX, minY, minZ}, {minX, minY, maxZ}, {minX, maxY, maxZ}, {minX, maxY, minZ}, {1, 0, 0}, {0.2f, 0.2f, 0.4});

  // Right wall (X = maxX)
  addQuad({maxX, minY, minZ}, {maxX, minY, maxZ}, {maxX, maxY, maxZ}, {maxX, maxY, minZ}, {-1, 0, 0}, {0.4f, 0.2f, 0.2f});

  // Light quad on ceiling
  float lx = 50, lz = 50, ly = maxY - 1; // just below ceiling
  addQuad({-lx, ly, -lz}, {lx, ly, -lz}, {lx, ly, lz}, {-lx, ly, lz}, {0, -1, 0}, {0.0f, 0.0f, 0.0f});
  meshList.back().material = {.type = MaterialType_Solid,
                              .color = {1, 1, 1},
                              .emissionColor = {1.0f, 1.0f, 1.0f},
                              .emissionStrength = 5.0f,
                              .reflectiveness = 0.0f,
                              .specularProbability = 1};
}