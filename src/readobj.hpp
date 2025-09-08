#pragma once

#include "CL/cl.h"
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using float3 = cl_float3;

std::ostream& operator<<(std::ostream& os, const float3& v) {
  os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
  return os;
}
using float4 = cl_float4;

// Commented out because float3 is just an alias of float4
// std::ostream& operator<<(std::ostream& os, const float4& v) {
//   os << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")";
//   return os;
// }

typedef struct {
  float3 boundsMin = {0.0f, 0.0f, 0.0f};
  float3 boundsMax = {0.0f, 0.0f, 0.0f};
  size_t firstTriangleIdx = 0;
  size_t numTriangles = 0;
} MeshCache;

typedef struct {
  float3 position = {0.0f, 0.0f, 0.0f};
  float pitch = 0.0f, yaw = 0.0f, roll = 0.0f;
  float fov = 45.0f;
  float aspectRatio = 16.0f / 9.0f;
} CameraInformation;

typedef enum { MaterialType_Solid = 0, MaterialType_Checker = 1, MaterialType_Invisible = 2 } MaterialType;

typedef struct {
  MaterialType type = MaterialType_Solid;
  float3 color = {1.0f, 1.0f, 1.0f};
  float3 emissionColor = {0.0f, 0.0f, 0.0f};
  float emissionStrength = 0.0f;
  float reflectiveness = 0.0f;
  // float specularProbability = 0.0f;
} RayTracingMaterial;

typedef struct {
  float3 posA = {0.0f, 0.0f, 0.0f};
  float3 posB = {0.0f, 0.0f, 0.0f};
  float3 posC = {0.0f, 0.0f, 0.0f};
  // normal
  float3 normalA = {0.0f, 0.0f, 0.0f};
  float3 normalB = {0.0f, 0.0f, 0.0f};
  float3 normalC = {0.0f, 0.0f, 0.0f};
} Triangle;

typedef struct {
  uint firstTriangleIdx = 0;
  uint numTriangles = 0;
  float3 boundsMin = {0.0f, 0.0f, 0.0f};
  float3 boundsMax = {0.0f, 0.0f, 0.0f};
  float3 pos = {0.0f, 0.0f, 0.0f};
  float pitch = 0.0f, yaw = 0.0f, roll = 0.0f;
  float scale = 1.0f;
  RayTracingMaterial material;
} MeshInfo;

std::unordered_map<std::string, MeshCache> meshCaches = {};
std::vector<MeshInfo> meshList = {};
std::vector<Triangle> triangleList = {};

// An inefficient algorithm to read the contents of a Wavefront OBJ file into a list of triangles.
MeshInfo loadMeshFromOBJFile(const std::string& filename) {
  if (meshCaches.find(filename) != meshCaches.end()) {
    size_t idx = meshCaches[filename].firstTriangleIdx;
    // Return a new mesh with triangle info pointing to the existing triangles
    return MeshInfo{.firstTriangleIdx = (cl_uint)idx,
                    .numTriangles = (cl_uint)meshCaches[filename].numTriangles,
                    .boundsMin = meshCaches[filename].boundsMin, // It is the user's responsibility to add the Mesh's position to this
                    .boundsMax = meshCaches[filename].boundsMax,
                    .material = {.color = {1.0f, 1.0f, 1.0f}, .emissionColor = {0.0f, 0.0f, 0.0f}, .emissionStrength = 0.0f}};
  }
  std::ifstream file(filename);
  if (!file) {
    std::cerr << "Failed to open OBJ file: " << filename << std::endl;
    std::cerr << "is open: " << file.is_open() << "; bad bit: " << file.bad() << "; fail bit: " << file.fail() << "; eof bit: " << file.eof()
              << std::endl;
    exit(1);
  }

  std::vector<cl_float3> temp_vertices;
  std::vector<cl_float3> temp_normals;

  std::string line;
  int triCount = 0;
  while (std::getline(file, line)) {
    if (line.empty())
      continue;

    if (line.substr(0, 2) == "v ") { // vertex
      cl_float3 vertex;
      if (sscanf(line.c_str(), "v %f %f %f", &vertex.x, &vertex.y, &vertex.z) == 3) {
        temp_vertices.push_back(vertex);
      }
    } else if (line.substr(0, 3) == "vn ") { // vertex normal
      cl_float3 normal;
      if (sscanf(line.c_str(), "vn %f %f %f", &normal.x, &normal.y, &normal.z) == 3) {
        temp_normals.push_back(normal);
      }
    } else if (line.substr(0, 2) == "f ") { // face
      cl_uint vIdx[3], vtIdx[3], nIdx[3];
      triCount++;

      int matches = sscanf(line.c_str(), "f %d/%d/%d %d/%d/%d %d/%d/%d", &vIdx[0], &vtIdx[0], &nIdx[0], &vIdx[1], &vtIdx[1], &nIdx[1], &vIdx[2],
                           &vtIdx[2], &nIdx[2]);

      if (matches != 9) {
        // handle case with no uvs: "f v//n v//n v//n"
        matches = sscanf(line.c_str(), "f %d//%d %d//%d %d//%d", &vIdx[0], &nIdx[0], &vIdx[1], &nIdx[1], &vIdx[2], &nIdx[2]);

        if (matches != 6) {
          std::cerr << "Unsupported face format: " << line << std::endl;
          continue;
        }
      }

      // Convert 1-based to 0-based
      for (int k = 0; k < 3; k++) {
        vIdx[k] -= 1;
        nIdx[k] -= 1;
      }

      // Bounds check
      if (vIdx[0] >= temp_vertices.size() || vIdx[1] >= temp_vertices.size() || vIdx[2] >= temp_vertices.size() || nIdx[0] >= temp_normals.size() ||
          nIdx[1] >= temp_normals.size() || nIdx[2] >= temp_normals.size()) {
        std::cerr << "Index out of bounds in face: " << line << std::endl;
        continue;
      }

      Triangle tri;
      tri.posA = temp_vertices[vIdx[0]];
      tri.posB = temp_vertices[vIdx[1]];
      tri.posC = temp_vertices[vIdx[2]];

      tri.normalA = temp_normals[nIdx[0]];
      tri.normalB = temp_normals[nIdx[1]];
      tri.normalC = temp_normals[nIdx[2]];

      triangleList.push_back(tri);
    }
  }
  file.close();
  size_t firstTriangleIdx = triangleList.size() - (triCount);
  size_t numTriangles = triCount;
  MeshCache c = {
      .firstTriangleIdx = firstTriangleIdx,
      .numTriangles = numTriangles,
  };
  // Calculate bounds
  for (size_t tri = 0; tri < numTriangles; ++tri) {
    Triangle& t = triangleList[firstTriangleIdx + tri];
    c.boundsMin.x = std::min(c.boundsMin.x, std::min(t.posA.x, std::min(t.posB.x, t.posC.x)));
    c.boundsMin.y = std::min(c.boundsMin.y, std::min(t.posA.y, std::min(t.posB.y, t.posC.y)));
    c.boundsMin.z = std::min(c.boundsMin.z, std::min(t.posA.z, std::min(t.posB.z, t.posC.z)));
    c.boundsMax.x = std::max(c.boundsMax.x, std::max(t.posA.x, std::max(t.posB.x, t.posC.x)));
    c.boundsMax.y = std::max(c.boundsMax.y, std::max(t.posA.y, std::max(t.posB.y, t.posC.y)));
    c.boundsMax.z = std::max(c.boundsMax.z, std::max(t.posA.z, std::max(t.posB.z, t.posC.z)));
  }
  meshCaches[filename] = c;
  return MeshInfo{.firstTriangleIdx = (cl_uint)firstTriangleIdx,
                  .numTriangles = (cl_uint)numTriangles,
                  .boundsMin = c.boundsMin, // It is the user's responsibility to add the Mesh's position to this
                  .boundsMax = c.boundsMax,
                  .pitch = 0.0f,
                  .yaw = 0.0f,
                  .roll = 0.0f,
                  .scale = 1.0f,
                  .material = {
                      .type = MaterialType_Solid,
                      .color = {1.0f, 1.0f, 1.0f},
                      .emissionColor = {0.0f, 0.0f, 0.0f},
                      .emissionStrength = 0.0f,
                      .reflectiveness = 0.0f,
                      // .specularProbability = 0.0f,
                  }};
}

void recalculateMeshBounds(MeshInfo& mesh) {
  // Reset bounds
  mesh.boundsMin = {CL_FLT_MAX, CL_FLT_MAX, CL_FLT_MAX};
  mesh.boundsMax = {-CL_FLT_MAX, -CL_FLT_MAX, -CL_FLT_MAX};

  // Precompute sine and cosine of rotation angles
  float cosYaw = cos(mesh.yaw);
  float sinYaw = sin(mesh.yaw);
  float cosPitch = cos(mesh.pitch);
  float sinPitch = sin(mesh.pitch);
  float cosRoll = cos(mesh.roll);
  float sinRoll = sin(mesh.roll);

  for (size_t tri = 0; tri < mesh.numTriangles; ++tri) {
    Triangle& t = triangleList[mesh.firstTriangleIdx + tri];
    cl_float3 vertices[3] = {t.posA, t.posB, t.posC};

    for (int i = 0; i < 3; ++i) {
      cl_float3 v = vertices[i];

      // Apply scaling
      v.x *= mesh.scale;
      v.y *= mesh.scale;
      v.z *= mesh.scale;

      // Apply rotation (Yaw-Pitch-Roll)
      // Yaw (around Y axis)
      float x1 = v.x * cosYaw - v.z * sinYaw;
      float z1 = v.x * sinYaw + v.z * cosYaw;
      v.x = x1;
      v.z = z1;

      // Pitch (around X axis)
      float y2 = v.y * cosPitch - v.z * sinPitch;
      float z2 = v.y * sinPitch + v.z * cosPitch;
      v.y = y2;
      v.z = z2;

      // Roll (around Z axis)
      float x3 = v.x * cosRoll - v.y * sinRoll;
      float y3 = v.x * sinRoll + v.y * cosRoll;
      v.x = x3;
      v.y = y3;

      // Apply translation
      v.x += mesh.pos.x;
      v.y += mesh.pos.y;
      v.z += mesh.pos.z;

      // Update bounds
      mesh.boundsMin.x = std::min(mesh.boundsMin.x, v.x);
      mesh.boundsMin.y = std::min(mesh.boundsMin.y, v.y);
      mesh.boundsMin.z = std::min(mesh.boundsMin.z, v.z);
      mesh.boundsMax.x = std::max(mesh.boundsMax.x, v.x);
      mesh.boundsMax.y = std::max(mesh.boundsMax.y, v.y);
      mesh.boundsMax.z = std::max(mesh.boundsMax.z, v.z);
    }
  }
}
