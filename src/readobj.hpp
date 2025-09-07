#pragma once

#include <CL/cl.h>
#include <fstream>
#include <iostream>
#include <string>
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
  float3 boundsMin;
  float3 boundsMax;
  size_t firstTriangleIdx;
  size_t numTriangles;
} MeshCache;

typedef struct {
  float3 position;
  float pitch, yaw, roll;
  float fov;
  float aspectRatio;
} CameraInformation;

typedef struct {
  float3 color;
  float3 emissionColor;
  float emissionStrength;
} RayTracingMaterial;

typedef struct {
  float3 center;
  float radius;
  RayTracingMaterial material;
} Sphere;

typedef struct {
  float3 origin;
  float3 direction;
} Ray;

typedef struct {
  float3 posA, posB, posC;
  // normal
  float3 normalA, normalB, normalC;
} Triangle;

typedef struct {
  uint firstTriangleIdx = 0;
  uint numTriangles = 0;
  float3 boundsMin = {0.0f, 0.0f, 0.0f};
  float3 boundsMax = {0.0f, 0.0f, 0.0f};
  float3 pos = {0.0f, 0.0f, 0.0f};
  RayTracingMaterial material;
} MeshInfo;

typedef struct {
  bool didHit;
  float dst;
  float3 hitPoint;
  float3 normal;
  RayTracingMaterial material;
} HitInfo;

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
                  .material = {.color = {1.0f, 1.0f, 1.0f}, .emissionColor = {0.0f, 0.0f, 0.0f}, .emissionStrength = 0.0f}};
}