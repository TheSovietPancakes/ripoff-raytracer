#pragma once

#include <CL/cl.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

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
  uint firstTriangleIdx;
  uint numTriangles;
  float3 boundsMin;
  float3 boundsMax;
  RayTracingMaterial material;
} MeshInfo;

typedef struct {
  bool didHit;
  float dst;
  float3 hitPoint;
  float3 normal;
  RayTracingMaterial material;
} HitInfo;

// An inefficient algorithm to read the contents of a Wavefront OBJ file into a list of triangles.
void loadOBJFile(const std::string& filename, std::vector<Triangle>& triangles) {
  std::ifstream file(filename);
  if (!file) {
    std::cerr << "Failed to open OBJ file: " << filename << std::endl;
    std::cerr << "is open: " << file.is_open() << "; bad bit: " << file.bad() << "; fail bit: " << file.fail() << "; eof bit: " << file.eof() << std::endl;
    exit(1);
  }

  std::vector<cl_float3> temp_vertices;
  std::vector<cl_float3> temp_normals;

  std::string line;
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

      triangles.push_back(tri);
    }
  }
}
