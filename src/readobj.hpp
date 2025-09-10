#pragma once

#include <CL/cl.h>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
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

cl_float3 operator+(const cl_float3& a, const cl_float3& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
cl_float3 operator-(const cl_float3& a, const cl_float3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
cl_float3 operator*(const cl_float3& a, const cl_float3& b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
cl_float3 operator*(const cl_float3& a, float b) { return {a.x * b, a.y * b, a.z * b}; }
cl_float3 operator/(const cl_float3& a, float b) { return {a.x / b, a.y / b, a.z / b}; }

typedef struct {
  float3 min = {CL_FLT_MAX, CL_FLT_MAX, CL_FLT_MAX};
  float3 max = {CL_FLT_MIN, CL_FLT_MIN, CL_FLT_MIN};
} BoundingBox;

typedef struct {
  BoundingBox bounds;
  cl_ulong childIndex = 0;
  cl_ulong firstTriangleIdx = 0;
  cl_ulong numTriangles = 0;
} Node;

typedef struct {
  float3 position;
  float pitch, yaw, roll;
  float fov;
  float aspectRatio;
} CameraInformation;

typedef enum { MaterialType_Solid = 0, MaterialType_Checker = 1, MaterialType_Invisible = 2 } MaterialType;

typedef struct {
  MaterialType type;
  float3 color;
  float3 emissionColor;
  float emissionStrength;
  float reflectiveness;
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
  size_t nodeIdx;
  float3 pos = {0.0f, 0.0f, 0.0f};
  float pitch = 0.0f, yaw = 0.0f, roll = 0.0f;
  float scale = 1.0f;
  RayTracingMaterial material;
} MeshInfo;

typedef struct {
  bool didHit;
  float dst;
  float3 hitPoint;
  float3 normal;
  RayTracingMaterial material;
} HitInfo;

std::unordered_map<std::string, Node> meshCaches = {};
std::vector<MeshInfo> meshList = {};
std::vector<Triangle> triangleList = {};
std::vector<Node> nodeList = {};

void GrowToInclude(BoundingBox& box, const float3& point) {
  box.min.x = std::min(box.min.x, point.x);
  box.min.y = std::min(box.min.y, point.y);
  box.min.z = std::min(box.min.z, point.z);
  box.max.x = std::max(box.max.x, point.x);
  box.max.y = std::max(box.max.y, point.y);
  box.max.z = std::max(box.max.z, point.z);
}

void GrowToInclude(BoundingBox& box, const Triangle& tri) {
  GrowToInclude(box, tri.posA);
  GrowToInclude(box, tri.posB);
  GrowToInclude(box, tri.posC);
}

typedef struct {
  uint splitAxis;
  float splitPos;
} SplitAxisAndPos;

SplitAxisAndPos ChooseSplitAxisAndPosition(const Node& parent) {
  // Choose the axis with the largest extent
  float3 extents = parent.bounds.max - parent.bounds.min;
  uint axis = 0;
  if (extents.y > extents.x && extents.y >= extents.z)
    axis = 1;
  else if (extents.z > extents.x && extents.z >= extents.y)
    axis = 2;

  // Choose the split position as the midpoint along that axis
  float splitPos = 0.5f * (parent.bounds.min.s[axis] + parent.bounds.max.s[axis]);

  return {axis, splitPos};
}

float3 CalculateTriangleCentroid(const Triangle& tri) { return (tri.posA + tri.posB + tri.posC) / 3.0f; }

void SplitBVH(Node& parent, int depth = 10) {
  if (depth == 0 || parent.numTriangles <= 2)
    return;

  SplitAxisAndPos splitInfo = ChooseSplitAxisAndPosition(parent);

  // Partition triangles in-place
  size_t leftCount = 0;
  size_t rightStart = parent.firstTriangleIdx;
  size_t rightEnd = parent.firstTriangleIdx + parent.numTriangles - 1;

  // Partition triangles using two pointers approach
  while (rightStart <= rightEnd) {
    float3 centroid = CalculateTriangleCentroid(triangleList[rightStart]);
    bool isLeftSide = centroid.s[splitInfo.splitAxis] < splitInfo.splitPos;

    if (isLeftSide) {
      leftCount++;
      rightStart++;
    } else {
      // Swap with triangle at the end
      std::swap(triangleList[rightStart], triangleList[rightEnd]);
      rightEnd--;
    }
  }

  // Avoid degenerate splits
  if (leftCount == 0 || leftCount == parent.numTriangles) {
    return;
  }

  // Create child nodes
  parent.childIndex = nodeList.size();

  Node childA = {.childIndex = 0, .firstTriangleIdx = parent.firstTriangleIdx, .numTriangles = leftCount};

  Node childB = {.childIndex = 0, .firstTriangleIdx = parent.firstTriangleIdx + leftCount, .numTriangles = parent.numTriangles - leftCount};

  // Calculate bounding boxes for children
  for (size_t i = 0; i < childA.numTriangles; ++i) {
    GrowToInclude(childA.bounds, triangleList[childA.firstTriangleIdx + i]);
  }

  for (size_t i = 0; i < childB.numTriangles; ++i) {
    GrowToInclude(childB.bounds, triangleList[childB.firstTriangleIdx + i]);
  }

  nodeList.push_back(childA);
  nodeList.push_back(childB);

  // Recursively split children
  SplitBVH(nodeList[parent.childIndex], depth - 1);
  SplitBVH(nodeList[parent.childIndex + 1], depth - 1);
}

// An inefficient algorithm to read the contents of a Wavefront OBJ file into a list of triangles.
MeshInfo loadMeshFromOBJFile(const std::string& filename) {
  if (meshCaches.find(filename) != meshCaches.end()) {
    size_t idx = meshCaches[filename].firstTriangleIdx;
    // Return a new mesh with triangle info pointing to the existing triangles
    return MeshInfo{.nodeIdx = idx, .material = {.color = {1.0f, 1.0f, 1.0f}, .emissionColor = {0.0f, 0.0f, 0.0f}, .emissionStrength = 0.0f}};
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
  Node c = {
      .childIndex = 0,
      .firstTriangleIdx = firstTriangleIdx,
      .numTriangles = numTriangles,
  };
  // Calculate bounds
  for (size_t tri = 0; tri < numTriangles; ++tri) {
    Triangle& t = triangleList[firstTriangleIdx + tri];
    c.bounds.min.x = std::min(c.bounds.min.x, std::min(t.posA.x, std::min(t.posB.x, t.posC.x)));
    c.bounds.min.y = std::min(c.bounds.min.y, std::min(t.posA.y, std::min(t.posB.y, t.posC.y)));
    c.bounds.min.z = std::min(c.bounds.min.z, std::min(t.posA.z, std::min(t.posB.z, t.posC.z)));
    c.bounds.max.x = std::max(c.bounds.max.x, std::max(t.posA.x, std::max(t.posB.x, t.posC.x)));
    c.bounds.max.y = std::max(c.bounds.max.y, std::max(t.posA.y, std::max(t.posB.y, t.posC.y)));
    c.bounds.max.z = std::max(c.bounds.max.z, std::max(t.posA.z, std::max(t.posB.z, t.posC.z)));
  }
  meshCaches[filename] = c;
  size_t rootIdx = nodeList.size();
  nodeList.push_back(c);
  // SplitBVH(rootIdx, 32);
  SplitBVH(nodeList[rootIdx], 64);
  return MeshInfo{
      .nodeIdx = rootIdx,
      .pitch = 0.0f,
      .yaw = 0.0f,
      .roll = 0.0f,
      .scale = 1.0f,
      .material = {.type = MaterialType_Solid, .color = {1.0f, 1.0f, 1.0f}, .emissionColor = {0.0f, 0.0f, 0.0f}, .emissionStrength = 0.0f}};
}