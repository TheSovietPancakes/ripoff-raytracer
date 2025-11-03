#pragma once

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "math.hpp"
#include <functional>

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
  BoundingBox bounds;
  cl_ulong index = 0; // If this is a leaf node, then its the FIRST_TRI_INDEX. otherwise it is CHILD_INDEX.
  cl_ulong numTriangles = 0;
} GPUNode;

typedef struct {
  float3 position;
  float pitch, yaw, roll;
  float fov;
  float aspectRatio;
} CameraInformation;

typedef enum {
  MaterialType_Solid = 0,
  MaterialType_Checker = 1,
  MaterialType_Invisible = 2,
  MaterialType_Glassy = 3, // TODO
  MaterialType_OneSided = 4,
} MaterialType;

typedef struct {
  MaterialType type;
  float ior = 1.0f;
  float3 color;
  float3 emissionColor;
  float emissionStrength;
  float reflectiveness;
  float specularProbability; // 0.0 = diffuse, 1.0 = perfect mirror
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
  box.min.s[0] = std::min(box.min.s[0], point.s[0]);
  box.min.s[1] = std::min(box.min.s[1], point.s[1]);
  box.min.s[2] = std::min(box.min.s[2], point.s[2]);
  box.max.s[0] = std::max(box.max.s[0], point.s[0]);
  box.max.s[1] = std::max(box.max.s[1], point.s[1]);
  box.max.s[2] = std::max(box.max.s[2], point.s[2]);
}

void GrowToInclude(BoundingBox& box, const Triangle& tri) {
  GrowToInclude(box, tri.posA);
  GrowToInclude(box, tri.posB);
  GrowToInclude(box, tri.posC);
}

float3 CalculateTriangleCentroid(const Triangle& tri) { return (tri.posA + tri.posB + tri.posC) / 3.0f; }

typedef struct {
  int splitAxis;
  float splitPos;
  float cost;
} SplitAxisAndPos;

float NodeCost(float3 size, int numTriangles) {
  float halfArea = size.s[0] * (size.s[1] + size.s[2]) + size.s[1] * size.s[2];
  return halfArea * numTriangles;
}

float EvaluateSplit(const Node& node, int splitAxis, float splitPos) {
  BoundingBox boxA, boxB;
  int numInA = 0, numInB = 0;

  for (size_t i = node.firstTriangleIdx; i < node.firstTriangleIdx + node.numTriangles; i++) {
    float3 centroid = CalculateTriangleCentroid(triangleList[i]);
    if (centroid.s[splitAxis] < splitPos) {
      GrowToInclude(boxA, triangleList[i]);
      numInA++;
    } else {
      GrowToInclude(boxB, triangleList[i]);
      numInB++;
    }
  }

  return NodeCost(boxA.max - boxA.min, numInA) + NodeCost(boxB.max - boxB.min, numInB);
}

SplitAxisAndPos ChooseSplitAxisAndPosition(const Node& node) {
  const int NumTestsPerAxis = 5;
  float bestCost = CL_FLT_MAX;
  float bestPos = 0;
  int bestAxis = 0;

  for (int axis = 0; axis < 3; axis++) {
    float boundsStart = node.bounds.min.s[axis];
    float boundsEnd = node.bounds.max.s[axis];

    for (int i = 0; i < NumTestsPerAxis; i++) {
      float splitT = (i + 1.0f) / (NumTestsPerAxis + 1.0f);
      float pos = boundsStart + (boundsEnd - boundsStart) * splitT;
      float cost = EvaluateSplit(node, axis, pos);
      if (cost < bestCost) {
        bestCost = cost;
        bestPos = pos;
        bestAxis = axis;
      }
    }
  }
  return {bestAxis, bestPos, bestCost};

  // Use the midpoint along the longest axis as the split position
  // cl_float3 size = parent.bounds.max - parent.bounds.min;
  // int longestAxis = 0;
  // if (size.s[1] > size.s[0]) longestAxis = 1;
  // if (size.s[2] > size[longestAxis]) longestAxis = 2;
  // float splitPos = parent.bounds.min[longestAxis] + size[longestAxis] * 0.5f;

  // return {longestAxis, splitPos, 0.0f};
}

void PrintDebugBVH(size_t rootNodeIdx) {
  float averageTrianglesPerLeaf = 0.0f;
  size_t leafCount = 0;
  size_t internalNodeCount = 0;
  size_t maxDepth = 0;
  std::function<void(size_t, size_t)> recurse = [&](size_t nodeIdx, size_t depth) {
    if (nodeIdx >= nodeList.size()) {
      std::cerr << "Invalid node index: " << nodeIdx << "\n";
      return;
    }
    const Node& node = nodeList[nodeIdx];
    if (node.numTriangles > 0 && node.childIndex == 0) {
      // Leaf node
      leafCount++;
      averageTrianglesPerLeaf += node.numTriangles;
      if (depth > maxDepth)
        maxDepth = depth;
    } else {
      // Internal node
      internalNodeCount++;
      recurse(node.childIndex, depth + 1);
      recurse(node.childIndex + 1, depth + 1);
    }
  };
  recurse(rootNodeIdx, 1);
  if (leafCount > 0)
    averageTrianglesPerLeaf /= leafCount;
  std::cout << "BVH Stats: " << leafCount << " leaf nodes, " << internalNodeCount << " internal nodes, average " << averageTrianglesPerLeaf
            << " triangles per leaf, max depth " << maxDepth << "\n";
}

void SplitBVH(size_t parentIdx, int depth = 10) {
  Node& parent = nodeList[parentIdx];
  if (depth == 0 || parent.numTriangles <= 2)
    return;

  SplitAxisAndPos splitInfo = ChooseSplitAxisAndPosition(parent);
  if (splitInfo.cost >= NodeCost(parent.bounds.max - parent.bounds.min, parent.numTriangles)) {
    // We shouldn't split here, since the parent is better together than split
    return;
  }

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

  parent.numTriangles = 0; // now an internal node
  // Calculate bounding boxes for children
  for (size_t i = 0; i < childA.numTriangles; ++i) {
    GrowToInclude(childA.bounds, triangleList[childA.firstTriangleIdx + i]);
  }

  for (size_t i = 0; i < childB.numTriangles; ++i) {
    GrowToInclude(childB.bounds, triangleList[childB.firstTriangleIdx + i]);
  }

  nodeList.emplace_back(childA);
  nodeList.emplace_back(childB);

  // Recursively split children
  // Parent must be accessed via [index] here, as the vector
  // may have been reallocated with the above emplace_back calls
  SplitBVH(nodeList[parentIdx].childIndex, depth - 1);
  SplitBVH(nodeList[parentIdx].childIndex + 1, depth - 1);
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

  std::vector<float3> temp_vertices;
  std::vector<float3> temp_normals;

  std::string line;
  int triCount = 0;
  while (std::getline(file, line)) {
    if (line.empty())
      continue;

    if (line.substr(0, 2) == "v ") { // vertex
      float3 vertex;
      if (sscanf(line.c_str(), "v %f %f %f", &vertex.s[0], &vertex.s[1], &vertex.s[2]) == 3) {
        temp_vertices.emplace_back(vertex);
      }
    } else if (line.substr(0, 3) == "vn ") { // vertex normal
      float3 normal;
      if (sscanf(line.c_str(), "vn %f %f %f", &normal.s[0], &normal.s[1], &normal.s[2]) == 3) {
        temp_normals.emplace_back(normal);
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

      triangleList.emplace_back(tri);
    }
  }
  file.close();
  size_t firstTriangleIdx = triangleList.size() - (triCount);
  size_t numTriangles = triCount;
  Node c = {
      .childIndex = nodeList.size() + 1, // will be correct after pushing this here node
      .firstTriangleIdx = firstTriangleIdx,
      .numTriangles = numTriangles,
  };
  // Calculate bounds
  for (size_t tri = 0; tri < numTriangles; ++tri) {
    Triangle& t = triangleList[firstTriangleIdx + tri];
    c.bounds.min.s[0] = std::min(c.bounds.min.s[0], std::min(t.posA.s[0], std::min(t.posB.s[0], t.posC.s[0])));
    c.bounds.min.s[1] = std::min(c.bounds.min.s[1], std::min(t.posA.s[1], std::min(t.posB.s[1], t.posC.s[1])));
    c.bounds.min.s[2] = std::min(c.bounds.min.s[2], std::min(t.posA.s[2], std::min(t.posB.s[2], t.posC.s[2])));
    c.bounds.max.s[0] = std::max(c.bounds.max.s[0], std::max(t.posA.s[0], std::max(t.posB.s[0], t.posC.s[0])));
    c.bounds.max.s[1] = std::max(c.bounds.max.s[1], std::max(t.posA.s[1], std::max(t.posB.s[1], t.posC.s[1])));
    c.bounds.max.s[2] = std::max(c.bounds.max.s[2], std::max(t.posA.s[2], std::max(t.posB.s[2], t.posC.s[2])));
  }
  meshCaches[filename] = c;
  size_t rootIdx = nodeList.size();
  nodeList.emplace_back(c);
  // SplitBVH(rootIdx, 32);
  SplitBVH(rootIdx, 64);
  // PrintDebugBVH(rootIdx);
  return MeshInfo{
      .nodeIdx = rootIdx,
      .pitch = 0.0f,
      .yaw = 0.0f,
      .roll = 0.0f,
      .scale = 1.0f,
      .material = {.type = MaterialType_Solid, .color = {1.0f, 1.0f, 1.0f}, .emissionColor = {0.0f, 0.0f, 0.0f}, .emissionStrength = 0.0f}};
}

void addQuad(float3 a, float3 b, float3 c, float3 d, float3 normal, float3 color) {
  Node n =
      {
          .bounds =
              {
                  .min = {std::min(std::min(a.s[0], b.s[0]), std::min(c.s[0], d.s[0])), std::min(std::min(a.s[1], b.s[1]), std::min(c.s[1], d.s[1])),
                          std::min(std::min(a.s[2], b.s[2]), std::min(c.s[2], d.s[2]))},
                  .max = {std::max(std::max(a.s[0], b.s[0]), std::max(c.s[0], d.s[0])), std::max(std::max(a.s[1], b.s[1]), std::max(c.s[1], d.s[1])),
                          std::max(std::max(a.s[2], b.s[2]), std::max(c.s[2], d.s[2]))},
              },
          .childIndex = 0,
          .firstTriangleIdx = (cl_uint)triangleList.size(),
          .numTriangles = 2,
      };
  nodeList.emplace_back(n);
  SplitBVH(nodeList.size() - 1);
  MeshInfo quadMesh = {.nodeIdx = nodeList.size() - 1, // will be correct after SplitBVH
                       .material = {
                           .type = MaterialType_Solid,
                           //  .color = {0, 0, 0},
                           .color = color,
                           .emissionColor = {0, 0, 0},
                           .emissionStrength = 0.0f,
                           .reflectiveness = 1.0f,
                           .specularProbability = 1.0f,
                       }};

  // two triangles
  triangleList.emplace_back(Triangle{a, b, c, normal, normal, normal});
  triangleList.emplace_back(Triangle{a, c, d, normal, normal, normal});
  meshList.emplace_back(quadMesh);
};