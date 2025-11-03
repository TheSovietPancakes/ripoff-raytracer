// Render quality
#define BVHStackSize 64

// Math
#define tau 6.28318530717958647692f // 2pi
#define EPSILON 1e-6f               // Tiny number for finding precision errors
#define iorAir 1.0f

typedef struct {
  float3 min;
  float3 max;
} BoundingBox;

typedef struct {
  BoundingBox bounds;
  ulong index; // If this is a leaf node, then its the FIRST_TRI_INDEX.
               // otherwise it is CHILD_INDEX.
  ulong numTriangles;
} Node;

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
  MaterialType_Glassy = 3,
  MaterialType_OneSided = 4,
} MaterialType;

typedef struct {
  MaterialType type;
  float ior; // Index of Refraction; "How fast does light travel though me,
             // versus through air?""
  float3 color;
  float3 emissionColor;
  float emissionStrength;
  float reflectiveness;
  float specularProbability; // 0.0 = diffuse, 1.0 = perfect mirror
} RayTracingMaterial;

typedef struct {
  float3 origin;
  float3 direction;
  float3 invDir;
} Ray;

typedef struct {
  float3 posA, posB, posC;
  // normal
  float3 normalA, normalB, normalC;
} Triangle;

typedef struct {
  size_t nodeIdx;
  float3 pos;
  float pitch, yaw, roll;
  float scale;
  RayTracingMaterial material;
} MeshInfo;

typedef struct {
  bool didHit;
  float dst;
  float3 hitPoint;
  float3 normal;
  bool isBackface;
  RayTracingMaterial material;
} HitInfo;

typedef struct {
  float3 s0, s1, s2;
} float3x3;

float3 clamp3(float3 v, float minVal, float maxVal) {
  return fmin(fmax(v, (float3)(minVal)), (float3)(maxVal));
}

float3 lerp3(float3 a, float3 b, float t) { return a * (1.0f - t) + b * t; }

float lerp(float a, float b, float t) { return a * (1.0f - t) + b * t; }

float2 mod2(float2 x, float2 y) { return x - y * floor(x / y); }

inline float3x3 makeRotation(float pitch, float yaw, float roll) {
  float cx = native_cos(pitch), sx = native_sin(pitch);
  float cy = native_cos(yaw), sy = native_sin(yaw);
  float cz = native_cos(roll), sz = native_sin(roll);

  float3x3 result;
  result.s0 = (float3)(cy * cz, cy * sz, -sy);
  result.s1 = (float3)(cz * sy * sx - cx * sz, cx * cz + sx * sy * sz, cy * sx);
  result.s2 = (float3)(sx * sz + cx * cz * sy, cx * sy * sz - cz * sx, cx * cy);
  return result;
}

// helpers for float3x3 (works with your typedef: typedef struct { float3 s0,
// s1, s2; } float3x3;)

static inline float3 mul_mat_vec(const float3x3 m, const float3 v) {
  return (float3)(dot(m.s0, v), dot(m.s1, v), dot(m.s2, v));
}

static inline float3x3 transpose_mat(const float3x3 m) {
  // rows become columns
  float3x3 result;
  result.s0 = (float3)(m.s0.x, m.s1.x, m.s2.x);
  result.s1 = (float3)(m.s0.y, m.s1.y, m.s2.y);
  result.s2 = (float3)(m.s0.z, m.s1.z, m.s2.z);
  return result;
}

inline Ray WorldToLocalRay(Ray worldRay, float3x3 Rinv, float3 pos,
                           float scale) {
  float3 localOrigin = mul_mat_vec(Rinv, (worldRay.origin - pos));
  float3 localDirection = mul_mat_vec(Rinv, worldRay.direction);

  // Apply scale after rotation to maintain numerical stability
  if (fabs(scale) > EPSILON) {
    localOrigin /= scale;
    localDirection /= scale;
  }

  // fast_Normalize direction to maintain numerical stability
  localDirection = fast_normalize(localDirection);
  Ray r;
  r.invDir = 1.0f / localDirection;
  r.origin = localOrigin;
  r.direction = localDirection;

  return r;
}

inline HitInfo LocalToWorldHit(HitInfo localHit, float3x3 R, float3 pos,
                               float scale, Ray worldRay) {
  if (!localHit.didHit)
    return localHit;

  HitInfo worldHit = localHit;

  // Transform hit point back to world space
  worldHit.hitPoint = mul_mat_vec(R, localHit.hitPoint * scale) + pos;

  // Transform normal to world space (no translation needed for normals)
  worldHit.normal = fast_normalize(mul_mat_vec(R, localHit.normal));

  // Calculate correct world space distance
  worldHit.dst = length(worldHit.hitPoint - worldRay.origin);

  return worldHit;
}

inline float SafelyMapU32ToFloat(uint s) {
  // map 0..2^32-1 -> (0,1) using (s+1)/2^32 so we never get exactly 0 or 1
  return (float)(s + 1u) * (1.0f / 4294967296.0f);
}

float RandomValue(__private uint *state) {
  *state = *state * 747796405 + 2891336453;
  uint result = ((*state >> ((*state >> 28) + 4)) ^ *state) * 277803737;
  result = (result >> 22) ^ result;
  return SafelyMapU32ToFloat(result); // 2^32-1 ; 32-bit int limit
}

inline uint MakeSeed(uint pixelIndex, int frameIndex, uint rayIdx) {
  // simple mixing that stays within 32-bit arithmetic
  uint s = pixelIndex * 1664525u + (uint)frameIndex * 1013904223u;
  s ^= (rayIdx + 0x9E3779B9u);
  // do one LCG step to avoid zero seeds
  s = s * 22695477u + 1u;
  return s;
}

float RandomNormal(__private uint *state) {
  float u1 = RandomValue(state);
  float u2 = RandomValue(state);
  // ensure u1 is in (0,1]
  u1 = fmax(u1, EPSILON);
  float r = native_sqrt(-2.0f * native_log(u1));
  float theta = tau * u2;
  return r * native_cos(theta);
}

float3 RandomDirection(__private uint *state) {
  float x = RandomNormal(state);
  float y = RandomNormal(state);
  float z = RandomNormal(state);
  float3 v = fast_normalize((float3)(x, y, z));
  // guard against any NaN/infinite by fallback
  if (!isfinite(v.x) || !isfinite(v.y) || !isfinite(v.z)) {
    // fallback to deterministic direction
    v = (float3)(0.0f, 1.0f, 0.0f);
  }
  return v;
}

float3 RandomHemisphereDirection(float3 normal, __private uint *state) {
  float3 dir = RandomDirection(state);
  if (dot(dir, normal) < 0.0f)
    dir = -dir;
  return dir;
}

inline float rand01(__private uint *state) {
  *state = *state * 747796405u + 2891336453u;
  uint z = *state;
  // xor shift mash (cheap)
  z = (z ^ (z >> 16)) * 0x7feb352du;
  z = (z ^ (z >> 15)) * 0x846ca68bu;
  z = z ^ (z >> 16);
  return SafelyMapU32ToFloat(z);
}

float3 refract(float3 inDir, float3 normal, float iorA, float iorB) {
  float refractRatio = iorA / iorB;
  float cosAngleIn = -dot(inDir, normal);
  float sinSqrAngleOfRefraction =
      refractRatio * refractRatio * (1 - cosAngleIn * cosAngleIn);
  if (sinSqrAngleOfRefraction > 1)
    return 0; // was not refracted (total internal reflection, aka it just
              // reflected lmao)

  float3 refractDir =
      refractRatio * inDir +
      (refractRatio * cosAngleIn - sqrt(1 - sinSqrAngleOfRefraction)) * normal;
  return refractDir;
}

float3 reflect(float3 inDir, float3 normal) {
  return inDir - 2 * dot(inDir, normal) * normal;
}

float3 SampleHemisphereCosine(float3 n, __private uint *state) {
  float r1 = rand01(state);
  float r2 = rand01(state);

  // sample disk
  float r = native_sqrt(r1);
  float phi = tau * r2;
  float x = r * native_cos(phi);
  float y = r * native_sin(phi);
  float z = native_sqrt(fmax(0.0f, 1.0f - r1)); // z is "up" in tangent space

  // build orthonormal basis (n, t, b)
  float3 up = fabs(n.z) < 0.999f ? (float3)(0.0f, 0.0f, 1.0f)
                                 : (float3)(1.0f, 0.0f, 0.0f);
  float3 t = fast_normalize(cross(up, n));
  float3 b = cross(n, t);

  // map disk sample to hemisphere direction in world space (cosine-weighted)
  return fast_normalize(t * x + b * y + n * z);
}

bool RayBoundingBox(Ray ray, float3 boundsMin, float3 boundsMax,
                    __private float *outDist) {
  // Pitch, yaw, roll, and scale are handled outside this function
  // Do a normal cube AABB test
  // float3 invDir = 1.0f / ray.direction;
  float3 invDir = ray.invDir;
  float3 t0s = (boundsMin - ray.origin) * invDir;
  float3 t1s = (boundsMax - ray.origin) * invDir;
  float3 tsmaller = fmin(t0s, t1s);
  float3 tbigger = fmax(t0s, t1s);
  float tmin = fmax(fmax(tsmaller.x, tsmaller.y), tsmaller.z);
  float tmax = fmin(fmin(tbigger.x, tbigger.y), tbigger.z);
  if (outDist)
    *outDist = tmin;
  return tmax >= fmax(tmin, 0.0f);
}

inline HitInfo RayTriangle(Ray ray, const Triangle tri, bool cullBackface) {
  float3 edge1 = tri.posB - tri.posA;
  float3 edge2 = tri.posC - tri.posA;

  float3 h = cross(ray.direction, edge2);
  float a = dot(edge1, h);
  if (fabs(a) < EPSILON)
    return (HitInfo){.didHit = false};

  float f = 1.0f / a;
  float3 s = ray.origin - tri.posA;
  float u = f * dot(s, h);
  if (u < 0.0f || u > 1.0f)
    return (HitInfo){.didHit = false};

  float3 q = cross(s, edge1);
  float v = f * dot(ray.direction, q);
  if (v < 0.0f || u + v > 1.0f)
    return (HitInfo){.didHit = false};

  float t = f * dot(edge2, q);
  if (t <= EPSILON)
    return (HitInfo){.didHit = false};

  // Calculate the triangle's normal using the cross product of its edges
  float3 n = fast_normalize(tri.normalA * (1.0f - u - v) + tri.normalB * u +
                            tri.normalC * v);

  bool didHitBack = false;
  if (dot(ray.direction, n) > EPSILON) {
    if (cullBackface)
      return (HitInfo){.didHit = false}; // ignore backface hits
    // ray is inside (it hit the backface)
    didHitBack = true;
    n = -n; // flip normal to always point against the ray
  }
  return (HitInfo){.didHit = true,
                   .dst = t,
                   .hitPoint = ray.origin + ray.direction * t,
                   .normal = n,
                   .isBackface = didHitBack};
}

HitInfo RayTriangleBVH(int nodeIdx, Ray ray, __global const Node *nodeList,
                       __global const Triangle *triangles, bool cullBackface) {
  HitInfo closestHit;
  closestHit.didHit = false;
  closestHit.dst = INFINITY;

  typedef struct {
    int nodeIdx;
    float dist;
  } BVHStackEntry;

private
  BVHStackEntry stack[BVHStackSize];
private
  size_t stackPtr = 0;
  float distToRoot;
  if (!RayBoundingBox(ray, nodeList[nodeIdx].bounds.min,
                      nodeList[nodeIdx].bounds.max, &distToRoot)) {
    return closestHit; // no hit
  }
  // push root
  stack[stackPtr++] = (BVHStackEntry){.nodeIdx = nodeIdx, .dist = distToRoot};

  while (stackPtr > 0) {
    BVHStackEntry entry = stack[--stackPtr];
    Node node = nodeList[entry.nodeIdx];
    if (node.numTriangles == 0 && node.index == 0)
      continue; // invalid node

    if (entry.dist >= closestHit.dst)
      continue;

    if (node.numTriangles > 0) { // "if (isLeafNode)"
      for (ulong i = 0; i < node.numTriangles; ++i) {
        ulong triIdx = node.index + i;
        HitInfo hit = RayTriangle(ray, triangles[triIdx], cullBackface);
        if (hit.didHit && hit.dst < closestHit.dst) {
          closestHit = hit;
        }
      }
    } else {
      // If we are not a leaf node, then 'index' means the childIndex
      Node childA = nodeList[node.index];
      Node childB = nodeList[node.index + 1];

      float distA, distB;
      bool hitA =
          RayBoundingBox(ray, childA.bounds.min, childA.bounds.max, &distA);
      bool hitB =
          RayBoundingBox(ray, childB.bounds.min, childB.bounds.max, &distB);

      // Both hit, push the closer one first
      if (!hitB && !hitA)
        continue;
      if (!hitB && hitA) {
        if (distA < closestHit.dst) {
          stack[stackPtr++] = (BVHStackEntry){node.index, distA};
        }
        continue;
      }
      if (hitB && !hitA) {
        if (distB < closestHit.dst) {
          stack[stackPtr++] = (BVHStackEntry){node.index + 1, distB};
        }
        continue;
      }

      if (distA < distB) {
        stack[stackPtr++] = (BVHStackEntry){node.index + 1, distB};
        stack[stackPtr++] = (BVHStackEntry){node.index, distA};
      } else {
        stack[stackPtr++] = (BVHStackEntry){node.index, distA};
        stack[stackPtr++] = (BVHStackEntry){node.index + 1, distB};
      }
    }
  }
  // VIBE CODED WW
  return closestHit;
}

// reflectance value is 1 - this value
// thanks sebastian lague <3
float CalculateReflectance(float3 inDir, float3 normal, float iorA,
                           float iorB) {
  float refractRatio = iorA / iorB;
  float cosAngleIn = -dot(inDir, normal);
  if (cosAngleIn <= 0) return 1;
  float sinSqrAngleOfRefraction =
      refractRatio * refractRatio * (1 - cosAngleIn * cosAngleIn);
  if (sinSqrAngleOfRefraction >= 1)
    return 1; // was not refracted (total internal reflection, aka it just
              // reflected lmao)

  float cosAngleOfRefraction = sqrt(1 - sinSqrAngleOfRefraction);
  float denominatorPerpendicular =
      iorA * cosAngleIn + iorB * cosAngleOfRefraction;
  float denominatorParallel = iorA * cosAngleIn + iorB * cosAngleOfRefraction;

  if (min(denominatorPerpendicular, denominatorParallel) < EPSILON)
    return 1;

  // Perpendicular polarization
  float rPerpendicular = (iorA * cosAngleIn - iorB * cosAngleOfRefraction) /
                         denominatorPerpendicular;
  rPerpendicular *= rPerpendicular;
  // Parallel polarization
  float rParallel =
      (iorB * cosAngleIn - iorA * cosAngleOfRefraction) / denominatorParallel;
  rParallel *= rParallel;

  // Return the average of the perpendicular and parallel polarizations
  return (rPerpendicular + rParallel) / 2;
}

HitInfo CalculateRayCollisionWithTriangle(Ray worldRay,
                                          __global const MeshInfo *meshes,
                                          int meshCount,
                                          __global const Triangle *triangles,
                                          __global const Node *nodeList) {

  HitInfo closestHit;
  closestHit.dst = INFINITY;
  closestHit.didHit = false;

  for (int meshIdx = 0; meshIdx < meshCount; ++meshIdx) {
    MeshInfo info = meshes[meshIdx];

    // Skip degenerate meshes
    if (info.scale <= EPSILON)
      continue;

    // Build rotation matrices
    float3x3 R = makeRotation(info.pitch, info.yaw, info.roll);
    float3x3 Rinv =
        transpose_mat(R); // Inverse for orthonormal matrix is transpose

    // Transform ray to local space
    Ray localRay = WorldToLocalRay(worldRay, Rinv, info.pos, info.scale);

    // Perform intersection test in local space
    bool cullBackface = (info.material.type != MaterialType_Glassy &&
                         info.material.type != MaterialType_Invisible &&
                         info.material.type != MaterialType_OneSided);
    HitInfo localHit =
        RayTriangleBVH(info.nodeIdx, localRay, nodeList, triangles, cullBackface);

    if (localHit.didHit) {
      // Check if the ray hit the backface of a one-sided material
      if (info.material.type == MaterialType_OneSided && localHit.isBackface) {
        // Skip this hit; effectively, we never hit this triangle.
        continue;
      }
      // Transform hit back to world space
      HitInfo worldHit =
          LocalToWorldHit(localHit, R, info.pos, info.scale, worldRay);

      // Update closest hit if this is closer
      if (worldHit.dst < closestHit.dst) {
        worldHit.material = info.material;
        closestHit = worldHit;
      }
    }
  }

  return closestHit;
}

float3 Trace(Ray ray, __private uint *rngState, __global const MeshInfo *meshes,
             int meshCount, __global const Triangle *triangles,
             __global const Node *nodeList, uint maxBounceCount) {
  float3 incomingLight = (float3)(0.0f, 0.0f, 0.0f);
  float3 throughput = (float3)(1.0f, 1.0f, 1.0f);
  uint bounceCount = 0;
  while (bounceCount < maxBounceCount) {
    HitInfo hit = CalculateRayCollisionWithTriangle(ray, meshes, meshCount,
                                                    triangles, nodeList);

    if (!hit.didHit) {
      // environment / background contribution (if any)
      // incomingLight += throughput * (float3)(1);
      break;
    }
    if (hit.material.type == MaterialType_Invisible) {
      // Skip the hit; just go forward again
      ray.origin = hit.hitPoint + ray.direction * EPSILON;
      continue;
    }
    // One-sided will be checked in the CalculateRayCollisionWithTriangle
    // function. This way, we can continue the ray's path without any artifacts.
    if (hit.material.type == MaterialType_Checker) {
      float checkerSize =
          hit.material.emissionStrength; // size of each checker square

      // which checker cell we're in
      int xi = (int)floor(hit.hitPoint.x / checkerSize);
      int zi = (int)floor(hit.hitPoint.z / checkerSize);

      // alternate based on parity
      bool isEven = ((xi + zi) & 1) == 0;

      float3 checkerColor =
          isEven ? hit.material.color : hit.material.emissionColor;
      hit.material.color = checkerColor;
      hit.material.emissionStrength = 0.0f; // no emission for checker
    }
    if (hit.material.type == MaterialType_Glassy) {
      float iorCurrent = hit.isBackface ? hit.material.ior : iorAir;
      float iorNext = hit.isBackface ? iorAir : hit.material.ior;

      // Calculate reflection and refraction directions
      float3 reflectDir = reflect(ray.direction, hit.normal);
      float3 refractDir =
          refract(ray.direction, hit.normal, iorCurrent, iorNext);

      // Calculate the proportion of light [0, 1] that takes each path
      float reflectWeight =
          CalculateReflectance(ray.direction, hit.normal, iorCurrent, iorNext);
      float refractWeight = 1.0f - reflectWeight;

      // Randomly choose between reflection and refraction based on weights
      bool willReflect = rand01(rngState) < reflectWeight;

      // Update ray direction and origin
      ray.direction = willReflect ? reflectDir : refractDir;
      ray.origin = hit.hitPoint + EPSILON * hit.normal * sign(dot(hit.normal, ray.direction));

      // Adjust throughput to account for reflectance and transmittance
      throughput *= willReflect ? reflectWeight : refractWeight;
    }
    if (hit.material.type == MaterialType_Solid || 
        hit.material.type == MaterialType_Checker) {
      bool isSpecularBounce =
          hit.material.specularProbability >= RandomValue(rngState);

      ray.origin = hit.hitPoint + (hit.normal * EPSILON);
      float3 diffuseDir = normalize(hit.normal + RandomDirection(rngState));
      float3 specularDir = reflect(ray.direction, hit.normal);
      ray.direction =
          normalize(lerp3(diffuseDir, specularDir,
                          hit.material.reflectiveness * isSpecularBounce));

      // Update light info
      float3 emittedLight =
          hit.material.emissionColor * hit.material.emissionStrength;
    }

    // accumulate emission
    incomingLight += throughput * (hit.material.emissionColor *
                                   hit.material.emissionStrength);

    // pick next direction cosine-weighted
    ray.origin = hit.hitPoint +
                 ray.direction * EPSILON; // offset to avoid self-intersection
    // multiply throughput by surface color
    throughput *= hit.material.color;
    // Russian roulette (optional) - simple energy termination to save work
    float p = fmax(throughput.x, fmax(throughput.y, throughput.z));
    if (bounceCount > 3) {
      float q = fmax(0.05f, 1.0f - p);
      if (rand01(rngState) < q)
        break;
      throughput /= (1.0f - q);
    }
    bounceCount++;
  }
  return incomingLight;
}

Ray MakeRay(CameraInformation camInfo, float2 uv) {
  float2 ndc =
      (uv * 2.0f -
       (float2)(1.0f, 1.0f));    // Convert from UV [0 - 1] to NDC [-1 - 1]
  ndc[0] *= camInfo.aspectRatio; // Adjust for aspect ratio
  float scale = tan(radians(camInfo.fov * 0.5f));
  float3 rayDirCameraSpace =
      fast_normalize((float3)(ndc[0] * scale, ndc[1] * scale, 1.0f));
  float cx = native_cos(camInfo.pitch), sx = native_sin(camInfo.pitch);
  float cy = native_cos(camInfo.yaw), sy = native_sin(camInfo.yaw);
  float cz = native_cos(camInfo.roll), sz = native_sin(camInfo.roll);

  float3x3 rotation = {
      (float3)(cy * cz, cz * sy * sx - cx * sz, sx * sz + cx * cz * sy),
      (float3)(cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx),
      (float3)(-sy, cy * sx, cx * cy)};
  // Multiply matrices: rotation * rayDirCameraSpace
  float3 rayDirWorldSpace =
      fast_normalize((float3)(dot(rotation.s0, rayDirCameraSpace),
                              dot(rotation.s1, rayDirCameraSpace),
                              dot(rotation.s2, rayDirCameraSpace)));
  Ray ray;
  ray.origin = camInfo.position;
  ray.direction = rayDirWorldSpace;
  return ray;
}

__kernel void raytrace(__global const MeshInfo *meshes,
                       __global const Triangle *triangles, int meshCount,
                       __global uchar4 *image, int width, int height,
                       CameraInformation camInfo, int frameIndex,
                       __global const Node *nodeList, uint maxBounceCount,
                       uint incomingRaysPerPixel) {
  uint x = get_global_id(0);
  uint y = get_global_id(1);
  uint pixelIndex = y * width + x;
  uint rngState = MakeSeed(pixelIndex, frameIndex, 0);

  float2 uv =
      (float2)((float)x / (float)width, (float)(1.0f - y / (float)height));
  Ray ray = MakeRay(camInfo, uv);

  float3 accum = (float3)(0.0f, 0.0f, 0.0f);
  for (uint s = 0; s < incomingRaysPerPixel; ++s) {
    accum += Trace(ray, &rngState, meshes, meshCount, triangles, nodeList,
                   maxBounceCount);
  }
  float3 pixelColor = accum / (float)incomingRaysPerPixel;

  // gamma and clamp
  pixelColor = clamp(pixelColor, 0.0f, 1.0f);
  // optional gamma correction (approx)
  pixelColor = native_powr(pixelColor, (float3)(1.0f / 2.2f));

  image[pixelIndex] =
      (uchar4)((uchar)(pixelColor.x * 255.0f), (uchar)(pixelColor.y * 255.0f),
               (uchar)(pixelColor.z * 255.0f), 0);
}

__kernel void checkIntersectingRay(__global const MeshInfo *meshList,
                                   int meshCount,
                                   __global const Triangle *triangleList,
                                   __global const Node *nodeList,
                                   CameraInformation camInfo, float2 uv,
                                   __global int *outMeshIdx) {
  Ray ray = MakeRay(camInfo, uv);

  float closestDst = INFINITY;
  int closestMesh = -1;

  for (int meshIdx = 0; meshIdx < meshCount; ++meshIdx) {
    MeshInfo info = meshList[meshIdx];

    // Skip degenerate meshes
    if (info.scale <= EPSILON)
      continue;

    // Build rotation matrices
    float3x3 R = makeRotation(info.pitch, info.yaw, info.roll);
    float3x3 Rinv =
        transpose_mat(R); // Inverse for orthonormal matrix is transpose

    // Transform ray to local space
    Ray localRay = WorldToLocalRay(ray, Rinv, info.pos, info.scale);

    // Perform intersection test in local space
    HitInfo localHit =
        RayTriangleBVH(info.nodeIdx, localRay, nodeList, triangleList,
                       (info.material.type == MaterialType_OneSided));
    if (localHit.didHit) {
      // Transform hit back to world space
      HitInfo worldHit =
          LocalToWorldHit(localHit, R, info.pos, info.scale, ray);

      // Update closest hit if this is closer
      if (worldHit.dst < closestDst) {
        closestDst = worldHit.dst;
        closestMesh = meshIdx;
      }
    }
  }

  *outMeshIdx = closestMesh;
}