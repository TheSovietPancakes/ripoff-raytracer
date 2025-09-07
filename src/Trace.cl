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
  float3 pos;
  RayTracingMaterial material;
} MeshInfo;

typedef struct {
  bool didHit;
  float dst;
  float3 hitPoint;
  float3 normal;
  RayTracingMaterial material;
} HitInfo;

typedef struct {
  float3 s0, s1, s2;
} float3x3;

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
  u1 = fmax(u1, 1e-7f);
  float r = sqrt(-2.0f * log(u1));
  float theta = 2.0f * 3.14159265358979323846f * u2;
  return r * cos(theta);
}

float3 RandomDirection(__private uint *state) {
  float x = RandomNormal(state);
  float y = RandomNormal(state);
  float z = RandomNormal(state);
  float3 v = normalize((float3)(x, y, z));
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

HitInfo RayTriangle(Ray ray, Triangle tri, float3 offset) {
  const float EPSILON = 1e-6f;
  Triangle triangle = tri;
  tri.posA += offset;
  tri.posB += offset;
  tri.posC += offset;
  float3 edge1 = triangle.posB - triangle.posA;
  float3 edge2 = triangle.posC - triangle.posA;

  float3 h = cross(ray.direction, edge2);
  float a = dot(edge1, h);
  if (fabs(a) < EPSILON) {
    HitInfo miss = {.didHit = false};
    return miss; // Ray parallel to triangle
  }

  float f = 1.0f / a;
  float3 s = ray.origin - triangle.posA;
  float u = f * dot(s, h);
  if (u < 0.0f || u > 1.0f) {
    HitInfo miss = {.didHit = false};
    return miss;
  }

  float3 q = cross(s, edge1);
  float v = f * dot(ray.direction, q);
  if (v < 0.0f || u + v > 1.0f) {
    HitInfo miss = {.didHit = false};
    return miss;
  }

  float t = f * dot(edge2, q);
  if (t > EPSILON) {
    HitInfo hit;
    hit.didHit = true;
    hit.dst = t;
    hit.hitPoint = ray.origin + ray.direction * t;
    // Interpolate vertex normals
    float w = 1.0f - u - v;
    hit.normal = normalize(triangle.normalA * w + triangle.normalB * u + triangle.normalC * v);
    
    // Check if ray hits the back face - if so, treat as miss
    if (dot(ray.direction, hit.normal) > 0.0f) {
      HitInfo miss = {.didHit = false};
      return miss;
    }
    
    return hit;
  }

  HitInfo miss = {.didHit = false};
  return miss;
}

HitInfo RaySphere(Ray ray, float3 sphereCenter, float sphereRadius) {
  HitInfo hitInfo;
  hitInfo.didHit = false;
  float3 oc = ray.origin - sphereCenter;
  float a = dot(ray.direction, ray.direction);
  float b = 2.0f * dot(oc, ray.direction);
  float c = dot(oc, oc) - sphereRadius * sphereRadius;
  float discriminant = b * b - 4 * a * c;
  if (discriminant < 0) {
    return hitInfo; // No hit
  } else {
    float t = (-b - sqrt(discriminant)) / (2.0f * a);
    if (t > 0) {
      hitInfo.didHit = true;
      hitInfo.dst = t;
      hitInfo.hitPoint = ray.origin + t * ray.direction;
      hitInfo.normal = normalize(hitInfo.hitPoint - sphereCenter);
    }
    return hitInfo;
  }
}

HitInfo CalculateRayCollisionWithSphere(Ray ray, __global const Sphere *spheres,
                                        int sphereCount) {
  HitInfo closestHit;
  closestHit.dst = INFINITY;
  closestHit.didHit = false;
  for (int i = 0; i < sphereCount; i++) {
    Sphere sphere = spheres[i];
    HitInfo hit = RaySphere(ray, sphere.center, sphere.radius);
    if (hit.didHit &&
        (hit.dst < closestHit.dst || closestHit.didHit == false)) {
      closestHit = hit;
      closestHit.material = sphere.material;
      closestHit.didHit = true;
    }
  }
  return closestHit;
}

bool RayBoundingBox(Ray ray, float3 boxMin, float3 boxMax) {
  float3 invDir = 1.0f / ray.direction;
  float3 tMin = (boxMin - ray.origin) * invDir;
  float3 tMax = (boxMax - ray.origin) * invDir;

  float3 t1 = fmin(tMin, tMax);
  float3 t2 = fmax(tMin, tMax);

  float tNear = fmax(fmax(t1.x, t1.y), t1.z);
  float tFar = fmin(fmin(t2.x, t2.y), t2.z);

  return tNear <= tFar && tFar >= 0.0f;
}

HitInfo CalculateRayCollisionWithTriangle(Ray ray,
                                          __global const MeshInfo *meshes,
                                          int meshCount,
                                          __global const Triangle *triangles) {
  HitInfo closestHit;
  closestHit.dst = INFINITY;
  closestHit.didHit = false;
  for (int meshIdx = 0; meshIdx < meshCount; meshIdx++) {
    MeshInfo info = meshes[meshIdx];
    if (!RayBoundingBox(ray, info.boundsMin + info.pos, info.boundsMax + info.pos)) {
      continue;
    }

    for (int i = 0; i < info.numTriangles; i++) {
      int triIdx = info.firstTriangleIdx + i;
      Triangle tri = triangles[triIdx];

      HitInfo hit = RayTriangle(ray, tri, info.pos);
      if (hit.didHit &&
          (hit.dst < closestHit.dst || closestHit.didHit == false)) {
        closestHit = hit;
        closestHit.material = info.material;
        closestHit.didHit = true;
      }
    }
  }
  return closestHit;
}

Ray MakeRay(CameraInformation camInfo, float2 uv) {
  float2 ndc =
      (uv * 2.0f -
       (float2)(1.0f, 1.0f));    // Convert from UV [0 - 1] to NDC [-1 - 1]
  ndc[0] *= camInfo.aspectRatio; // Adjust for aspect ratio
  float scale = tan(radians(camInfo.fov * 0.5f));
  float3 rayDirCameraSpace =
      normalize((float3)(ndc[0] * scale, ndc[1] * scale, 1.0f));
  float cx = cos(camInfo.pitch), sx = sin(camInfo.pitch);
  float cy = cos(camInfo.yaw), sy = sin(camInfo.yaw);
  float cz = cos(camInfo.roll), sz = sin(camInfo.roll);

  float3x3 rotation = {
      (float3)(cy * cz, cz * sy * sx - cx * sz, sx * sz + cx * cz * sy),
      (float3)(cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx),
      (float3)(-sy, cy * sx, cx * cy)};
  // Multiply matrices: rotation * rayDirCameraSpace
  float3 rayDirWorldSpace =
      normalize((float3)(dot(rotation.s0, rayDirCameraSpace),
                         dot(rotation.s1, rayDirCameraSpace),
                         dot(rotation.s2, rayDirCameraSpace)));
  Ray ray;
  ray.origin = camInfo.position;
  ray.direction = rayDirWorldSpace;
  return ray;
}

float3 lerp3(float3 a, float3 b, float t) { return a * (1.0f - t) + b * t; }

float lerp(float a, float b, float t) { return a * (1.0f - t) + b * t; }

float3 Trace(Ray ray, __private uint *rngState, __global const MeshInfo *meshes,
             int meshCount, __global const Triangle *triangles) {
  float3 incomingLight = 0;
  float3 rayColor = 1;
  const uint MaxBounceCount = 30;
  for (uint i = 0; i <= MaxBounceCount; i++) {
    HitInfo hitInfo =
        CalculateRayCollisionWithTriangle(ray, meshes, meshCount, triangles);
    if (hitInfo.didHit) {
      ray.origin = hitInfo.hitPoint;
      ray.direction = normalize(hitInfo.normal + RandomDirection(rngState));

      RayTracingMaterial material = hitInfo.material;
      float3 emittedLight = material.emissionColor * material.emissionStrength;
      // float lightStrength = dot(hitInfo.normal, ray.direction);
      incomingLight += emittedLight * rayColor;
      rayColor *= material.color;
    } else {
      // incomingLight += rayColor * (float3)(0.2f, 0.4f, 0.6f);
      break;
    }
  }
  return incomingLight;
}

__kernel void raytrace(__global const MeshInfo *meshes,
                       __global const Triangle *triangles, int meshCount,
                       __global uchar4 *image, int width, int height,
                       CameraInformation camInfo, int frameIndex) {
  uint x = get_global_id(0);
  uint y = get_global_id(1);
  uint pixelIndex = y * width + x;

  float2 uv = (float2)((float)x / (float)width, (float)(1.0f - y / (float)height));
  Ray ray = MakeRay(camInfo, uv);

  const uint IncomingRaysPerPixel = 5;
  float3 accum = (float3)(0.0f, 0.0f, 0.0f);
  for (uint s = 0; s < IncomingRaysPerPixel; ++s) {
    // use different seed per sample
    // uint seed = MakeSeed(pixelIndex, frameIndex, s);
    // inside Trace we compute seed again but you can also pass seed pointer
    // if you prefer
    uint seed = MakeSeed(pixelIndex, frameIndex, s);
    __private uint rngState = seed;
    // accum += Trace(ray, &rngState, meshes, meshCount, triangles);
    accum += Trace(ray, &rngState, meshes, meshCount, triangles);
  }
  float3 pixelColor = accum / (float)IncomingRaysPerPixel;

  // gamma and clamp
  pixelColor = clamp(pixelColor, 0.0f, 1.0f);
  // optional gamma correction (approx)
  pixelColor = pow(pixelColor, (float3)(1.0f / 2.2f));

  image[pixelIndex] =
      (uchar4)((uchar)(pixelColor.x * 255.0f), (uchar)(pixelColor.y * 255.0f),
               (uchar)(pixelColor.z * 255.0f), 255u);
}