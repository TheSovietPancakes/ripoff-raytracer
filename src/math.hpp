#pragma once

// surpress warnings
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
using float3 = cl_float3;
#include <fstream>
#include <string>
#include <vector>
#include <cmath>

struct HSV {
  double h, s, v; // Degrees, [0,1], [0,1]
};

struct RGB {
  double r, g, b; // [0,1], [0,1], [0,1]
};

// https://stackoverflow.com/questions/3018313/algorithm-to-convert-rgb-to-hsv-and-hsv-to-rgb-in-range-0-255-for-both
RGB hsv2rgb(HSV in) {
  double hh, p, q, t, ff;
  long i;
  RGB out;

  if (in.s <= 0.0) { // < is bogus, just shuts up warnings
    out.r = in.v;
    out.g = in.v;
    out.b = in.v;
    return out;
  }
  hh = in.h;
  if (hh >= 360.0)
    hh = 0.0;
  hh /= 60.0;
  i = (long)hh;
  ff = hh - i;
  p = in.v * (1.0 - in.s);
  q = in.v * (1.0 - (in.s * ff));
  t = in.v * (1.0 - (in.s * (1.0 - ff)));

  switch (i) {
  case 0:
    out.r = in.v;
    out.g = t;
    out.b = p;
    break;
  case 1:
    out.r = q;
    out.g = in.v;
    out.b = p;
    break;
  case 2:
    out.r = p;
    out.g = in.v;
    out.b = t;
    break;

  case 3:
    out.r = p;
    out.g = q;
    out.b = in.v;
    break;
  case 4:
    out.r = t;
    out.g = p;
    out.b = in.v;
    break;
  case 5:
  default:
    out.r = in.v;
    out.g = p;
    out.b = q;
    break;
  }
  return out;
}

// ---

std::ostream& operator<<(std::ostream& os, const float3& v) {
  os << "(" << v.s[0] << ", " << v.s[1] << ", " << v.s[2] << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const cl_float2& v) {
  os << "(" << v.s[0] << ", " << v.s[1] << ")";
  return os;
}

using float4 = cl_float4;

// Commented out because float3 is just an alias of float4
// std::ostream& operator<<(std::ostream& os, const float4& v) {
//   os << "(" << v.s[0] << ", " << v.s[1] << ", " << v.s[2] << ", " << v.w << ")";
//   return os;
// }

float3 operator+(const float3& a, const float3& b) { return {a.s[0] + b.s[0], a.s[1] + b.s[1], a.s[2] + b.s[2]}; }
float3 operator+=(float3& a, const float3& b) {
  a.s[0] += b.s[0];
  a.s[1] += b.s[1];
  a.s[2] += b.s[2];
  return a;
}
float3 operator/=(float3& a, const float b) {
  a.s[0] /= b;
  a.s[1] /= b;
  a.s[2] /= b;
  return a;
}
float3 operator-(const float3& a, const float3& b) { return {a.s[0] - b.s[0], a.s[1] - b.s[1], a.s[2] - b.s[2]}; }
float3 operator*(const float3& a, const float b) { return {a.s[0] * b, a.s[1] * b, a.s[2] * b}; }
float3 operator/(const float3& a, const float b) { return {a.s[0] / b, a.s[1] / b, a.s[2] / b}; }

float lerp(float a, float b, float f) { return a + f * (b - a); }

void placeImageDataIntoBMP(const std::vector<unsigned char>& pixels, int width, int height, const std::string& filename) {
  std::ofstream file(filename, std::ios::binary);
  if (!file)
    return;

  int padSize = (4 - (width * 3) % 4) % 4;
  int rowSize = 3 * width + padSize;
  int dataSize = rowSize * height;
  int fileSize = 54 + dataSize;

  // BMP Header
  unsigned char header[54] = {0};
  header[0] = 'B';
  header[1] = 'M';
  header[2] = fileSize & 0xFF;
  header[3] = (fileSize >> 8) & 0xFF;
  header[4] = (fileSize >> 16) & 0xFF;
  header[5] = (fileSize >> 24) & 0xFF;
  header[10] = 54;
  header[14] = 40;
  header[18] = width & 0xFF;
  header[19] = (width >> 8) & 0xFF;
  header[20] = (width >> 16) & 0xFF;
  header[21] = (width >> 24) & 0xFF;
  header[22] = height & 0xFF;
  header[23] = (height >> 8) & 0xFF;
  header[24] = (height >> 16) & 0xFF;
  header[25] = (height >> 24) & 0xFF;
  header[26] = 1;
  header[28] = 24;
  file.write(reinterpret_cast<char*>(header), 54);

  std::vector<unsigned char> padding(padSize, 0);

  for (int y = height - 1; y >= 0; --y) {
    for (int x = 0; x < width; ++x) {
      size_t i = (y * width + x) * 4;
      const char R = pixels[i + 0]; // .x
      const char G = pixels[i + 1]; // .y
      const char B = pixels[i + 2]; // .z
      file.write(const_cast<const char*>(&B), 1);
      file.write(const_cast<const char*>(&G), 1);
      file.write(const_cast<const char*>(&R), 1);
    }
    file.write(reinterpret_cast<char*>(padding.data()), padSize);
  }
  file.close();
}

float3 cross(const float3& a, const float3& b) {
  return {a.s[1] * b.s[2] - a.s[2] * b.s[1], a.s[2] * b.s[0] - a.s[0] * b.s[2], a.s[0] * b.s[1] - a.s[1] * b.s[0]};
}

float3 normalize(float3 a, float3 b, float3 c) {
  float3 edge1 = b - a;
  float3 edge2 = c - a;
  float3 n = cross(edge1, edge2);
  float len = sqrt(n.s[0] * n.s[0] + n.s[1] * n.s[1] + n.s[2] * n.s[2]);
  if (len > 0.0f) {
    return n / len;
  }
  return {0.0f, 0.0f, 0.0f};
}