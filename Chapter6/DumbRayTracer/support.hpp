#ifndef SUPPORT_HPP
#define SUPPORT_HPP

#include "common.hpp"

constexpr int NUM_SPHERES = 20;

struct Sphere {
  float3 position;
  float radius;
  uint8_t r, g, b;

  __host__ __device__ float hit(float x, float y, float *__restrict__ n) const {
    float dx = x - position.x;
    float dy = y - position.y;

    if (dx * dx + dy * dy < radius * radius) {
      float dz = sqrtf(radius * radius - dx * dx - dy * dy);
      *n = dz / sqrtf(radius * radius);

      return dz + position.z;
    }

    return -INFINITY;
  }
};

__host__ __device__ uchar3 computePixel(float, float,
                                        const Sphere *__restrict__, size_t);

void cudaRenderSceneConstMem(uint8_t *__restrict__ pixels, int2 dim,
                             size_t numSpheres, dim3 blocks,
                             dim3 threadsPerBlock,
                             const std::vector<Sphere> &spheres);

#endif // SUPPORT_HPP
