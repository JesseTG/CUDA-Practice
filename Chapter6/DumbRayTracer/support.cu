#include "support.hpp"
#include "common.hpp"

__constant__ Sphere cudaSpheresConst[NUM_SPHERES];

__host__ __device__ uchar3 computePixel(float x, float y, const Sphere * __restrict__ spheres,
                                        size_t numSpheres) {
  uchar3 result{0, 0, 0};

  for (size_t s = 0; s < numSpheres; ++s) {
    float n = 0;
    float t = spheres[s].hit(x, y, &n);

    if (t > -INFINITY) {
      // If we actually hit anything...
      result.x = spheres[s].r * n;
      result.y = spheres[s].g * n;
      result.z = spheres[s].b * n;
    }
  }

  return result;
}


__global__ static void _cudaRenderSceneConstMem(uint8_t * __restrict__ pixels, int2 dim,
                                        size_t numSpheres) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  uint64_t offset = x + y * dim.x;

  float ox = x - dim.x / 2;
  float oy = y - dim.y / 2;
  uchar3 pixel = computePixel(ox, oy, cudaSpheresConst, numSpheres);

  pixels[offset * 3 + 0] = pixel.x;
  pixels[offset * 3 + 1] = pixel.y;
  pixels[offset * 3 + 2] = pixel.z;
}
// NOTE: Use constant memory if all threads in a half-warp are referring to the
// same address

void cudaRenderSceneConstMem(uint8_t * __restrict__ pixels, int2 dim,
                                        size_t numSpheres, dim3 blocks, dim3 threadsPerBlock, const std::vector<Sphere>& spheres) {
  handle(cudaMemcpyToSymbol(cudaSpheresConst, spheres.data(),
                            spheres.size() * sizeof(Sphere)));

  _cudaRenderSceneConstMem<<<blocks, threadsPerBlock>>>(
      pixels, dim, spheres.size());
}
