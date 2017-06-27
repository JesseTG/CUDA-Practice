#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>

#include <CLI/CLI11.hpp>
#include <SDL2/SDL.h>
#include <cuda_runtime.h>

#include <random>

#include "common.hpp"

constexpr int NUM_SPHERES = 20;
using namespace std;

struct Sphere {
  float3 position;
  float radius;
  uint8_t r, g, b;

  __host__ __device__ float hit(float x, float y, float * __restrict__ n) const {
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

void cpuRenderScene(uint8_t * __restrict__ pixels, int2 dim, const Sphere * __restrict__ spheres,
                    size_t numSpheres) {

  for (int y = 0; y < dim.y; ++y) {
    float oy = y - dim.y / 2;

    for (int x = 0; x < dim.x; ++x) {
      uint64_t offset = x + y * dim.x;

      float ox = (x - dim.x / 2);

      uchar3 pixel = computePixel(ox, oy, spheres, numSpheres);

      pixels[offset * 3 + 0] = pixel.x;
      pixels[offset * 3 + 1] = pixel.y;
      pixels[offset * 3 + 2] = pixel.z;
    }
  }
}

__global__ void cudaRenderScene(uint8_t * __restrict__ pixels, int2 dim,
                                const Sphere * __restrict__ spheres, size_t numSpheres) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  uint64_t offset = x + y * dim.x;

  float ox = x - dim.x / 2;
  float oy = y - dim.y / 2;
  uchar3 pixel = computePixel(ox, oy, spheres, numSpheres);

  pixels[offset * 3 + 0] = pixel.x;
  pixels[offset * 3 + 1] = pixel.y;
  pixels[offset * 3 + 2] = pixel.z;
}

__global__ void cudaRenderSceneConstMem(uint8_t * __restrict__ pixels, int2 dim,
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

enum class Mode {
  CPU,
  CUDA_PTR,
  CUDA_CONSTMEM,
};

using random_type = minstd_rand;

int main(int argc, char *argv[]) {
  CLI::App app("Stuff");

  random_type::result_type seed = 0;

  float radiusMean = 70;
  float radiusStdDev = 20;
  float coordinateMean = 0;
  float coordinateStdDev = 300;
  int2 dim{640, 480};
  string modeString = "CPU";
  Mode mode = Mode::CPU;
  try {
    app.add_option("--seed", seed, "RNG seed", true);
    app.add_option("--radius-mean", radiusMean, "Mean sphere radius", true);
    app.add_option("--radius-stddev", radiusStdDev, "Radius variation", true);
    app.add_option("--coord-mean", coordinateMean, "Coordinate mean", true);
    app.add_option("--coord-stddev", coordinateStdDev, "Coordinate variation",
                   true);
    app.add_option("--width", dim.x, "Image width in pixels", true);
    app.add_option("--height", dim.y, "Image height in pixels", true);
    app.add_option("--mode", modeString,
                   "Rendering mode (CPU, CUDA_PTR, CUDA_CONSTMEM)", true)
        ->check([](const string &m) {
          return (m == "CPU" || m == "CUDA_PTR" || m == "CUDA_CONSTMEM");
        });

    app.parse(argc, argv);

    if (modeString == "CPU") {
      mode = Mode::CPU;
    } else if (modeString == "CUDA_PTR") {
      mode = Mode::CUDA_PTR;
    } else if (modeString == "CUDA_CONSTMEM") {
      mode = Mode::CUDA_CONSTMEM;
    }

  } catch (CLI::Error &e) {
    return app.exit(e);
  }

  random_type randomEngine(seed);
  dim3 threadsPerBlock(16, 16);
  dim3 blocks(dim.x / threadsPerBlock.x, dim.y / threadsPerBlock.y);

  // Initialization
  handleSdl(SDL_Init(SDL_INIT_VIDEO) == 0);

  SdlWindow window("", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, dim.x,
                   dim.y, SDL_WINDOW_HIDDEN);
  vector<uint8_t> pixels;
  vector<Sphere> spheres;

  pixels.resize(dim.x * dim.y * 3, 0);
  spheres.resize(NUM_SPHERES, {});

  uniform_int_distribution<int> color_distribution(0, 255);
  normal_distribution<float> radius_distribution(radiusMean, radiusStdDev);
  normal_distribution<float> position_distribution(coordinateMean,
                                                   coordinateStdDev);

  for (Sphere &s : spheres) {
    s.r = color_distribution(randomEngine);
    s.g = color_distribution(randomEngine);
    s.b = color_distribution(randomEngine);
    s.radius = radius_distribution(randomEngine);
    s.position.x = position_distribution(randomEngine);
    s.position.y = position_distribution(randomEngine);
    s.position.z = position_distribution(randomEngine);
  }

  float duration = 0.0; // milliseconds

  if (mode == Mode::CPU) {
    // CPU version
    auto cpuStart = chrono::high_resolution_clock::now();
    { cpuRenderScene(pixels.data(), dim, spheres.data(), spheres.size()); }
    auto cpuFinish = chrono::high_resolution_clock::now();

    duration = chrono::duration_cast<chrono::microseconds>(cpuFinish - cpuStart)
                   .count() /
               1000.0;
  } else if (mode == Mode::CUDA_PTR || mode == Mode::CUDA_CONSTMEM) {
    CudaEvent cudaStart;
    CudaEvent cudaFinish;

    // CUDA version
    CudaMemory<uint8_t> cudaResults(dim.x * dim.y * 3);
    CudaMemory<Sphere> cudaSpheres(NUM_SPHERES);
    handle(cudaMemcpy(cudaSpheres.ptr, spheres.data(),
                      spheres.size() * sizeof(Sphere),
                      cudaMemcpyHostToDevice));
    handle(cudaMemcpyToSymbol(cudaSpheresConst, spheres.data(),
                              spheres.size() * sizeof(Sphere)));

    handle(cudaEventRecord(cudaStart.event, 0));
    {

      if (mode == Mode::CUDA_PTR) {
        // If we're just passing the spheres to the kernel with a pointer...

        cudaRenderScene<<<blocks, threadsPerBlock>>>(
            cudaResults.ptr, dim, cudaSpheres.ptr, spheres.size());
      } else if (mode == Mode::CUDA_CONSTMEM) {

        cudaRenderSceneConstMem<<<blocks, threadsPerBlock>>>(
            cudaResults.ptr, dim, spheres.size());
      }

      handle(cudaGetLastError());

      handle(cudaMemcpy(pixels.data(), cudaResults.ptr, pixels.size(),
                        cudaMemcpyDeviceToHost));
    }
    handle(cudaEventRecord(cudaFinish.event, 0));
    handle(cudaEventSynchronize(cudaFinish.event));
    handle(cudaEventElapsedTime(&duration, cudaStart.event, cudaFinish.event));
  }

  {
    SDL_ShowWindow(window.window);
    handleSdl(SDL_UpdateTexture(window.texture, nullptr, pixels.data(),
                                3 * dim.x) == 0);
    handleSdl(SDL_RenderClear(window.renderer) == 0);
    handleSdl(
        SDL_RenderCopy(window.renderer, window.texture, nullptr, nullptr) == 0);
    SDL_RenderPresent(window.renderer);

    ostringstream windowTitle;
    windowTitle << "Ray-Tracing (" << modeString << ") " << duration << "ms";
    SDL_SetWindowTitle(window.window, windowTitle.str().c_str());
  }

  bool quit = false;
  SDL_Event event;
  while (!quit) {
    while (SDL_PollEvent(&event) != 0) {
      if (isWindowClosed(event)) {
        quit = true;
      }
    }

    SDL_Delay(33);
  }

  SDL_Quit();

  return 0;
}
