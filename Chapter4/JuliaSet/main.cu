#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>

#define GLM_ENABLE_EXPERIMENTAL

#include <CLI/CLI11.hpp>
#include <SDL2/SDL.h>
#include <cuda_runtime.h>

#include "common.hpp"

using namespace std;

__host__ __device__ bool julia(int x, int y, float2 c, int w, int h, float scale,
                               int iterations, float threshold) {
  float2 j{scale * static_cast<float>(w / 2 - x) / (w / 2),
         scale * static_cast<float>(h / 2 - y) / (h / 2)};

  for (int i = 0; i < iterations; ++i) {
    j = {
        (j.x * j.x - j.y * j.y) + c.x, (j.y * j.x + j.x * j.y) + c.y,
    };

    if ((j.x * j.x + j.y * j.y) > threshold) {
      return false;
    }
  }

  return true;
}

void juliaCpu(uint8_t *pixels, float2 c, int2 dim, float scale, int iterations,
              float threshold, bool omp) {
  size_t len = dim.x * dim.y;

  #pragma omp parallel for if(omp)
  for (size_t i = 0; i < len; ++i) {
    pixels[i * 3] = julia(i % dim.x, i / dim.x, c, dim.x, dim.y, scale, iterations, threshold) ? 255 : 0;
  }
}

__global__ void juliaCuda(uint8_t *pixels, float2 c, float scale, int iterations,
                          float threshold) {
  int offset = blockIdx.x + blockIdx.y * gridDim.x;

  pixels[offset * 3] = julia(blockIdx.x, blockIdx.y, c, gridDim.x, gridDim.y,
                             scale, iterations, threshold)
                           ? 255
                           : 0;
}

int main(int argc, char *argv[]) {
  CLI::App app("Example program that generates a Julua fractal with both CUDA "
               "and the CPU");

  int2 dim { 640, 480 };
  float2 c { -0.8f, 0.156f };
  float scale = 1.5;
  int iterations = 200;
  float threshold = 1000.0;
  try {
    app.add_option("--real", c.x, "It does CX", true);
    app.add_option("--imag", c.y, "It does CY", true);
    app.add_option("--width", dim.x, "Image width in pixels", true);
    app.add_option("--height", dim.y, "Image height in pixels", true);
    app.add_option("--scale", scale, "Image scale", true);
    app.add_option("--iterations", iterations, "iterations", true);
    app.add_option("--threshold", threshold, "threshold stuff", true);
    app.add_flag("-n", "Don't initialize GPU memory with zeroes");
    app.parse(argc, argv);
  } catch (CLI::Error &e) {
    return app.exit(e);
  }

  // Initialization
  handleSdl(SDL_Init(SDL_INIT_VIDEO) == 0);

  SdlWindow cpuWindow("Julia Set (CPU)", SDL_WINDOWPOS_UNDEFINED,
                      SDL_WINDOWPOS_UNDEFINED, dim.x, dim.y, SDL_WINDOW_HIDDEN);

  SdlWindow ompWindow("Julia Set (OpenMP)", SDL_WINDOWPOS_UNDEFINED,
                       SDL_WINDOWPOS_UNDEFINED, dim.x, dim.y,
                       SDL_WINDOW_HIDDEN);

  SdlWindow cudaWindow("Julia Set (CUDA)", SDL_WINDOWPOS_UNDEFINED,
                       SDL_WINDOWPOS_UNDEFINED, dim.x, dim.y,
                       SDL_WINDOW_HIDDEN);

  CudaEvent cudaStart;
  CudaEvent cudaFinish;

  vector<uint8_t> pixelsCpu;
  pixelsCpu.resize(dim.x * dim.y * 3, 0);

  vector<uint8_t> pixelsOmp;
  pixelsOmp.resize(dim.x * dim.y * 3, 0);

  vector<uint8_t> pixelsCuda;
  pixelsCuda.resize(dim.x * dim.y * 3, 0);

  // CPU version
  auto cpuStart = chrono::high_resolution_clock::now();
  { juliaCpu(pixelsCpu.data(), c, dim, scale, iterations, threshold, false); }
  auto cpuFinish = chrono::high_resolution_clock::now();

  // OMP version
  auto ompStart = chrono::high_resolution_clock::now();
  { juliaCpu(pixelsOmp.data(), c, dim, scale, iterations, threshold, true); }
  auto ompFinish = chrono::high_resolution_clock::now();

  // CUDA version
  float cudaDuration = 0.0;
  uint8_t *cudaResults = nullptr;
  handle(cudaMalloc(&cudaResults, dim.x * dim.y * 3 * sizeof(*cudaResults)));
  if (app.count("-n") == 0) {
    handle(cudaMemset(cudaResults, 0, dim.x * dim.y * 3));
  }
  handle(cudaEventRecord(cudaStart.event, 0));
  {
    juliaCuda<<<dim3(dim.x, dim.y), 1>>>(cudaResults, c, scale, iterations,
                                         threshold);

    handle(cudaMemcpy(pixelsCuda.data(), cudaResults, pixelsCuda.size(),
                      cudaMemcpyDeviceToHost));
  }
  handle(cudaEventRecord(cudaFinish.event, 0));
  handle(cudaEventSynchronize(cudaFinish.event));
  handle(
      cudaEventElapsedTime(&cudaDuration, cudaStart.event, cudaFinish.event));
  handle(cudaFree(cudaResults));

  // Show the CPU version
  {
    SDL_ShowWindow(cpuWindow.window);
    handleSdl(SDL_UpdateTexture(cpuWindow.texture, nullptr, pixelsCpu.data(),
                                3 * dim.x) == 0);
    handleSdl(SDL_RenderClear(cpuWindow.renderer) == 0);
    handleSdl(SDL_RenderCopy(cpuWindow.renderer, cpuWindow.texture, nullptr,
                             nullptr) == 0);
    SDL_RenderPresent(cpuWindow.renderer);

    auto cpuDuration =
        chrono::duration_cast<chrono::microseconds>(cpuFinish - cpuStart)
            .count();

    ostringstream windowTitleCpu;
    windowTitleCpu << "Julia Set (CPU) " << cpuDuration / 1000.0 << "ms";
    SDL_SetWindowTitle(cpuWindow.window, windowTitleCpu.str().c_str());
  }

  // Show the OMP version
  {
    int x = 0;
    int y = 0;
    SDL_GetWindowPosition(cpuWindow.window, &x, &y);

    SDL_SetWindowPosition(ompWindow.window, static_cast<int>(x + 0.25 * x),
                          static_cast<int>(y + 0.25 * y));

    SDL_ShowWindow(ompWindow.window);
    handleSdl(SDL_UpdateTexture(ompWindow.texture, nullptr, pixelsOmp.data(),
                                3 * dim.x) == 0);
    handleSdl(SDL_RenderClear(ompWindow.renderer) == 0);
    handleSdl(SDL_RenderCopy(ompWindow.renderer, ompWindow.texture, nullptr,
                             nullptr) == 0);
    SDL_RenderPresent(ompWindow.renderer);

    auto cpuDuration =
        chrono::duration_cast<chrono::microseconds>(ompFinish - ompStart)
            .count();

    ostringstream windowTitleOmp;
    windowTitleOmp << "Julia Set (OpenMP) " << cpuDuration / 1000.0 << "ms";
    SDL_SetWindowTitle(ompWindow.window, windowTitleOmp.str().c_str());
  }

  // Show the CUDA version
  {
    int x = 0;
    int y = 0;
    SDL_GetWindowPosition(cpuWindow.window, &x, &y);

    SDL_SetWindowPosition(cudaWindow.window, static_cast<int>(x + 0.5 * x),
                          static_cast<int>(y + 0.5 * y));
    SDL_ShowWindow(cudaWindow.window);
    handleSdl(SDL_UpdateTexture(cudaWindow.texture, nullptr, pixelsCuda.data(),
                                3 * dim.x) == 0);
    handleSdl(SDL_RenderClear(cudaWindow.renderer) == 0);
    handleSdl(SDL_RenderCopy(cudaWindow.renderer, cudaWindow.texture, nullptr,
                             nullptr) == 0);
    SDL_RenderPresent(cudaWindow.renderer);

    ostringstream windowTitleCuda;
    windowTitleCuda << "Julia Set (CUDA) " << cudaDuration << "ms";
    SDL_SetWindowTitle(cudaWindow.window, windowTitleCuda.str().c_str());
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
