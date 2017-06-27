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

constexpr float PI = 3.1415926535897932f;

using namespace std;

__device__ __host__ uint8_t computePixel(int x, int y, float period) {
  return 255 * (sinf(x*2.0f*PI/ period) + 1.0f) * (sinf(y*2.0f*PI/ period) + 1.0f) / 4.0f;
}

__global__ void computeDotsCuda(uint8_t* pixels, int2 dim, float period) {
  __shared__ uint8_t shared[16][16];

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * dim.x;

  shared[threadIdx.x][threadIdx.y] = computePixel(x, y, period);

  __syncthreads();

  pixels[offset * 3 + 2] = shared[15 - threadIdx.x][15 - threadIdx.y];
}

__global__ void computeDotsCudaNoSync(uint8_t* pixels, int2 dim, float period) {
  __shared__ uint8_t shared[16][16];

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * dim.x;

  shared[threadIdx.x][threadIdx.y] = computePixel(x, y, period);

  pixels[offset * 3 + 2] = shared[15 - threadIdx.x][15 - threadIdx.y];
}

void computeDotsCpu(uint8_t* pixels, int2 dim, float period, bool omp) {
  #pragma omp parallel for if(omp)
  for (int y = 0; y < dim.y; ++y) {

    #pragma omp parallel for if(omp)
    for (int x = 0; x < dim.x; ++x) {
      uint64_t offset = x + y * dim.x;
      uint8_t value = computePixel(x, y, period);

      pixels[offset * 3 + 2] = value;
    }
  }
}

enum class Mode {
  CPU,
  OMP,
  CUDA,
  CUDA_NOSYNC
};

int main(int argc, char *argv[]) {
  CLI::App app("Stuff");

  dim3 threadsPerBlock { 16, 16 };
  int2 dim { 640, 480 };
  float period = 128.0;
  string modeString = "CPU";
  Mode mode = Mode::CPU;
  try {
    app.add_option("--period", period, "Dot period", true);
    app.add_option("--width", dim.x, "Image width in pixels", true);
    app.add_option("--height", dim.y, "Image height in pixels", true);
    app.add_option("--threadx", threadsPerBlock.x, "Horizontal threads per block", true);
    app.add_option("--thready", threadsPerBlock.y, "Vertical threads per block", true);
    app.add_flag("-n", "Don't initialize memory with zeroes");
    app.add_option("--mode", modeString, "Rendering mode (CPU, OMP, CUDA, CUDA_NOSYNC)", true)->check([](const string& m) {
      return (m == "CPU" || m == "OMP" || m == "CUDA" || m == "CUDA_NOSYNC");
    });

    app.parse(argc, argv);

    if (modeString == "CPU") {
      mode = Mode::CPU;
    }
    else if (modeString == "OMP") {
      mode = Mode::OMP;
    }
    else if (modeString == "CUDA") {
      mode = Mode::CUDA;
    }
    else if (modeString == "CUDA_NOSYNC") {
      mode = Mode::CUDA_NOSYNC;
    }

  } catch (CLI::Error &e) {
    return app.exit(e);
  }

  dim3 blocks(dim.x / threadsPerBlock.x, dim.y / threadsPerBlock.y);

  // Initialization
  handleSdl(SDL_Init(SDL_INIT_VIDEO) == 0);

  SdlWindow window("", SDL_WINDOWPOS_UNDEFINED,
                   SDL_WINDOWPOS_UNDEFINED, dim.x, dim.y, SDL_WINDOW_HIDDEN);
  vector<uint8_t> pixels;

  if (app.count("-n") == 0) {
    // If we want to zero the memory initially...
    pixels.resize(dim.x * dim.y * 3, 0);
  }
  else {
    pixels.reserve(dim.x * dim.y * 3);
  }

  float duration = 0.0;

  if (mode == Mode::CPU || mode == Mode::OMP) {
    // CPU version
    auto cpuStart = chrono::high_resolution_clock::now();
    { computeDotsCpu(pixels.data(), dim, period, mode == Mode::OMP); }
    auto cpuFinish = chrono::high_resolution_clock::now();

    duration =
        chrono::duration_cast<chrono::microseconds>(cpuFinish - cpuStart)
            .count() / 1000.0;
  }
  else if (mode == Mode::CUDA || mode == Mode::CUDA_NOSYNC) {
    CudaEvent cudaStart;
    CudaEvent cudaFinish;

    // CUDA version
    uint8_t *cudaResults = nullptr;
    handle(cudaMalloc(&cudaResults, dim.x * dim.y * 3 * sizeof(*cudaResults)));
    if (app.count("-n") == 0) {
      handle(cudaMemset(cudaResults, 0, dim.x * dim.y * 3));
    }
    handle(cudaEventRecord(cudaStart.event, 0));
    {
      if (mode == Mode::CUDA) {
        computeDotsCuda<<<blocks, threadsPerBlock>>>(cudaResults, dim, period);
      }
      else if (mode == Mode::CUDA_NOSYNC) {
        computeDotsCudaNoSync<<<blocks, threadsPerBlock>>>(cudaResults, dim, period);
      }

      handle(cudaMemcpy(pixels.data(), cudaResults, pixels.size(),
                        cudaMemcpyDeviceToHost));
    }
    handle(cudaEventRecord(cudaFinish.event, 0));
    handle(cudaEventSynchronize(cudaFinish.event));
    handle(
        cudaEventElapsedTime(&duration, cudaStart.event, cudaFinish.event));
    handle(cudaFree(cudaResults));
  }

  {
    SDL_ShowWindow(window.window);
    handleSdl(SDL_UpdateTexture(window.texture, nullptr, pixels.data(),
                                3 * dim.x) == 0);
    handleSdl(SDL_RenderClear(window.renderer) == 0);
    handleSdl(SDL_RenderCopy(window.renderer, window.texture, nullptr,
                             nullptr) == 0);
    SDL_RenderPresent(window.renderer);

    ostringstream windowTitle;
    windowTitle << "Blue Dots (" << modeString << ") " << duration << "ms";
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
