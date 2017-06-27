#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>

#define GL_GLEXT_PROTOTYPES

#include <CLI/CLI11.hpp>
#include <GL/glew.h>
#include <SDL2/SDL.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include "common.hpp"

using namespace std;

__host__ __device__ uint8_t computePixel(int x, int y, uint2 dim) {
  float fx = (x / static_cast<float>(dim.x)) - 0.5f;
  float fy = (y / static_cast<float>(dim.y)) - 0.5f;

  return static_cast<uint8_t>(128 + 127 * sin(abs(fx * 100) - abs(fy * 100)));
}

__global__ void drawCuda(uchar3 *pixels, uint2 dim) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * dim.x;

  pixels[offset].x = computePixel(x, y, dim);
  pixels[offset].y = 0;
  pixels[offset].z = 0;
}

void drawCpu(uchar3* pixels, uint2 dim, bool omp) {
  size_t len = dim.x * dim.y;

  #pragma omp parallel for if(omp)
  for (size_t i = 0; i < len; ++i) {
    pixels[i].x = computePixel(i % dim.x, i / dim.x, dim);
    pixels[i].y = 0;
    pixels[i].z = 0;
  }
}

enum class Mode {
  CPU,
  OMP,
  CUDA,
  CUDA_GL,
};

int main(int argc, char *argv[]) {
  CLI::App app("Draw a static ripple image");

  uint2 dim{640, 480};
  string modeString = "CUDA";
  Mode mode = Mode::CUDA;
  try {
    app.add_option("--width", dim.x, "Image width in pixels", true);
    app.add_option("--height", dim.y, "Image height in pixels", true);
    app.add_option("--mode", modeString,
                   "Rendering mode (CPU, OMP, CUDA, CUDA_GL)", true)
        ->check([](const string &m) {
          return (m == "CPU" || m == "OMP" || m == "CUDA" || m == "CUDA_GL");
        });

    app.parse(argc, argv);

    if (modeString == "CPU") {
      mode = Mode::CPU;
    } else if (modeString == "OMP") {
      mode = Mode::OMP;
    } else if (modeString == "CUDA") {
      mode = Mode::CUDA;
    } else if (modeString == "CUDA_GL") {
      mode = Mode::CUDA_GL;
    }

  } catch (CLI::Error &e) {
    return app.exit(e);
  }

  dim3 threadsPerBlock(16, 16);
  dim3 blocks(dim.x / threadsPerBlock.x, dim.y / threadsPerBlock.y);

  // Initialization
  handleSdl(SDL_Init(SDL_INIT_VIDEO) == 0);

  glewExperimental = GL_TRUE;
  if (mode == Mode::CUDA_GL) {

  handleSdl(SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3) == 0);
  handleSdl(SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1) == 0);
  handleSdl(SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,
                                SDL_GL_CONTEXT_PROFILE_CORE) == 0);
  }

  SdlWindow window("", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, dim.x,
                   dim.y, SDL_WINDOW_HIDDEN | SDL_WINDOW_OPENGL);
  handleGlew(glewInit());
  vector<uint8_t> pixels;

  cudaGraphicsResource *resource = nullptr;
  size_t cudaSize = 0;
  uchar3 *cudaPixels = nullptr;
  pixels.resize(dim.x * dim.y * 3, 0);

  unique_ptr<GlBuffer> buffer;
  if (mode == Mode::CUDA_GL) {
    buffer.reset(new GlBuffer(GL_PIXEL_UNPACK_BUFFER));
    handleGl(
        glNamedBufferData(buffer->id, dim.x * dim.y * 3, NULL, GL_DYNAMIC_DRAW));
    handle(cudaGraphicsGLRegisterBuffer(&resource, buffer->id,
                                        cudaGraphicsMapFlagsNone));
    handle(cudaGraphicsMapResources(1, &resource));

    handle(cudaGraphicsResourceGetMappedPointer(
        reinterpret_cast<void **>(&cudaPixels), &cudaSize, resource));
  }
  else if (mode == Mode::CUDA) {
    handle(cudaMalloc(&cudaPixels, dim.x * dim.y * 3));
  }

  float duration = 0.0; // milliseconds

  if (mode == Mode::CPU || mode == Mode::OMP) {
    // CPU version
    auto cpuStart = chrono::high_resolution_clock::now();
    { drawCpu(reinterpret_cast<uchar3*>(pixels.data()), dim, mode == Mode::OMP);
    }
    auto cpuFinish = chrono::high_resolution_clock::now();

    duration = chrono::duration_cast<chrono::microseconds>(cpuFinish - cpuStart)
                   .count() /
               1000.0;
  } else if (mode == Mode::CUDA || mode == Mode::CUDA_GL) {
    CudaEvent cudaStart;
    CudaEvent cudaFinish;

    handle(cudaEventRecord(cudaStart.event, 0));
    {
      // Render via CUDA
      drawCuda<<<blocks, threadsPerBlock>>>(cudaPixels, dim);
      handle(cudaGetLastError());

      if (mode == Mode::CUDA) {
        handle(cudaMemcpy(pixels.data(), cudaPixels, dim.x * dim.y * 3,
                          cudaMemcpyDeviceToHost));

        handle(cudaFree(cudaPixels));
      }
    }
    handle(cudaEventRecord(cudaFinish.event, 0));
    handle(cudaEventSynchronize(cudaFinish.event));
    handle(cudaEventElapsedTime(&duration, cudaStart.event, cudaFinish.event));
  }

  if (mode == Mode::CUDA_GL) {
    SDL_ShowWindow(window.window);
    handleGl(glDrawPixels(
        dim.x, dim.y, GL_RGB, GL_UNSIGNED_BYTE,
        0)); // TODO: glDrawPixels is deprecated, can I use another function?
    SDL_GL_SwapWindow(window.window);
  } else {
    SDL_ShowWindow(window.window);
    handleSdl(SDL_UpdateTexture(window.texture, nullptr, pixels.data(),
                                3 * dim.x) == 0);
    handleSdl(SDL_RenderClear(window.renderer) == 0);
    handleSdl(
        SDL_RenderCopy(window.renderer, window.texture, nullptr, nullptr) == 0);
    SDL_RenderPresent(window.renderer);
  }

  {
    ostringstream windowTitle;
    windowTitle << "Static Ripple (" << modeString << ") " << duration << "ms";
    SDL_SetWindowTitle(window.window, windowTitle.str().c_str());
  }

  bool quit = false;
  SDL_Event event;
  while (!quit) {
    while (SDL_PollEvent(&event) != 0) {
      if (isWindowClosed(event)) {
        // TODO: Check to see if window was adjusted
        quit = true;
      }
    }

    SDL_Delay(33);
  }

  SDL_Quit();

  return 0;
}
