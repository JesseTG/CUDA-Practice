#include <chrono>
#include <cmath>
#include <numeric>

#include <CLI/CLI11.hpp>
#include <GL/glew.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_surface.h>
#include <SDL2/SDL_ttf.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include "common.hpp"

using namespace std;

__device__ __host__ uint8_t computePixel(int x, int y, int2 dim, size_t ticks) {
  float fx = x - dim.x / 2;
  float fy = y - dim.y / 2;
  float d = sqrtf(fx * fx + fy * fy);

  return static_cast<uint8_t>(
      128.0f + 127.0f * cos(d / 10.0f - ticks / 7.0f) / (d / 10.0f + 1.0f));
}

void animateCpu(uint8_t *pixels, int2 dim, size_t ticks, bool omp) {
  size_t length = dim.x * dim.y;

#pragma omp parallel for if (omp)
  for (size_t i = 0; i < length; ++i) {
    uint8_t grey = computePixel(i % dim.x, i / dim.x, dim, ticks);

    pixels[i * 3 + 0] = grey;
    pixels[i * 3 + 1] = grey;
    pixels[i * 3 + 2] = grey;
  }
}

__global__ void animateCuda(uint8_t *pixels, int2 dim, size_t ticks) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  uint64_t offset = x + y * dim.x;

  if (x < dim.x && y < dim.y) {
    uint8_t grey = computePixel(x, y, dim, ticks);

    pixels[offset * 3 + 0] = grey;
    pixels[offset * 3 + 1] = grey;
    pixels[offset * 3 + 2] = grey;
  }
}

const string DEFAULT_FONT =
    "/usr/share/fonts/truetype/ubuntu-font-family/UbuntuMono-B.ttf";

enum class Mode {
  CPU,
  OMP,
  CUDA,
  CUDA_GL,
};

int main(int argc, char *argv[]) {
  CLI::App app("Example program that generates a ripple animation");

  int ptsize = 28;
  Mode mode = Mode::CUDA_GL;
  string modeString = "CUDA_GL";
  string fontPath = DEFAULT_FONT;
  int2 dim{640, 480};
  try {
    app.add_option("--mode", modeString,
                   "Animation mode (CPU, OMP, CUDA, CUDA_GL)", true)
        ->check([](const string &m) {
          return (m == "CPU" || m == "OMP" || m == "CUDA" || m == "CUDA_GL");
        });
    app.add_option("--font", fontPath, "Font to use", true)
        ->check(CLI::ExistingFile);
    app.add_option("--ptsize", ptsize, "Font size", true);
    app.add_option("--width", dim.x, "Image width in pixels", true);
    app.add_option("--height", dim.y, "Image height in pixels", true);
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

  dim3 threads(16, 16);
  dim3 blocks(dim.x / threads.x, dim.y / threads.y);

  // Initialization
  handleSdl(SDL_Init(SDL_INIT_VIDEO) == 0);
  handleSdl(TTF_Init() == 0);

  glewExperimental = GL_TRUE;
  {
    SdlFont font(fontPath, ptsize);

    ostringstream windowTitle;
    windowTitle << "OpenGL Ripple Animation (" << modeString << ')';
    SdlWindow window(windowTitle.str().c_str(), SDL_WINDOWPOS_UNDEFINED,
                     SDL_WINDOWPOS_UNDEFINED, dim.x, dim.y,
                     SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL);
    vector<uint8_t> pixels;
    cudaGraphicsResource *resource = nullptr;
    size_t cudaSize = 0;
    uint8_t *cudaPixels = nullptr;
    pixels.resize(dim.x * dim.y * 3, 0);

    unique_ptr<GlBuffer> buffer;
    if (mode == Mode::CUDA_GL) {
      handleSdl(SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3) == 0);
      handleSdl(SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1) == 0);
      handleSdl(SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,
                                    SDL_GL_CONTEXT_PROFILE_CORE) == 0);
      handleSdl(SDL_GL_SetSwapInterval(1) == 0);
      handleGlew(glewInit());

      buffer.reset(new GlBuffer(GL_PIXEL_UNPACK_BUFFER));
      handleGl(glNamedBufferData(buffer->id, dim.x * dim.y * 3, NULL,
                                 GL_DYNAMIC_DRAW));
      handle(cudaGraphicsGLRegisterBuffer(&resource, buffer->id,
                                          cudaGraphicsMapFlagsNone));

    } else if (mode == Mode::CUDA) {
      handle(cudaMalloc(&cudaPixels, dim.x * dim.y * 3));
    }

    SDL_Color textColor{255, 0, 0, 255};
    ostringstream stats;

    bool quit = false;
    SDL_Event event;
    CudaEvent cudaStart;
    CudaEvent cudaFinish;
    float cudaDuration = 0.0;

    size_t ticks = 0;
    float totalTime = 0.0f;
    while (!quit) {
      while (SDL_PollEvent(&event) != 0) {
        if (isWindowClosed(event)) {
          quit = true;
        }
      }

      ticks++;

      if (mode == Mode::CPU || mode == Mode::OMP) {
        auto now = chrono::steady_clock::now();
        animateCpu(pixels.data(), dim, ticks, mode == Mode::OMP);
        auto end = chrono::steady_clock::now();

        totalTime +=
            chrono::duration_cast<chrono::microseconds>(end - now).count() /
            1000.0;
      } else if (mode == Mode::CUDA || mode == Mode::CUDA_GL) {
        handle(cudaEventRecord(cudaStart.event, 0));
        {
          if (mode == Mode::CUDA_GL) {
            handle(cudaGraphicsMapResources(1, &resource));

            handle(cudaGraphicsResourceGetMappedPointer(
                reinterpret_cast<void **>(&cudaPixels), &cudaSize, resource));
          }
          animateCuda<<<blocks, threads>>>(cudaPixels, dim, ticks);
          handle(cudaGetLastError());

          if (mode == Mode::CUDA_GL) {
            handle(cudaGraphicsUnmapResources(1, &resource, NULL));
          } else if (mode == Mode::CUDA) {

            handle(cudaMemcpy(pixels.data(), cudaPixels, pixels.size(),
                              cudaMemcpyDeviceToHost));
          }
        }
        handle(cudaEventRecord(cudaFinish.event, 0));
        handle(cudaEventSynchronize(cudaFinish.event));
        handle(cudaEventElapsedTime(&cudaDuration, cudaStart.event,
                                    cudaFinish.event));

        totalTime += cudaDuration;
      }

      stats.str("Time: ");
      stats << (totalTime / ticks) << "ms";

      if (mode == Mode::CUDA_GL) {
        handleGl(glDrawPixels(dim.x, dim.y, GL_RGB, GL_UNSIGNED_BYTE, 0));
        // TODO: glDrawPixels is deprecated, can I use another function?

        SDL_GL_SwapWindow(window.window);
        SDL_SetWindowTitle(window.window, stats.str().c_str());
      } else {
        SDL_Surface *statsSurface =
            TTF_RenderText_Solid(font.font, stats.str().c_str(), textColor);
        handleSdl(statsSurface != nullptr);

        SDL_Texture *statsTexture =
            SDL_CreateTextureFromSurface(window.renderer, statsSurface);
        handleSdl(statsTexture != nullptr);

        handleSdl(SDL_UpdateTexture(window.texture, nullptr, pixels.data(),
                                    3 * dim.x) == 0);
        handleSdl(SDL_RenderCopy(window.renderer, window.texture, nullptr,
                                 nullptr) == 0);
        handleSdl(SDL_RenderCopy(window.renderer, statsTexture, nullptr,
                                 &statsSurface->clip_rect) == 0);
        SDL_RenderPresent(window.renderer);

        SDL_DestroyTexture(statsTexture);
        SDL_FreeSurface(statsSurface);
      }

      SDL_Delay(16);
    }

    if (mode == Mode::CUDA) {
      handle(cudaFree(cudaPixels));
    }
  }

  TTF_Quit();
  SDL_Quit();

  return 0;
}
