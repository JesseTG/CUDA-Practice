#include <chrono>
#include <cmath>
#include <numeric>

#include <CLI/CLI11.hpp>
#include <SDL2/SDL.h>
#include <SDL2/SDL_surface.h>
#include <SDL2/SDL_ttf.h>
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

void animateCpu(uint8_t *pixels, int2 dim, size_t ticks) {
  for (int y = 0; y < dim.y; ++y) {
    for (int x = 0; x < dim.x; ++x) {
      uint64_t offset = x + y * dim.x;
      uint8_t grey = computePixel(x, y, dim, ticks);

      pixels[offset * 3 + 0] = grey;
      pixels[offset * 3 + 1] = grey;
      pixels[offset * 3 + 2] = grey;
    }
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

int main(int argc, char *argv[]) {
  CLI::App app("Example program that generates a ripple animation");

  cudaDeviceProp deviceProperties;
  int deviceNumber = 0;
  handle(cudaGetDevice(&deviceNumber));
  handle(cudaGetDeviceProperties(&deviceProperties, deviceNumber));
  int ptsize = 28;
  int2 blocks { 40, 30 };
  string mode = "CUDA";
  string fontPath = DEFAULT_FONT;
  int2 dim{640, 480};
  try {
    app.add_option("--mode", mode, "Animation mode (CPU or CUDA)", true)
        ->check([](const string &m) { return (m == "CPU" || m == "CUDA"); });
    app.add_option("--font", fontPath, "Font to use", true)
        ->check(CLI::ExistingFile);
    app.add_option("--ptsize", ptsize, "Font size", true);
    app.add_option("--blocksx", blocks.x, "Number of blocks on the X axis", true)
        ->check(CLI::Range(1, deviceProperties.maxGridSize[0]));
    app.add_option("--blocksy", blocks.y, "Number of blocks on the Y axis", true)
        ->check(CLI::Range(1, deviceProperties.maxGridSize[1]));
    app.add_option("--width", dim.x, "Image width in pixels", true);
    app.add_option("--height", dim.y, "Image height in pixels", true);
    app.parse(argc, argv);
  } catch (CLI::Error &e) {
    return app.exit(e);
  }

  // TODO: Color the ripple based on which block is covering which part of the
  // image
  // given: screen width, num blocks
  // get: num threads per block
  dim3 threads(ceil(dim.x / static_cast<float>(blocks.x)), ceil(dim.y / static_cast<float>(blocks.y)));

  cout << "Blocks: (" << blocks.x << ", " << blocks.y << ')' << endl;
  cout << "Threads per block: (" << threads.x << ", " << threads.y << ')' << endl;
  cout << "Image size: (" << dim.x << ", " << dim.y << ')' << endl;

  // Initialization
  handleSdl(SDL_Init(SDL_INIT_VIDEO) == 0);
  handleSdl(TTF_Init() == 0);

  {
    SdlFont font(fontPath, ptsize);

    ostringstream windowTitle;
    windowTitle << "Ripple Animation (" << mode << ')';
    SdlWindow window(windowTitle.str().c_str(), SDL_WINDOWPOS_UNDEFINED,
                     SDL_WINDOWPOS_UNDEFINED, dim.x, dim.y, SDL_WINDOW_SHOWN);
    vector<uint8_t> pixels;
    pixels.resize(dim.x * dim.y * 3, 0);

    SDL_Color textColor{255, 0, 0, 255};
    ostringstream stats;

    bool quit = false;
    SDL_Event event;
    CudaEvent cudaStart;
    CudaEvent cudaFinish;
    float cudaDuration = 0.0;
    uint8_t *cudaResults = nullptr;

    if (mode == "CUDA") {
      handle(cudaMalloc(&cudaResults, dim.x * dim.y * 3));
    }

    size_t ticks = 0;
    while (!quit) {
      while (SDL_PollEvent(&event) != 0) {
        if (isWindowClosed(event)) {
          quit = true;
        }
      }

      auto now = chrono::steady_clock::now();
      auto nowCount = now.time_since_epoch().count();
      if (mode == "CPU") {
        animateCpu(pixels.data(), dim, ticks);
        auto end = chrono::steady_clock::now();

        stats.str("Time: ");
        stats << chrono::duration_cast<chrono::microseconds>(
                     end -
                     now).count() /
                     1000.0
              << "ms";
      } else if (mode == "CUDA") {
        handle(cudaEventRecord(cudaStart.event, 0));
        {
          animateCuda<<<dim3(blocks.x, blocks.y), threads>>>(
              cudaResults, dim, ticks);
          //handle(cudaGetLastError());

          handle(cudaMemcpy(pixels.data(), cudaResults, pixels.size(),
                            cudaMemcpyDeviceToHost));
        }
        handle(cudaEventRecord(cudaFinish.event, 0));
        handle(cudaEventSynchronize(cudaFinish.event));
        handle(cudaEventElapsedTime(&cudaDuration, cudaStart.event,
                                    cudaFinish.event));

        stats.str("Time: ");
        stats << cudaDuration << "ms";
      }

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

      SDL_Delay(16);
      ticks++;
    }

    if (mode == "CUDA") {
      handle(cudaFree(cudaResults));
    }
  }

  TTF_Quit();
  SDL_Quit();

  return 0;
}
