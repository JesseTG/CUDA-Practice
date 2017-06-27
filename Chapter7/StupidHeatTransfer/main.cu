#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>

#include <CLI/CLI11.hpp>
#include <SDL2/SDL.h>
#include <cuda_runtime.h>

#include <random>

#include "BaseHeatModel.hpp"
#include "CpuHeatModel.hpp"
#include "Cuda1DTextureHeatModel.hpp"
#include "Cuda2DTextureHeatModel.hpp"
#include "CudaBufferHeatModel.hpp"
#include "CudaOpenGLHeatModel.hpp"
#include "common.hpp"

using namespace std;

enum class Mode {
  CPU,
  OMP,
  CUDA_BUFFER,
  CUDA_GL,
  CUDA_1DTEXTURE,
  CUDA_2DTEXTURE,
};

const string DEFAULT_FONT =
    "/usr/share/fonts/truetype/ubuntu-font-family/UbuntuMono-B.ttf";

using random_type = minstd_rand;

int main(int argc, char *argv[]) {
  CLI::App app("Stuff");

  vector<Heater> heaters;
  random_type::result_type seed = 0;
  uint32_t numHeaters = 8;
  uint2 dim{640, 480};
  string modeString = "CUDA_GL";
  string fontPath = DEFAULT_FONT;
  int ptsize = 28;
  Mode mode = Mode::CUDA_GL;
  try {
    app.add_option("--font", fontPath, "Font to use", true)
        ->check(CLI::ExistingFile);
    app.add_option("--ptsize", ptsize, "Font size", true);
    app.add_option("--seed", seed, "Random seed", true);
    app.add_option("--num-heaters", numHeaters, "Number of heaters", true);
    app.add_option("--width", dim.x, "Image width in pixels", true);
    app.add_option("--height", dim.y, "Image height in pixels", true);
    app.add_option("--mode", modeString, "Rendering mode (CPU, OMP, "
                                         "CUDA_BUFFER, CUDA_GL, CUDA_1DTEXTURE, "
                                         "CUDA_2DTEXTURE)",
                   true)
        ->check([](const string &m) {
          return (m == "CPU" || m == "OMP" || m == "CUDA_BUFFER" ||
                  m == "CUDA_GL" || m == "CUDA_1DTEXTURE" ||
                  m == "CUDA_2DTEXTURE");
        });

    app.parse(argc, argv);

    if (modeString == "CPU") {
      mode = Mode::CPU;
    } else if (modeString == "OMP") {
      mode = Mode::OMP;
    } else if (modeString == "CUDA_BUFFER") {
      mode = Mode::CUDA_BUFFER;
    } else if (modeString == "CUDA_GL") {
      mode = Mode::CUDA_GL;
    } else if (modeString == "CUDA_1DTEXTURE") {
      mode = Mode::CUDA_1DTEXTURE;
    } else if (modeString == "CUDA_2DTEXTURE") {
      mode = Mode::CUDA_2DTEXTURE;
    }

  } catch (CLI::Error &e) {
    return app.exit(e);
  }

  // Initialization
  handleSdl(SDL_Init(SDL_INIT_VIDEO) == 0);
  handleSdl(TTF_Init() == 0);

  ostringstream windowTitle;
  windowTitle << "Stupid Heat Transfer (" << modeString << ")";
  SdlWindow window(windowTitle.str().c_str(), SDL_WINDOWPOS_UNDEFINED,
                   SDL_WINDOWPOS_UNDEFINED, dim.x, dim.y, SDL_WINDOW_SHOWN);

  random_type randomEngine(seed);
  normal_distribution<float> coordinateDistribution(dim.x / 4, dim.x * 0.2);
  // TODO: Make it max dimension

  normal_distribution<float> sizeDistribution(100, 20);
  normal_distribution<float> temperatureDistribution(1, 0.2);

  heaters.reserve(numHeaters);

  for (uint32_t i = 0; i < numHeaters; ++i) {
    Heater heater;

    heater.rect.x = coordinateDistribution(randomEngine);
    heater.rect.y = coordinateDistribution(randomEngine);
    heater.rect.w = sizeDistribution(randomEngine);
    heater.rect.h = sizeDistribution(randomEngine);
    heater.temperature = temperatureDistribution(randomEngine);

    heaters.push_back(heater);
  }

  {
    SdlFont font(fontPath, ptsize);

    unique_ptr<BaseHeatModel> model;

    switch (mode) {
    case Mode::CPU:
      model.reset(new CpuHeatModel(dim, heaters, false));
      break;
    case Mode::OMP:
      model.reset(new CpuHeatModel(dim, heaters, true));
      break;
    case Mode::CUDA_GL:
      model.reset(new CudaOpenGLHeatModel(dim, heaters));
      break;
    case Mode::CUDA_BUFFER:
      model.reset(new CudaBufferHeatModel(dim, heaters));
      break;
    case Mode::CUDA_1DTEXTURE:
      model.reset(new Cuda1DTextureHeatModel(dim, heaters));
      break;
    case Mode::CUDA_2DTEXTURE:
      model.reset(new Cuda2DTextureHeatModel(dim, heaters));
      break;
    }

    bool quit = false;
    ostringstream stats;
    SDL_Color textColor{255, 0, 255, 255};
    SDL_Event event;
    size_t frames = 0;
    float totalTime = 0.0f;
    while (!quit) {
      while (SDL_PollEvent(&event) != 0) {
        if (isWindowClosed(event)) {
          quit = true;
        }
      }

      float duration = model->update();

      totalTime += duration;
      frames++;

      stats.str("Time: ");
      stats << (totalTime / frames) << "ms";

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

        handleSdl(SDL_UpdateTexture(window.texture, nullptr,
                                    model->getPixels().data(), 3 * dim.x) == 0);
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
  }

  TTF_Quit();
  SDL_Quit();

  return 0;
}
