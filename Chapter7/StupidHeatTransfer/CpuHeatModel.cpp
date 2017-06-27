#include "CpuHeatModel.hpp"
#include <iostream>
using namespace std;

CpuHeatModel::CpuHeatModel(uint2 d, const vector<Heater> &h, bool o)
    : BaseHeatModel(d, h), omp(o), source(nullptr), dest(nullptr) {
  cellsA.resize(d.x * d.y, 0.0f);
  cellsB.resize(d.x * d.y, 0.0f);
  heaterCells.resize(d.x * d.y, 0.0f);

  source = &cellsA;
  dest = &cellsB;

  init_heaters();
}

CpuHeatModel::~CpuHeatModel() {}

void CpuHeatModel::init_heaters() {

  for (const Heater& h : heaters) {

    #pragma omp parallel for if(omp)
    for (size_t y = h.rect.y; y < h.rect.y + h.rect.h; ++y) {


      #pragma omp parallel for if(omp)
      for (size_t x = h.rect.x; x < h.rect.x + h.rect.w; ++x) {
        size_t offset = x + y * dim.x;

        heaterCells[offset] = h.temperature;
      }
    }
  }
}

void CpuHeatModel::start_timing() {
  frameStart = std::chrono::high_resolution_clock::now();
}

void CpuHeatModel::copy_heaters() {
  #pragma omp parallel for if(omp)
  for (size_t i = 0; i < dim.x * dim.y; ++i) {
    if (heaterCells[i] > 0) {
      (*source)[i] = heaterCells[i];
    }
  }
}

void CpuHeatModel::update_model() {
  // By this point, we've already filled the grid with the heater

  auto data = source->data();
  uint32_t len = dim.x * dim.y;

  #pragma omp parallel for if(omp)
  for (uint32_t i = 0; i < len; ++i) {
    uint32_t x = i % dim.x;
    uint32_t y = i / dim.x;

    (*dest)[i] = BaseHeatModel::computeHeat(data, dim, x, y);
  }

  // Swap the bufers
  vector<float> *temp = source;
  source = dest;
  dest = temp;
}

void CpuHeatModel::copy_model_to_pixels() {

  #pragma omp parallel for if(omp)
  for (size_t i = 0; i < source->size(); ++i) {
    uchar3 color = BaseHeatModel::tempToColor((*dest)[i]);

    pixels[i * 3 + 0] = color.x;
    pixels[i * 3 + 1] = color.y;
    pixels[i * 3 + 2] = color.z;
  }
}

float CpuHeatModel::stop_timing() {
  chrono::high_resolution_clock::time_point frameStop =
      chrono::high_resolution_clock::now();

  auto duration =
      chrono::duration_cast<chrono::microseconds>(frameStop - frameStart);

  return duration.count() / 1000.0f;
}
