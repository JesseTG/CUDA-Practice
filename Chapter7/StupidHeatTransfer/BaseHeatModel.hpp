#ifndef BASEHEATMODEL_HPP
#define BASEHEATMODEL_HPP

#include <cstdint>
#include <vector>
#include <SDL_rect.h>

#include <cuda_runtime.h>

#include "Heater.hpp"

class BaseHeatModel {
public:
  virtual ~BaseHeatModel();

  explicit BaseHeatModel(uint2, const std::vector<Heater>&);
  BaseHeatModel(const BaseHeatModel &) = delete;
  BaseHeatModel(BaseHeatModel &&) = delete;
  BaseHeatModel &operator=(const BaseHeatModel &) = delete;
  BaseHeatModel &operator=(BaseHeatModel &&) = delete;

  const std::vector<uint8_t>& getPixels() const {
    return pixels;
  }

  float update();

  __device__ __host__ static float computeHeat(const float * __restrict__, uint2, uint32_t, uint32_t);
  __device__ __host__ static uint8_t value(float, float, int);
  __device__ __host__ static uchar3 tempToColor(float);

protected /* fields */:
  std::vector<Heater> heaters;
  uint2 dim;
  std::vector<uint8_t> pixels;

protected /* methods */:
  virtual void start_timing() = 0;
  virtual void copy_heaters() = 0;
  virtual void update_model() = 0;
  virtual void copy_model_to_pixels() = 0;
  virtual float stop_timing() = 0;
};

#endif // BASEHEATMODEL_HPP
