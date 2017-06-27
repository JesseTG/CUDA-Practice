#include "BaseHeatModel.hpp"
#include "common.hpp"

using namespace std;

BaseHeatModel::BaseHeatModel(uint2 d, const std::vector<Heater> &h)
    : heaters(h), dim(d) {
  pixels.resize(dim.x * dim.y * 3, 0);

  SDL_Rect screen{0, 0, static_cast<int>(d.x), static_cast<int>(d.y)};

  for (Heater &heater : heaters) {
    // Constrain the heater to the screen
    SDL_Rect rect = heater.rect;
    if (!SDL_IntersectRect(&rect, &screen, &heater.rect)) {
      // TODO: Move non-intersecting rects to some default location
    }
  }
}

BaseHeatModel::~BaseHeatModel() {}

__device__ __host__ float BaseHeatModel::computeHeat(const float * __restrict__ in, uint2 dim,
                                                     uint32_t x, uint32_t y) {
  // T_new = T_old + sum(k * (n - T_old) for n in neighbors)
  uint32_t offset = x + (y * dim.x);
  uint32_t left = (x <= 0) ? offset : offset - 1;
  uint32_t right = (x >= (dim.x - 1)) ? offset : offset + 1;
  uint32_t top = (y <= 0) ? offset : offset - dim.x;
  uint32_t bottom = (y >= (dim.y - 1)) ? offset : offset + dim.x;

  return in[offset] +
         0.25f * (in[top] + in[bottom] + in[right] + in[left] - (in[offset] * 4));
}

__device__ __host__ uint8_t BaseHeatModel::value(float n1, float n2, int hue) {
  if (hue > 360)
    hue -= 360;
  else if (hue < 0)
    hue += 360;

  if (hue < 60)
    return static_cast<uint8_t>(255 * (n1 + (n2 - n1) * hue / 60));
  if (hue < 180)
    return static_cast<uint8_t>(255 * n2);
  if (hue < 240)
    return static_cast<uint8_t>(255 * (n1 + (n2 - n1) * (240 - hue) / 60));

  return static_cast<uint8_t>(255 * n1);
}

__device__ __host__ uchar3 BaseHeatModel::tempToColor(float temperature) {
  // map from threadIdx/BlockIdx to pixel position
  uchar3 result;

  float s = 1;
  int h = (180 + static_cast<int>(360.0f * temperature)) % 360;
  float m1, m2;

  if (temperature <= 0.5f)
    m2 = temperature * (1 + s);
  else
    m2 = temperature + s - temperature * s;

  m1 = 2 * temperature - m2;

  result.x = BaseHeatModel::value(m1, m2, h + 120);
  result.y = BaseHeatModel::value(m1, m2, h);
  result.z = BaseHeatModel::value(m1, m2, h - 120);

  return result;
}

float BaseHeatModel::update() {
  this->start_timing();
  this->copy_heaters();
  this->update_model();
  this->copy_model_to_pixels();
  return this->stop_timing();
}
