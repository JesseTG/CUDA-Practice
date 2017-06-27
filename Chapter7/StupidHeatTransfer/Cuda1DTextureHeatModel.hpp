#ifndef CUDA1DTEXTUREHEATMODEL_HPP
#define CUDA1DTEXTUREHEATMODEL_HPP

#include "BaseCudaCpuCopyModel.hpp"
#include "common.hpp"
#include <cuda_texture_types.h>

class Cuda1DTextureHeatModel : public BaseCudaCpuCopyModel {
public:
  Cuda1DTextureHeatModel(uint2 d, const std::vector<Heater> &h);
  ~Cuda1DTextureHeatModel();

protected /* methods */:
  void copy_heaters() override;
  void update_model() override;

private /* methods */:
  void init_heaters();

private /* members */:
  bool which;
};

#endif // CUDA1DTEXTUREHEATMODEL_HPP
