#ifndef CUDA2DTEXTUREHEATMODEL_HPP
#define CUDA2DTEXTUREHEATMODEL_HPP

#include "BaseCudaCpuCopyModel.hpp"
#include "common.hpp"
#include <cuda_texture_types.h>

class Cuda2DTextureHeatModel : public BaseCudaCpuCopyModel {
public:
  Cuda2DTextureHeatModel(uint2 d, const std::vector<Heater> &h);
  ~Cuda2DTextureHeatModel();

protected /* methods */:
  void copy_heaters() override;
  void update_model() override;

private /* methods */:
  void init_heaters();

private /* members */:
  bool which;
  cudaChannelFormatDesc desc;
};

#endif // CUDA2DTEXTUREHEATMODEL_HPP
