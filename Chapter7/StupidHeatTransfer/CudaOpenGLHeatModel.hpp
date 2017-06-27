#ifndef CUDAOPENGLHEATMODEL_HPP
#define CUDAOPENGLHEATMODEL_HPP

#include <memory>

#include "BaseCudaHeatModel.hpp"
#include "Heater.hpp"

class CudaOpenGLHeatModel : public BaseCudaHeatModel {

public:
  CudaOpenGLHeatModel(uint2 d, const std::vector<Heater> &h);
  ~CudaOpenGLHeatModel();

protected /* methods */:
  void copy_heaters() override;
  void update_model() override;
  void copy_model_to_pixels() override;

private /* members */:
  uint8_t* cudaBufferAddress;
  std::unique_ptr<GlBuffer> buffer;
  cudaGraphicsResource *resource;
  size_t cudaSize;
  bool which;
  cudaChannelFormatDesc desc;

};

#endif // CUDAOPENGLHEATMODEL_HPP
