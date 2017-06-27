#ifndef CUDABUFFERHEATMODEL_HPP
#define CUDABUFFERHEATMODEL_HPP

#include <vector>

#include "BaseCudaCpuCopyModel.hpp"
#include "Heater.hpp"

class CudaBufferHeatModel : public BaseCudaCpuCopyModel {

public:
  CudaBufferHeatModel(uint2 d, const std::vector<Heater> &h);
  ~CudaBufferHeatModel();

protected /* methods */:
  void copy_heaters() override;
  void update_model() override;

private /* methods */:
  void init_heaters();
};

#endif // CUDABUFFERHEATMODEL_HPP
