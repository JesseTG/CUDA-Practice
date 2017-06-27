#ifndef BASECUDACPUCOPYMODEL_HPP
#define BASECUDACPUCOPYMODEL_HPP

#include "BaseCudaHeatModel.hpp"
#include "common.hpp"

class BaseCudaCpuCopyModel : public BaseCudaHeatModel {
public:
  BaseCudaCpuCopyModel(uint2 d, const std::vector<Heater> &h);

  virtual ~BaseCudaCpuCopyModel();

protected /* methods */:
  void copy_model_to_pixels() override;

protected /* members */:
  uint8_t* cudaPixels;

};

#endif // BASECUDACPUCOPYMODEL_HPP
