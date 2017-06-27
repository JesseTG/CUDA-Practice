#ifndef BASECUDAHEATMODEL_HPP
#define BASECUDAHEATMODEL_HPP

#include "BaseHeatModel.hpp"
#include "common.hpp"

class BaseCudaHeatModel : public BaseHeatModel {
public:
  BaseCudaHeatModel(uint2 d, const std::vector<Heater> &h);

  virtual ~BaseCudaHeatModel();

protected /* methods */:
  void init_heaters();
  void start_timing() override final;
  float stop_timing() override final;

protected /* members */:
  float*  heaterCells;
  float* source;
  float* dest;
  dim3 blocks;
  dim3 threads;
  CudaEvent frameStart;
  CudaEvent frameStop;

};

#endif // BASECUDAHEATMODEL_HPP
