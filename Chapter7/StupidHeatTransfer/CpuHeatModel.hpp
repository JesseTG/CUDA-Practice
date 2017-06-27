#ifndef CPUHEATMODEL_HPP
#define CPUHEATMODEL_HPP

#include "BaseHeatModel.hpp"

#include <chrono>
#include <vector>

class CpuHeatModel : public BaseHeatModel {
public:
  explicit CpuHeatModel(uint2 d, const std::vector<Heater>& h, bool);

  virtual ~CpuHeatModel();

protected /* fields */:
  bool omp;
  std::vector<float> cellsA;
  std::vector<float> cellsB;
  std::vector<float> heaterCells;
  std::chrono::high_resolution_clock::time_point frameStart;

  std::vector<float>* source;
  std::vector<float>* dest;

protected /* methods */:
  void start_timing() override;
  void copy_heaters() override;
  void update_model() override;
  void copy_model_to_pixels() override;
  float stop_timing() override;

private /* methods */:
  void init_heaters();
};

#endif // CPUHEATMODEL_HPP
