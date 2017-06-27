#include "BaseCudaHeatModel.hpp"

__global__ static void fill_buffer_with_heaters(const Heater * __restrict__, size_t, float * __restrict__,
                                                uint2);

using namespace std;

BaseCudaHeatModel::BaseCudaHeatModel(uint2 d, const vector<Heater> &h)
    : BaseHeatModel(d, h), blocks(dim.x / 16, dim.y / 16), threads(16, 16) {
  handle(cudaMalloc(&source, d.x * d.y * sizeof(*source)));
  handle(cudaMalloc(&dest, d.x * d.y * sizeof(*dest)));
  handle(cudaMalloc(&heaterCells, d.x * d.y * sizeof(*heaterCells)));

  handle(cudaMemset(source, 0, d.x * d.y * sizeof(*source)));
  handle(cudaMemset(dest, 0, d.x * d.y * sizeof(*dest)));
  handle(cudaMemset(heaterCells, 0, d.x * d.y * sizeof(*heaterCells)));

  init_heaters();
}

BaseCudaHeatModel::~BaseCudaHeatModel() {
  handle(cudaFree(source));
  handle(cudaFree(dest));
  handle(cudaFree(heaterCells));
}

void BaseCudaHeatModel::init_heaters() {
  CudaMemory<Heater> cudaHeaters(heaters.size());
  handle(cudaMemcpy(cudaHeaters.ptr, heaters.data(),
                    heaters.size() * sizeof(Heater), cudaMemcpyHostToDevice));

  fill_buffer_with_heaters<<<blocks, threads>>>(cudaHeaters.ptr, heaters.size(),
                                                heaterCells, dim);
  handle(cudaGetLastError());
}

__global__ static void fill_buffer_with_heaters(const Heater * __restrict__ heaters,
                                                size_t numHeaters,
                                                float * __restrict__ heaterBuffer,
                                                uint2 dim) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  size_t offset = x + y * dim.x;

  for (size_t i = 0; i < numHeaters; ++i) {
    const Heater &heater = heaters[i];

    if (x >= heater.rect.x && y >= heater.rect.y &&
        x < (heater.rect.x + heater.rect.w) &&
        y < (heater.rect.y + heater.rect.h)) {
      heaterBuffer[offset] = heater.temperature;
    }
  }
}

void BaseCudaHeatModel::start_timing() {
  handle(cudaEventRecord(frameStart.event, 0));
}

float BaseCudaHeatModel::stop_timing() {
  float duration = 0.0;
  handle(cudaEventRecord(frameStop.event, 0));
  handle(cudaEventSynchronize(frameStop.event));
  handle(cudaEventElapsedTime(&duration, frameStart.event, frameStop.event));

  return duration;
}
