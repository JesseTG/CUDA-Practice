#include "CudaBufferHeatModel.hpp"

using namespace std;

__global__ static void copy_heater_cells(const float * __restrict__ , uint2, float *);
__global__ static void update_cell(const float * __restrict__ , float * __restrict__ , uint2);

CudaBufferHeatModel::CudaBufferHeatModel(uint2 d, const vector<Heater> &h)
    : BaseCudaCpuCopyModel(d, h) {
}

CudaBufferHeatModel::~CudaBufferHeatModel() {
}

void CudaBufferHeatModel::copy_heaters() {
  copy_heater_cells<<<blocks, threads>>>(heaterCells, dim, source);
  handle(cudaGetLastError());
}

__global__ static void copy_heater_cells(const float * __restrict__ heaterCells, uint2 dim,
                                         float * __restrict__ cells) {
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  size_t offset = x + y * dim.x;

  if (heaterCells[offset] > 0) {
    cells[offset] = heaterCells[offset];
  }
}

void CudaBufferHeatModel::update_model() {
  // By this point, we've already filled the grid with the heater

  update_cell<<<blocks, threads>>>(source, dest, dim);
  handle(cudaGetLastError());

  // Swap the bufers
  float *temp = source;
  source = dest;
  dest = temp;
}

__global__ static void update_cell(const float * __restrict__ source, float * __restrict__ dest,
                                   uint2 dim) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  size_t offset = x + y * dim.x;

  dest[offset] = BaseHeatModel::computeHeat(source, dim, x, y);
  // This is the problem
}
