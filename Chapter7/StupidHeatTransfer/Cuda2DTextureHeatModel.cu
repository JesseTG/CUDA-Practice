#include "Cuda2DTextureHeatModel.hpp"

__global__ static void copy_heater_cells(uint2, float * __restrict__);
__global__ static void update_cell(bool, float * __restrict__, uint2);
__device__ static float computeHeat2DTexture(bool, uint32_t, uint32_t);

static texture<float, 2> heaterCellsTexture;
static texture<float, 2> cellsA;
static texture<float, 2> cellsB;

__device__ static float computeHeat2DTexture(bool which, uint32_t x,
                                             uint32_t y) {
  // T_new = T_old + sum(k * (n - T_old) for n in neighbors)
  float t, l, c, r, b;

  if (which) {
    t = tex2D(cellsA, x, y - 1);
    l = tex2D(cellsA, x - 1, y);
    c = tex2D(cellsA, x, y);
    r = tex2D(cellsA, x + 1, y);
    b = tex2D(cellsA, x, y + 1);
  } else {
    t = tex2D(cellsB, x, y - 1);
    l = tex2D(cellsB, x - 1, y);
    c = tex2D(cellsB, x, y);
    r = tex2D(cellsB, x + 1, y);
    b = tex2D(cellsB, x, y + 1);
  }

  return c + 0.25f * (t + b + r + l - 4 * c);
}

Cuda2DTextureHeatModel::Cuda2DTextureHeatModel(uint2 d,
                                               const std::vector<Heater> &h)
    : BaseCudaCpuCopyModel(d, h), which(false) {

  desc = cudaCreateChannelDesc<float>();
  handle(cudaBindTexture2D(nullptr, cellsA, source, desc, dim.x, dim.y,
                           dim.x * sizeof(*source)));
  handle(cudaBindTexture2D(nullptr, cellsB, dest, desc, dim.x, dim.y,
                           dim.x * sizeof(*dest)));
  handle(cudaBindTexture2D(nullptr, heaterCellsTexture, heaterCells, desc,
                           dim.x, dim.y, dim.x * sizeof(*heaterCells)));
}

Cuda2DTextureHeatModel::~Cuda2DTextureHeatModel() {
  handle(cudaUnbindTexture(cellsA));
  handle(cudaUnbindTexture(cellsB));
  handle(cudaUnbindTexture(heaterCellsTexture));
}

void Cuda2DTextureHeatModel::copy_heaters() {
  copy_heater_cells<<<blocks, threads>>>(dim, source);
  handle(cudaGetLastError());
}

__global__ static void copy_heater_cells(uint2 dim, float * __restrict__ cells) {
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  size_t offset = x + y * dim.x;

  float c = tex2D(heaterCellsTexture, x, y);
  if (c > 0) {
    cells[offset] = c;
  }
}

void Cuda2DTextureHeatModel::update_model() {
  // By this point, we've already filled the grid with the heater

  update_cell<<<blocks, threads>>>(which, dest, dim);
  handle(cudaGetLastError());

  // Swap the bufers
  which = !which;
  float *temp = source;
  source = dest;
  dest = temp;
}

__global__ static void update_cell(bool which, float * __restrict__ dest, uint2 dim) {
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  size_t offset = x + y * dim.x;

  dest[offset] = computeHeat2DTexture(which, x, y);
  // This is the problem
}
