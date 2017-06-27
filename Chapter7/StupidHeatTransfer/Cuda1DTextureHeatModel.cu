#include "Cuda1DTextureHeatModel.hpp"

__global__ static void copy_heater_cells(uint2, float * __restrict__);
__global__ static void update_cell(bool, float * __restrict__, uint2);
__device__ static float computeHeat1DTexture(bool, uint2, uint32_t, uint32_t);

static texture<float> heaterCellsTexture;
static texture<float> cellsA;
static texture<float> cellsB;

__device__ static float computeHeat1DTexture(bool which, uint2 dim, uint32_t x,
                                             uint32_t y) {
  // T_new = T_old + sum(k * (n - T_old) for n in neighbors)
  uint32_t offset = x + (y * dim.x);
  uint32_t left = (x <= 0) ? offset : offset - 1;
  uint32_t right = (x >= (dim.x - 1)) ? offset : offset + 1;
  uint32_t top = (y <= 0) ? offset : offset - dim.x;
  uint32_t bottom = (y >= (dim.y - 1)) ? offset : offset + dim.x;

  float t, l, c, r, b;

  if (which) {
    t = tex1Dfetch(cellsA, top);
    l = tex1Dfetch(cellsA, left);
    c = tex1Dfetch(cellsA, offset);
    r = tex1Dfetch(cellsA, right);
    b = tex1Dfetch(cellsA, bottom);
  } else {
    t = tex1Dfetch(cellsB, top);
    l = tex1Dfetch(cellsB, left);
    c = tex1Dfetch(cellsB, offset);
    r = tex1Dfetch(cellsB, right);
    b = tex1Dfetch(cellsB, bottom);
  }

  return c + 0.25f * (t + b + r + l - 4 * c);
}

Cuda1DTextureHeatModel::Cuda1DTextureHeatModel(uint2 d,
                                               const std::vector<Heater> &h)
    : BaseCudaCpuCopyModel(d, h), which(false) {

  handle(cudaBindTexture(nullptr, cellsA, source,
                         dim.x * dim.y * sizeof(*source)));
  handle(cudaBindTexture(nullptr, cellsB, dest, dim.x * dim.y * sizeof(*dest)));
  handle(cudaBindTexture(nullptr, heaterCellsTexture, heaterCells,
                         dim.x * dim.y * sizeof(*heaterCells)));
}

Cuda1DTextureHeatModel::~Cuda1DTextureHeatModel() {
  handle(cudaUnbindTexture(cellsA));
  handle(cudaUnbindTexture(cellsB));
  handle(cudaUnbindTexture(heaterCellsTexture));
}

void Cuda1DTextureHeatModel::copy_heaters() {
  copy_heater_cells<<<blocks, threads>>>(dim, source);
  handle(cudaGetLastError());
}

__global__ static void copy_heater_cells(uint2 dim, float * __restrict__ cells) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int offset = x + y * dim.x;

  float c = tex1Dfetch(heaterCellsTexture, offset);
  if (c > 0) {
    cells[offset] = c;
  }
}

void Cuda1DTextureHeatModel::update_model() {
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
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  size_t offset = x + y * dim.x;

  dest[offset] = computeHeat1DTexture(which, dim, x, y);
  // This is the problem
}
