#include "BaseCudaCpuCopyModel.hpp"

__global__ static void copy_cells_to_pixels(const float * __restrict__, uint8_t * __restrict__, uint2);

using namespace std;

BaseCudaCpuCopyModel::BaseCudaCpuCopyModel(uint2 d, const vector<Heater> &h)
    : BaseCudaHeatModel(d, h) {

  handle(cudaMalloc(&cudaPixels, d.x * d.y * 3 * sizeof(*cudaPixels)));
  handle(cudaMemset(cudaPixels, 0, d.x * d.y * 3 * sizeof(*cudaPixels)));
}

BaseCudaCpuCopyModel::~BaseCudaCpuCopyModel() { handle(cudaFree(cudaPixels)); }

void BaseCudaCpuCopyModel::copy_model_to_pixels() {

  copy_cells_to_pixels<<<blocks, threads>>>(dest, cudaPixels, dim);
  handle(cudaGetLastError());

  handle(cudaMemcpy(pixels.data(), cudaPixels, dim.x * dim.y * 3,
                    cudaMemcpyDeviceToHost));
}

__global__ static void copy_cells_to_pixels(const float * __restrict__ cells, uint8_t * __restrict__ pixels,
                                            uint2 dim) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  size_t offset = x + y * dim.x;

  uchar3 color = BaseHeatModel::tempToColor(cells[offset]);

  pixels[offset * 3 + 0] = color.x;
  pixels[offset * 3 + 1] = color.y;
  pixels[offset * 3 + 2] = color.z;
}
