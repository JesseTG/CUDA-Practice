#include "CudaOpenGLHeatModel.hpp"

#include <cuda_gl_interop.h>

using namespace std;

__global__ static void copy_heater_cells(const float * __restrict__, uint2, float * __restrict__);
__global__ static void update_cell(const float * __restrict__, float * __restrict__, uint2);
__global__ static void copy_cells_to_pixels(const float * __restrict__, uint8_t * __restrict__, uint2);
CudaOpenGLHeatModel::CudaOpenGLHeatModel(uint2 d, const vector<Heater> &h)
    : BaseCudaHeatModel(d, h), cudaBufferAddress(nullptr), buffer(nullptr),
      resource(nullptr), cudaSize(0) {
  handleSdl(SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3) == 0);
  handleSdl(SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1) == 0);
  handleSdl(SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,
                                SDL_GL_CONTEXT_PROFILE_CORE) == 0);
  handleSdl(SDL_GL_SetSwapInterval(1) == 0);

  glewExperimental = GL_TRUE;
  handleGlew(glewInit());
  buffer.reset(new GlBuffer(GL_PIXEL_UNPACK_BUFFER));
  handleGl(
      glNamedBufferData(buffer->id, dim.x * dim.y * 3, NULL, GL_DYNAMIC_DRAW));
  handle(cudaGraphicsGLRegisterBuffer(&resource, buffer->id,
                                      cudaGraphicsMapFlagsNone));
}

CudaOpenGLHeatModel::~CudaOpenGLHeatModel() {
  handle(cudaGraphicsUnregisterResource(resource));
}

void CudaOpenGLHeatModel::copy_heaters() {
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

void CudaOpenGLHeatModel::update_model() {
  // By this point, we've already filled the grid with the heater

  handle(cudaGraphicsMapResources(1, &resource));

  handle(cudaGraphicsResourceGetMappedPointer(
      reinterpret_cast<void **>(&cudaBufferAddress), &cudaSize, resource));

  update_cell<<<blocks, threads>>>(source, dest, dim);
  handle(cudaGetLastError());

  handle(cudaGraphicsUnmapResources(1, &resource, NULL));

  // Swap the bufers
  float *temp = source;
  source = dest;
  dest = temp;
}

__global__ static void update_cell(const float * __restrict__ source,  float * __restrict__ dest,
                                   uint2 dim) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  size_t offset = x + y * dim.x;

  dest[offset] = BaseHeatModel::computeHeat(source, dim, x, y);
  // This is the problem
}

void CudaOpenGLHeatModel::copy_model_to_pixels() {
    copy_cells_to_pixels<<<blocks, threads>>>(dest, cudaBufferAddress, dim);
    handle(cudaGetLastError());

//  handle(cudaMemcpy(pixels.data(), cudaPixels, dim.x * dim.y * 3,
//                    cudaMemcpyDeviceToHost));
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
