#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>

#include <CLI/CLI11.hpp>
#include <cuda_runtime.h>

#include "common.hpp"

using namespace std;

constexpr int THREADS_PER_BLOCK = 256;

__global__ void dotCudaStackSharedMemory(const float *a, const float *b, float *dest,
                    const size_t length) {

  __shared__ float cache[THREADS_PER_BLOCK];

  if (threadIdx.x == 0 && threadIdx.y == 0) {

  }

  float temp = 0;
  for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < length;
       tid += blockDim.x * gridDim.x) {
    temp += a[tid] * b[tid];
  }

  cache[threadIdx.x] = temp;
}

__global__ void dotCudaNoSharedMemory(const float* a, const float* b, float* dest, const size_t length) {

}

__global__ void dotCudaHeapSharedMemory(const float* a, const float* b, float* dest, const size_t length) {

}


void dotCpu(const float* a, const float* b, float* dest, const size_t length) {
  for (size_t i = 0; i < length; ++i) {
    dest[i] = a[i] * b[i];
  }
}

int main(int argc, char *argv[]) {
  CLI::App app("Take the dot product two vectors");

  cudaDeviceProp deviceProperties;
  int deviceNumber = 0;
  handle(cudaGetDevice(&deviceNumber));
  handle(cudaGetDeviceProperties(&deviceProperties, deviceNumber));
  int blocks = 512;
  int threadsPerBlock = 1;
  size_t numberOfElements = 1024;
  try {
    app.add_option("--blocks", blocks, "Number of blocks", true)
        ->check(CLI::Range(1, deviceProperties.maxGridSize[0]));
    app.add_option("--threads", threadsPerBlock, "Threads per block", true)
        ->check(CLI::Range(1, deviceProperties.maxThreadsPerBlock));
    app.add_option("--size", numberOfElements, "Vector size", true);
    app.parse(argc, argv);
  } catch (CLI::Error &e) {
    return app.exit(e);
  }
  CudaEvent cudaStart;
  CudaEvent cudaFinish;

  vector<int> dataA;
  vector<int> dataB;
  vector<int> cpuResult;
  vector<int> cudaResult;
  int *deviceA = nullptr;
  int *deviceB = nullptr;
  int *deviceResult = nullptr;

  // Initialization
  dataA.resize(numberOfElements, 0);
  dataB.resize(numberOfElements, 0);
  cudaResult.resize(numberOfElements, 0);
  cpuResult.resize(numberOfElements, 0);

  for (size_t i = 0; i < numberOfElements; ++i) {
    dataA[i] = -i;
    dataB[i] = i * i;
  }

  // CPU version
  auto cpuStart = chrono::high_resolution_clock::now();
  {
    for (size_t i = 0; i < numberOfElements; ++i) {
      cpuResult[i] = (dataA[i] + dataB[i]);
    }
  }
  auto cpuFinish = chrono::high_resolution_clock::now();

  // CUDA version
  float cudaDuration = 0.0;
  handle(cudaEventRecord(cudaStart.event, 0));
  {
    handle(cudaMalloc(&deviceA, numberOfElements * sizeof(*deviceA)));
    handle(cudaMalloc(&deviceB, numberOfElements * sizeof(*deviceB)));
    handle(cudaMalloc(&deviceResult, numberOfElements * sizeof(*deviceResult)));

    assert(deviceA != nullptr);
    assert(deviceB != nullptr);
    assert(deviceResult != nullptr);

    handle(cudaMemcpy(deviceA, dataA.data(),
                      numberOfElements * sizeof(*deviceA),
                      cudaMemcpyHostToDevice));
    handle(cudaMemcpy(deviceB, dataB.data(),
                      numberOfElements * sizeof(*deviceB),
                      cudaMemcpyHostToDevice));

//    dotCpu<<<(numberOfElements + threadsPerBlock - 1) / threadsPerBlock,
//          threadsPerBlock>>>(deviceA, deviceB, deviceResult, numberOfElements);
    handle(cudaMemcpy(cudaResult.data(), deviceResult,
                      numberOfElements * sizeof(*deviceResult),
                      cudaMemcpyDeviceToHost));
  }
  handle(cudaEventRecord(cudaFinish.event, 0));
  handle(cudaEventSynchronize(cudaFinish.event));
  handle(
      cudaEventElapsedTime(&cudaDuration, cudaStart.event, cudaFinish.event));

  handle(cudaFree(deviceA));
  handle(cudaFree(deviceB));
  handle(cudaFree(deviceResult));

  assert(cpuResult == cudaResult);

  auto cpuDuration =
      chrono::duration_cast<chrono::microseconds>(cpuFinish - cpuStart).count();
  cout << "CPU:\t" << cpuDuration / 1000.0 << "ms" << endl;
  cout << "CUDA:\t" << cudaDuration << "ms" << endl;

  return 0;
}
