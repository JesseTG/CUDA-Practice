#include <chrono>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>

#include <cuda_runtime.h>

#include "common.hpp"

using namespace std;

__global__ void add(const int *a, const int *b, int *dest, const size_t length) {
  int tid = blockIdx.x;

  if (tid < length) {
    dest[tid] = a[tid] - b[tid];
  }
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    return 1;
  }

  CudaEvent cudaStart;
  CudaEvent cudaFinish;
  size_t numberOfElements = 0;

  vector<int> dataA;
  vector<int> dataB;
  vector<int> cpuResult;
  vector<int> cudaResult;
  vector<int> ompResult;
  int* deviceA = nullptr;
  int* deviceB = nullptr;
  int* deviceResult = nullptr;

  // Initialization
  numberOfElements = stoul(argv[1]);

  dataA.resize(numberOfElements, 0);
  dataB.resize(numberOfElements, 0);
  cudaResult.resize(numberOfElements, 0);
  cpuResult.resize(numberOfElements, 0);
  ompResult.resize(numberOfElements, 0);

  for (int i = 0; i < numberOfElements; ++i) {
    dataA[i] = -i;
    dataB[i] = i * i;
  }

  // CPU version
  auto cpuStart = chrono::high_resolution_clock::now();
  {
    for (int i = 0; i < numberOfElements; ++i) {
      cpuResult[i] = (dataA[i] - dataB[i]);
    }
  }
  auto cpuFinish = chrono::high_resolution_clock::now();

  // OpenMP version
  auto ompStart = chrono::high_resolution_clock::now();
  {
    #pragma omp parallel for
    for (int i = 0; i < numberOfElements; ++i) {
      ompResult[i] = (dataA[i] - dataB[i]);
    }
  }
  auto ompFinish = chrono::high_resolution_clock::now();

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

    handle(cudaMemcpy(deviceA, dataA.data(), numberOfElements * sizeof(*deviceA), cudaMemcpyHostToDevice));
    handle(cudaMemcpy(deviceB, dataB.data(), numberOfElements * sizeof(*deviceB), cudaMemcpyHostToDevice));

    add<<<numberOfElements, 1>>>(deviceA, deviceB, deviceResult, numberOfElements);
    handle(cudaMemcpy(cudaResult.data(), deviceResult, numberOfElements * sizeof(*deviceResult), cudaMemcpyDeviceToHost));
  }
  handle(cudaEventRecord(cudaFinish.event, 0));
  handle(cudaEventSynchronize(cudaFinish.event));
  handle(cudaEventElapsedTime(&cudaDuration, cudaStart.event, cudaFinish.event));

  handle(cudaFree(deviceA));
  handle(cudaFree(deviceB));
  handle(cudaFree(deviceResult));

  assert(cpuResult == ompResult);
  assert(cpuResult == cudaResult);

  auto cpuDuration =
      chrono::duration_cast<chrono::microseconds>(cpuFinish - cpuStart).count();

  auto ompDuration =
      chrono::duration_cast<chrono::microseconds>(ompFinish - ompStart).count();
  cout << "CPU:\t" << cpuDuration / 1000.0 << "ms" << endl;
  cout << "OpenMP:\t" << ompDuration / 1000.0 << "ms" << endl;
  cout << "CUDA:\t" << cudaDuration << "ms" << endl;

  return 0;
}
