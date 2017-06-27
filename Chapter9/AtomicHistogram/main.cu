#include <atomic>
#include <chrono>

#include <CLI/CLI11.hpp>
#include <cuda_runtime.h>
#include <random>

#include "common.hpp"

using namespace std;

using HistType = uint32_t;

enum class Mode {
  CPU,
  OMP,
  OMP_NOATOMIC,
  CUDA,
  CUDA_NOATOMIC,
  CUDA_SHARED,
};

enum class AtomicTypeCuda {
  NONE,
  STANDARD,
  SHARED,
};

void computeHistogramCpu(const vector<uint8_t> &bytes,
                         array<HistType, 256> &histogram) {
  for (uint8_t i : bytes) {
    histogram[i]++;
  }
}

void computeHistogramOmp(const vector<uint8_t> &bytes,
                         array<HistType, 256> &histogram) {
  array<atomic<HistType>, 256> atomicHistogram;

#pragma omp parallel for
  for (size_t i = 0; i < 256; ++i) {
    atomicHistogram[i] = 0u;
  }

  size_t len = bytes.size();

#pragma omp parallel for
  for (size_t i = 0; i < len; ++i) {
    atomicHistogram[bytes[i]]++;
  }

#pragma omp parallel for
  for (size_t i = 0; i < 256; ++i) {
    histogram[i] = atomicHistogram[i];
  }
}

void computeHistogramOmpNoAtomic(const vector<uint8_t> &bytes,
                                 array<HistType, 256> &histogram) {
  size_t len = bytes.size();

#pragma omp parallel for
  for (size_t i = 0; i < len; ++i) {
    histogram[bytes[i]]++;
  }
}

__global__ void
_computeHistogramCudaNoAtomic(const uint8_t *__restrict__ bytes, size_t length,
                              HistType *__restrict__ histogram) {
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < length;
       i += stride) {
    histogram[bytes[i]]++;
  }
}

__global__ void
_computeHistogramCudaStandardAtomic(const uint8_t *__restrict__ bytes,
                                    size_t length,
                                    HistType *__restrict__ histogram) {

  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < length;
       i += stride) {
    atomicAdd(&(histogram[bytes[i]]), 1u);
  }
}

__global__ void
_computeHistogramCudaSharedAtomic(const uint8_t *__restrict__ bytes,
                                  size_t length,
                                  HistType *__restrict__ histogram) {
  __shared__ HistType temp[256];
  temp[threadIdx.x] = 0;
  __syncthreads(); // Zero this block's temporary array

  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < length;
       i += stride) {
    atomicAdd(&(temp[bytes[i]]), 1u);
    // Make a histogram for a fraction of the bytes
  }
  __syncthreads();

  // Now add up the histograms
  atomicAdd(&(histogram[threadIdx.x]), temp[threadIdx.x]);

  // Lesson: Don't let too many threads touch the same memory addresses at once
}

float computeHistogramCuda(const vector<uint8_t> &bytes,
                           array<HistType, 256> &histogram,
                           AtomicTypeCuda atomic) {
  CudaEvent start;
  CudaEvent finish;
  cudaDeviceProp deviceProperties;
  int device = 0;

  handle(cudaEventRecord(start.event, 0));

  handle(cudaGetDevice(&device));
  handle(cudaGetDeviceProperties(&deviceProperties, device));
  int blocks = deviceProperties.multiProcessorCount * 2;

  CudaMemory<uint8_t> cudaBytes(bytes.size());
  CudaMemory<HistType> cudaHistogram(256);

  handle(cudaMemcpy(cudaBytes.ptr, bytes.data(), bytes.size(),
                    cudaMemcpyHostToDevice));
  handle(cudaMemset(cudaHistogram.ptr, 0, 256 * sizeof(histogram[0])));

  switch (atomic) {
  case AtomicTypeCuda::NONE:
    _computeHistogramCudaNoAtomic<<<blocks, 256>>>(cudaBytes.ptr, bytes.size(),
                                                   cudaHistogram.ptr);
    break;
  case AtomicTypeCuda::STANDARD:
    _computeHistogramCudaStandardAtomic<<<blocks, 256>>>(
        cudaBytes.ptr, bytes.size(), cudaHistogram.ptr);
    break;
  case AtomicTypeCuda::SHARED:
    _computeHistogramCudaSharedAtomic<<<blocks, 256>>>(
        cudaBytes.ptr, bytes.size(), cudaHistogram.ptr);
    break;
  }

  handle(cudaGetLastError());

  handle(cudaMemcpy(histogram.data(), cudaHistogram.ptr,
                    256 * sizeof(histogram[0]), cudaMemcpyDeviceToHost));

  float duration = 0.0;
  handle(cudaEventRecord(finish.event, 0));
  handle(cudaEventSynchronize(finish.event));
  handle(cudaEventElapsedTime(&duration, start.event, finish.event));

  return duration;
}

using random_type = minstd_rand;

int main(int argc, char *argv[]) {
  CLI::App app("Example program that computes a histogram");

  random_type::result_type seed = 0;
  size_t size = 4096;
  Mode mode = Mode::CPU;
  string modeString = "CPU";
  try {
    app.add_option("--size", size, "Number of bytes to generate", true);
    app.add_option("--seed", seed, "Random seed", true);
    app.add_option("--mode", modeString, "Running mode (CPU, OMP, "
                                         "OMP_NOATOMIC, CUDA, CUDA_NOATOMIC, "
                                         "CUDA_SHARED)",
                   true)
        ->check([](const string &m) {
          return (m == "CPU" || m == "OMP" || m == "OMP_NOATOMIC" ||
                  m == "CUDA" || m == "CUDA_NOATOMIC" || m == "CUDA_SHARED");
        });
    app.parse(argc, argv);

    if (modeString == "CPU") {
      mode = Mode::CPU;
    } else if (modeString == "OMP") {
      mode = Mode::OMP;
    } else if (modeString == "OMP_NOATOMIC") {
      mode = Mode::OMP_NOATOMIC;
    } else if (modeString == "CUDA") {
      mode = Mode::CUDA;
    } else if (modeString == "CUDA_NOATOMIC") {
      mode = Mode::CUDA_NOATOMIC;
    } else if (modeString == "CUDA_SHARED") {
      mode = Mode::CUDA_SHARED;
    }

  } catch (CLI::Error &e) {
    return app.exit(e);
  }

  array<HistType, 256> histogram;
  vector<uint8_t> bytes;
  uniform_int_distribution<int> distribution(0, 255);
  random_type randomEngine(seed);
  bytes.resize(size, 0);

  for (size_t i = 0; i < size; ++i) {
    bytes[i] = static_cast<uint8_t>(distribution(randomEngine));
  }

  fill(histogram.begin(), histogram.end(), 0u);

  // Initialization
  float duration = 0.0f;
  auto cpuStart = chrono::high_resolution_clock::now();
  decltype(cpuStart) cpuFinish;
  switch (mode) {
  case Mode::CPU:
    computeHistogramCpu(bytes, histogram);
    cpuFinish = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(cpuFinish - cpuStart)
                   .count() /
               1000.0f;
    break;
  case Mode::OMP:
    computeHistogramOmp(bytes, histogram);
    cpuFinish = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(cpuFinish - cpuStart)
                   .count() /
               1000.0f;
    break;
  case Mode::OMP_NOATOMIC:
    computeHistogramOmpNoAtomic(bytes, histogram);
    cpuFinish = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(cpuFinish - cpuStart)
                   .count() /
               1000.0f;
    break;
  case Mode::CUDA:
    duration = computeHistogramCuda(bytes, histogram, AtomicTypeCuda::STANDARD);
    break;
  case Mode::CUDA_NOATOMIC:
    duration = computeHistogramCuda(bytes, histogram, AtomicTypeCuda::NONE);
    break;
  case Mode::CUDA_SHARED:
    duration = computeHistogramCuda(bytes, histogram, AtomicTypeCuda::SHARED);
    break;
  }

  size_t sum = accumulate(histogram.begin(), histogram.end(), 0u);
  for (int i = 0; i < 256; ++i) {
    cout << i << '\t' << histogram[i] << endl;
  }
  cout << duration << "ms" << endl;
  cout << "Total elements: " << sum << endl;
  if (sum != size) {
    cout << "WARNING: RACE CONDITIONS ENCOUNTERED!" << endl;
  }

  return 0;
}
