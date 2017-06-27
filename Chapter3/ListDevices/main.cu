#include <cuda_runtime.h>
#include <iostream>

#include "common.hpp"

using namespace std;

int main() {
  cudaDeviceProp deviceProperties;

  int numberOfDevices = 0;
  int driverVersion = 0;
  int runtimeVersion = 0;
  handle(cudaGetDeviceCount(&numberOfDevices));
  handle(cudaDriverGetVersion(&driverVersion));
  handle(cudaRuntimeGetVersion(&runtimeVersion));

  cout << "CUDA Driver Version: " << driverVersion << endl;
  cout << "CUDA Runtime Version: " << runtimeVersion << endl;
  cout << "# of Devices: " << numberOfDevices << endl;

  for (int i = 0; i < numberOfDevices; ++i) {
    handle(cudaGetDeviceProperties(&deviceProperties, i));

    cout << "Device " << i << " (" << deviceProperties.name << ") " << endl;

    cout << "\tECC Enabled: " << truthString(deviceProperties.ECCEnabled)
         << endl;

    cout << "\tAsync Engines: " << deviceProperties.asyncEngineCount << endl;

    cout << "\tCan Map Host Memory: "
         << truthString(deviceProperties.canMapHostMemory) << endl;

    cout << "\tClock Rate: " << deviceProperties.clockRate << "kHz" << endl;

    cout << "\tCompute Mode: " << deviceProperties.computeMode << endl;

    cout << "\tConcurrent Kernels: "
         << truthString(deviceProperties.concurrentKernels) << endl;

    cout << "\tConcurrent Managed Access: "
         << truthString(deviceProperties.concurrentManagedAccess) << endl;

    cout << "\tGlobal L1 Cache Support: "
         << truthString(deviceProperties.globalL1CacheSupported) << endl;

    cout << "\tHost Native Atomics: "
         << truthString(deviceProperties.hostNativeAtomicSupported) << endl;

    cout << "\tIntegrated Device: " << truthString(deviceProperties.integrated)
         << endl;

    cout << "\tMulti-GPU Board: "
         << truthString(deviceProperties.isMultiGpuBoard) << endl;

    cout << "\tKernel Timeout Enabled: "
         << truthString(deviceProperties.kernelExecTimeoutEnabled) << endl;

    cout << "\tL2 Cache Size: " << deviceProperties.l2CacheSize << "B" << endl;

    cout << "\tCompute Capability: " << deviceProperties.major << '.'
         << deviceProperties.minor << endl;

    cout << "\tManaged Memory Support: "
         << truthString(deviceProperties.managedMemory) << endl;

    cout << "\tMax Grid Size: " << deviceProperties.maxGridSize[0] << '*'
         << deviceProperties.maxGridSize[1] << '*'
         << deviceProperties.maxGridSize[2] << endl;

    cout << "\tMax 1D Surface Size: " << deviceProperties.maxSurface1D << endl;

    cout << "\tMax 1D Layered Surface Size: "
         << deviceProperties.maxSurface1DLayered[0] << '*'
         << deviceProperties.maxSurface1DLayered[1] << endl;

    cout << "\tMax 2D Surface Size: " << deviceProperties.maxSurface2D[0] << "*"
         << deviceProperties.maxSurface2D[1] << endl;

    cout << "\tMax 2D Layered Surface Size: "
         << deviceProperties.maxSurface2DLayered[0] << '*'
         << deviceProperties.maxSurface2DLayered[1] << '*'
         << deviceProperties.maxSurface2DLayered[2] << endl;

    cout << "\tMax 3D Surface Size: " << deviceProperties.maxSurface3D[0] << '*'
         << deviceProperties.maxSurface3D[1] << '*'
         << deviceProperties.maxSurface3D[2] << endl;

    cout << "\tMax Cubemap Size: " << deviceProperties.maxSurfaceCubemap
         << endl;

    cout << "\tMax Layered Cubemap Size: "
         << deviceProperties.maxSurfaceCubemapLayered[0] << '*'
         << deviceProperties.maxSurfaceCubemapLayered[1] << endl;

    cout << "\tMax 1D Texture Size: " << deviceProperties.maxTexture1D << endl;

    cout << "\tMax 1D Layered Texture Size: "
         << deviceProperties.maxTexture1DLayered[0] << '*'
         << deviceProperties.maxTexture1DLayered[1] << endl;

    cout << "\tMax 1D Linear Texture Size: "
         << deviceProperties.maxTexture1DLinear << endl;

    cout << "\tMax 1D Mipmapped Texture Size: "
         << deviceProperties.maxTexture1DMipmap << endl;

    cout << "\tMax 2D Texture Size: " << deviceProperties.maxTexture2D[0] << '*'
         << deviceProperties.maxTexture2D[1] << endl;

    cout << "\tMax 2D Texture Size w. Gather: " << deviceProperties.maxTexture2DGather[0] << '*'
         << deviceProperties.maxTexture2DGather[1] << endl;

    cout << "\tMax 2D Layered Texture Size: "
         << deviceProperties.maxTexture2DLayered[0] << '*'
         << deviceProperties.maxTexture2DLayered[1] << '*'
         << deviceProperties.maxTexture2DLayered[2] << endl;

    cout << "\tMax 2D Linear Texture Size: "
         << deviceProperties.maxTexture2DLinear[0] << '*'
         << deviceProperties.maxTexture2DLinear[1] << '*'
         << deviceProperties.maxTexture2DLinear[2] << endl;

    cout << "\tMax 2D Mipmapped Texture Size: "
         << deviceProperties.maxTexture2DMipmap[0] << '*'
         << deviceProperties.maxTexture2DMipmap[1] << endl;

    cout << "\tMax Block Dimensions: " << deviceProperties.maxThreadsDim[0]
         << '*' << deviceProperties.maxThreadsDim[1] << '*'
         << deviceProperties.maxThreadsDim[2] << endl;

    cout << "\tMax Threads per Block: " << deviceProperties.maxThreadsPerBlock
         << endl;

    cout << "\tMax Threads per Processor: "
         << deviceProperties.maxThreadsPerMultiProcessor << endl;

    cout << "\tMemory Pitch: " << deviceProperties.memPitch << "B" << endl;

    cout << "\tMemory Bus Width: " << deviceProperties.memoryBusWidth << " bits"
         << endl;

    cout << "\tMemory Clock Rate: " << deviceProperties.memoryClockRate << "kHz"
         << endl;

    cout << "\tMulti-GPU ID: " << deviceProperties.multiGpuBoardGroupID << endl;

    cout << "\tNumber of Processors: " << deviceProperties.multiProcessorCount
         << endl;

    cout << "\tPageable Memory Access: "
         << truthString(deviceProperties.pageableMemoryAccess) << endl;

    cout << "\tRegisters per Multiprocessor: "
         << deviceProperties.regsPerMultiprocessor << endl;

    cout << "\tRegisters per Block: " << deviceProperties.regsPerBlock << endl;

    cout << "\tShared Memory per Block: " << deviceProperties.sharedMemPerBlock
         << "B" << endl;

    cout << "\tSingle/Double Float Performance Ratio: "
         << deviceProperties.singleToDoublePrecisionPerfRatio << endl;

    cout << "\tTotal Constant Memory: " << deviceProperties.totalConstMem << "B"
         << endl;

    cout << "\tTotal Global Memory: " << deviceProperties.totalGlobalMem << "B"
         << endl;

    cout << "\tTCC Driver: " << truthString(deviceProperties.tccDriver) << endl;

    cout << "\tUnified Addressing: "
         << truthString(deviceProperties.unifiedAddressing) << endl;

    cout << "\tWarp Size: " << deviceProperties.warpSize << " threads" << endl;
  }

  return 0;
}
