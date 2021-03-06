find_package(SDL2 REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
include(FindPkgConfig)

PKG_SEARCH_MODULE(SDL2 REQUIRED sdl2)
PKG_SEARCH_MODULE(SDL2TTF REQUIRED SDL2_ttf>=2.0.0)

CUDA_SELECT_NVCC_ARCH_FLAGS(Auto)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})
include_directories(${SDL2_INCLUDE_DIR} ${SDL2TTF_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIR})
CUDA_ADD_EXECUTABLE(StupidHeatTransfer
  "main.cu"
  Heater.hpp
  BaseHeatModel.hpp
  BaseHeatModel.cu
  BaseCudaHeatModel.hpp
  BaseCudaHeatModel.cu
  BaseCudaCpuCopyModel.hpp
  BaseCudaCpuCopyModel.cu
  CpuHeatModel.hpp
  CpuHeatModel.cpp
  CudaBufferHeatModel.hpp
  CudaBufferHeatModel.cu
  CudaOpenGLHeatModel.hpp
  CudaOpenGLHeatModel.cu
  Cuda1DTextureHeatModel.hpp
  Cuda1DTextureHeatModel.cu
  Cuda2DTextureHeatModel.hpp
  Cuda2DTextureHeatModel.cu
)
target_link_libraries(StupidHeatTransfer ${SDL2_LIBRARIES} ${SDL2TTF_LIBRARIES} ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} Common)
