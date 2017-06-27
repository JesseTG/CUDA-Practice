find_package(SDL2 REQUIRED)
include(FindPkgConfig)

PKG_SEARCH_MODULE(SDL2 REQUIRED sdl2)

CUDA_SELECT_NVCC_ARCH_FLAGS(Auto)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})
include_directories(${SDL2_INCLUDE_DIR})
CUDA_ADD_EXECUTABLE(DumbRayTracer "main.cu")
target_link_libraries(DumbRayTracer ${SDL2_LIBRARIES} Common)
