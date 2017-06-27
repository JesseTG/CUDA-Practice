find_package(SDL2 REQUIRED)

CUDA_SELECT_NVCC_ARCH_FLAGS(Auto)
include_directories(${SDL2_INCLUDE_DIR})
CUDA_ADD_EXECUTABLE(JuliaSet "main.cu")
target_link_libraries(JuliaSet ${SDL2_LIBRARIES} Common)
