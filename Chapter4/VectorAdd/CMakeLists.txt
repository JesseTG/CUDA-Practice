CUDA_SELECT_NVCC_ARCH_FLAGS(Auto)
CUDA_ADD_EXECUTABLE(VectorAdd "main.cu")

target_link_libraries(VectorAdd Common)
