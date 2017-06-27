CUDA_SELECT_NVCC_ARCH_FLAGS(Auto)
CUDA_ADD_EXECUTABLE(DotProduct "main.cu")
target_link_libraries(DotProduct Common)
