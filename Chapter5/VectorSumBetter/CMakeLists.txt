CUDA_SELECT_NVCC_ARCH_FLAGS(Auto)
CUDA_ADD_EXECUTABLE(VectorSumBetter "main.cu")
target_link_libraries(VectorSumBetter Common)
