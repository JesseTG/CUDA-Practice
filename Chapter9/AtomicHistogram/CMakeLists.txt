CUDA_SELECT_NVCC_ARCH_FLAGS(Auto)
CUDA_ADD_EXECUTABLE(AtomicHistogram "main.cu")

target_link_libraries(AtomicHistogram Common)
