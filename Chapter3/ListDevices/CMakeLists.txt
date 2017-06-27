CUDA_SELECT_NVCC_ARCH_FLAGS(Auto)
CUDA_ADD_EXECUTABLE(ListDevices "main.cu")

CUDA_BUILD_CLEAN_TARGET()
target_link_libraries(ListDevices Common)
