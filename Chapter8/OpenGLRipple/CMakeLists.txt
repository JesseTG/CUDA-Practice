find_package(SDL2 REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
include(FindPkgConfig)

PKG_SEARCH_MODULE(SDL2 REQUIRED sdl2)

CUDA_SELECT_NVCC_ARCH_FLAGS(Auto)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})
include_directories(${SDL2_INCLUDE_DIR} ${GLEW_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIR})
CUDA_ADD_EXECUTABLE(OpenGLRipple "main.cu")
target_link_libraries(OpenGLRipple ${SDL2_LIBRARIES} ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} Common)
