find_package(SDL2 REQUIRED)
include(FindPkgConfig)

PKG_SEARCH_MODULE(SDL2 REQUIRED sdl2)
PKG_SEARCH_MODULE(SDL2TTF REQUIRED SDL2_ttf>=2.0.0)

CUDA_SELECT_NVCC_ARCH_FLAGS(Auto)

include_directories(${SDL2_INCLUDE_DIR} ${SDL2TTF_INCLUDE_DIRS})
CUDA_ADD_EXECUTABLE(RippleAnimation "main.cu")
target_link_libraries(RippleAnimation ${SDL2_LIBRARIES} ${SDL2TTF_LIBRARIES} Common)
