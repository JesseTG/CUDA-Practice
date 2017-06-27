#ifndef COMMON_HPP
#define COMMON_HPP

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef __CUDACC__

// Dummy definitions to pacify the syntax highlighter and autocomplete
const extern dim3 gridDim;
const extern uint3 blockIdx;
const extern dim3 blockDim;
const extern uint3 threadIdx;
const extern int warpSize;
__device__ void __syncthreads();

template <class T, int texType = cudaTextureType1D,
          enum cudaTextureReadMode mode = cudaReadModeElementType>
struct texture : textureReference {};

template <class T, int dim, enum cudaTextureReadMode readMode>
__host__ cudaError_t cudaBindTexture(size_t *offset,
                                     const texture<T, dim, readMode> &tex,
                                     const void *devPtr,
                                     size_t size = UINT_MAX);

template <class T, int dim, enum cudaTextureReadMode readMode>
__host__ cudaError_t cudaBindTexture2D(size_t *offset,
                                       const texture<T, dim, readMode> &tex,
                                       const void *devPtr,
                                       const cudaChannelFormatDesc &desc,
                                       size_t width, size_t height,
                                       size_t pitch);

template <class T, int dim, enum cudaTextureReadMode readMode>
__host__ cudaError_t cudaUnbindTexture(const texture<T, dim, readMode> &tex);

template <class T> __device__ T tex1Dfetch(cudaTextureObject_t texObj, int x);

template <class T>
__device__ T tex1Dfetch(texture<T, cudaTextureType1D, cudaReadModeElementType>,
                        int);

template <class T, enum cudaTextureReadMode readMode>
__device__ T tex2D(texture<T, cudaTextureType2D, readMode>, float, float);

template <class T> __device__ T atomicAdd(T *address, T val);
#endif

#define handle(error)                                                          \
  do {                                                                         \
    auto e = error;                                                            \
    if (e != cudaSuccess) {                                                    \
      std::ostringstream message;                                              \
      message << cudaGetErrorName(e) << " in " << __FILE__ << ":" << __func__  \
              << ":" << __LINE__ << ": " << cudaGetErrorString(e);             \
                                                                               \
      throw std::runtime_error(message.str());                                 \
    }                                                                          \
  } while (false)

#define handleSdl(error)                                                       \
  do {                                                                         \
    if (!(error)) {                                                            \
      std::ostringstream message;                                              \
      message << "Error in " << __FILE__ << ":" << __func__ << ":" << __LINE__ \
              << ": " << SDL_GetError();                                       \
                                                                               \
      throw std::runtime_error(message.str());                                 \
    }                                                                          \
  } while (false)

#define handleGl(call)                                                         \
  do {                                                                         \
    call;                                                                      \
    GLenum error = glGetError();                                               \
    if (error != GL_NO_ERROR) {                                                \
      std::ostringstream message;                                              \
      message << "Error in " << __FILE__ << ':' << __func__ << ':' << __LINE__ \
              << ": " << gluErrorString(error);                                \
                                                                               \
      throw std::runtime_error(message.str());                                 \
    }                                                                          \
  } while (false)

#define handleGlew(error)                                                      \
  do {                                                                         \
    auto e = error;                                                            \
    if (e != GLEW_OK) {                                                        \
      std::ostringstream message;                                              \
      message << "Error in " << __FILE__ << ':' << __func__ << ':' << __LINE__ \
              << ": " << glewGetErrorString(e);                                \
                                                                               \
      throw std::runtime_error(message.str());                                 \
    }                                                                          \
  } while (false)

constexpr const char *truthString(int i) { return (i == 0) ? "false" : "true"; }

constexpr bool isWindowClosed(const SDL_Event &e) {
  return e.type == SDL_QUIT ||
         (e.type == SDL_WINDOWEVENT && e.window.event == SDL_WINDOWEVENT_CLOSE);
}

template <class T> __host__ __device__ T clamp(T val, T min, T max) {
  if (val < min)
    return min;
  if (val > max)
    return max;
  return val;
}

struct CudaEvent {
  cudaEvent_t event;

  CudaEvent();
  ~CudaEvent();

  CudaEvent(const CudaEvent &) = delete;
  CudaEvent(CudaEvent &&) = delete;
  CudaEvent &operator=(const CudaEvent &) = delete;
  CudaEvent &operator=(CudaEvent &&) = delete;
};

template <class T> struct CudaMemory {
  T *ptr;
  size_t elements;

  CudaMemory() = delete;
  CudaMemory(size_t elements) {
    handle(cudaMalloc(&ptr, elements * sizeof(T)));
    this->elements = elements;
  }

  CudaMemory(const CudaEvent &) = delete;
  CudaMemory(CudaEvent &&) = delete;
  CudaMemory &operator=(const CudaEvent &) = delete;
  CudaMemory &operator=(CudaEvent &&) = delete;

  ~CudaMemory() { handle(cudaFree(ptr)); }
};

struct SdlWindow {
  SDL_Window *window;
  SDL_Renderer *renderer;
  SDL_Surface *surface;
  SDL_Texture *texture;

  SdlWindow(const char *title, int x, int y, int w, int h, Uint32 flags);
  ~SdlWindow();

  SdlWindow() = delete;
  SdlWindow(const SdlWindow &) = delete;
  SdlWindow(SdlWindow &&) = delete;
  SdlWindow &operator=(const SdlWindow &) = delete;
  SdlWindow &operator=(SdlWindow &&) = delete;
};

struct SdlFont {
  TTF_Font *font;

  SdlFont(const std::string &file, int ptsize);
  ~SdlFont();

  SdlFont() = delete;
  SdlFont(const SdlFont &) = delete;
  SdlFont(SdlFont &&) = delete;
  SdlFont &operator=(const SdlFont &) = delete;
  SdlFont &operator=(SdlFont &&) = delete;
};

struct GlBuffer {
  GLuint id;
  GLenum type;

  GlBuffer(GLenum t);
  ~GlBuffer();

  GlBuffer() = delete;
  GlBuffer(const GlBuffer &) = delete;
  GlBuffer(GlBuffer &&) = delete;
  GlBuffer &operator=(const GlBuffer &) = delete;
  GlBuffer &operator=(GlBuffer &&) = delete;
};

#endif // COMMON_HPP
