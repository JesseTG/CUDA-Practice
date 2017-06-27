#include "common.hpp"

CudaEvent::CudaEvent() { handle(cudaEventCreate(&event)); }
CudaEvent::~CudaEvent() { handle(cudaEventDestroy(event)); }

SdlWindow::SdlWindow(const char *title, int x, int y, int w, int h,
                     Uint32 flags) {
  window = SDL_CreateWindow(title, x, y, w, h, flags);
  handleSdl(window != nullptr);

  surface = SDL_GetWindowSurface(window);
  handleSdl(surface != nullptr);

  renderer = SDL_GetRenderer(window);
  handleSdl(renderer != nullptr);

  texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB24,
                              SDL_TEXTUREACCESS_TARGET, w, h);
  handleSdl(texture != nullptr);
}

SdlWindow::~SdlWindow() {
  SDL_DestroyTexture(texture);
  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
}

SdlFont::SdlFont(const std::string &file, int ptsize) {
  font = TTF_OpenFont(file.c_str(), ptsize);
  handleSdl(font != nullptr);
}

SdlFont::~SdlFont() { TTF_CloseFont(font); }

GlBuffer::GlBuffer(GLenum t) : type(t) {
  handleGl(glGenBuffers(1, &id));
  handleGl(glBindBuffer(type, id));
}

GlBuffer::~GlBuffer() {
  handleGl(glBindBuffer(type, 0));
  handleGl(glDeleteBuffers(1, &id));
}
