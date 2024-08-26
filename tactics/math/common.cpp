#include "tactics/math/common.h"
#include "tactics/math/vec.h"
#include <algorithm>
#include <string>

using Vec4 = tactics::Vec<float, 4>;

void matrix_sub(float *C, const float *A, const float *B, size_t width, size_t c_stride, size_t as_stride, size_t b_stride, size_t height) {
  for (int y = 0; y < height; ++y) {
    auto a = A + as_stride * y;
    auto b = B + b_stride * y;
    auto c = C + c_stride * y;
    for (int x = 0; x < width; ++x) {
      Vec4::save(c + 4 * x, Vec4::load(a + 4 * x) - Vec4::load(b + 4 * x));
    }
  }
}

void matrix_add(float *C, const float *A, const float *B, size_t width, size_t c_stride, size_t a_stride, size_t b_stride, size_t height) {
  for (int y = 0; y < height; ++y) {
    auto a = A + a_stride * y;
    auto b = B + b_stride * y;
    auto c = C + c_stride * y;
    for (int x = 0; x < width; ++x) {
      Vec4::save(c + 4 * x, Vec4::load(a + 4 * x) + Vec4::load(b + 4 * x));
    }
  }
}

void matrix_prod(float *C, const float *A, const float *B, size_t width, size_t c_stride, size_t a_stride, size_t b_stride, size_t height) {
  for (int y = 0; y < height; ++y) {
    auto a = A + a_stride * y;
    auto b = B + b_stride * y;
    auto c = C + c_stride * y;
    for (int x = 0; x < width; ++x) {
      auto av = Vec4::load(a + 4 * x);
      auto bv = Vec4::load(b + 4 * x);
      Vec4::save(c + 4 * x, av * bv);
    }
  }
}

void matrix_add_common(float *C, const float *A, const float *B, size_t width, size_t c_stride, size_t a_stride, size_t b_stride, size_t height) {
  int width_c4 = (int)width / 4;
  if (width_c4 > 0) {
    matrix_add(C, A, B, width_c4, c_stride, a_stride, b_stride, height);
    width = width - 4 * width_c4;
    C = C + width_c4 * 4;
    A = A + width_c4 * 4;
    B = B + width_c4 * 4;
  }
  if (width > 0) {
    for (int y = 0; y < height; ++y) {
      auto a = A + a_stride * y;
      auto b = B + b_stride * y;
      auto c = C + c_stride * y;
      for (int x = 0; x < width; ++x) {
        c[x] = a[x] + b[x];
      }
    }
  }
}

void matrix_sub_common(float *C, const float *A, const float *B, size_t width, size_t c_stride, size_t a_stride, size_t b_stride, size_t height) {
  int width_c4 = (int)width / 4;
  if (width_c4 > 0) {
    matrix_sub(C, A, B, width_c4, c_stride, a_stride, b_stride, height);
    width = width - 4 * width_c4;
    C = C + width_c4 * 4;
    A = A + width_c4 * 4;
    B = B + width_c4 * 4;
  }
  if (width > 0) {
    for (int y = 0; y < height; ++y) {
      auto a = A + a_stride * y;
      auto b = B + b_stride * y;
      auto c = C + c_stride * y;
      for (int x = 0; x < width; ++x) {
        c[x] = a[x] + b[x];
      }
    }
  }
}

void matrix_prod_common(float *C, const float *A, const float *B, size_t width, size_t c_stride, size_t a_stride, size_t b_stride, size_t height) {
  int width_c4 = (int)width / 4;
  if (width_c4 > 0) {
    matrix_prod(C, A, B, width_c4, c_stride, a_stride, b_stride, height);
    width = width - 4 * width_c4;
    C = C + width_c4 * 4;
    A = A + width_c4 * 4;
    B = B + width_c4 * 4;
  }
  if (width > 0) {
    for (int y = 0; y < height; ++y) {
      auto a = A + a_stride * y;
      auto b = B + b_stride * y;
      auto c = C + c_stride * y;
      for (int x = 0; x < width; ++x) {
        c[x] = b[x] * a[x];
      }
    }
  }
}