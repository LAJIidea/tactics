#ifndef TACTICS_MATH_MATRIX_H
#define TACTICS_MATH_MATRIX_H

#include <memory>
#include "tactics/core/tensor.h"

namespace tactics {

class Matrix {
public:
  static Tensor* create_shape(int w, int h, void* data = nullptr);
  static Tensor* create(int w, int h);

  static void multi(Tensor* C, const Tensor* A, const Tensor* B);
  static void add(Tensor* C, const Tensor* A, const Tensor* B);
  static void sub(Tensor* C, const Tensor* A, const Tensor* B);
  static void dot(Tensor* C, const Tensor* A, const Tensor* B);
  // static void div_per_line(Tensor* C, const Tensor* A, const Tensor* B);
  static void invert(Tensor* dst, const Tensor* src);
  static void transpose(Tensor *dst, const Tensor* src);
  static void print(const Tensor* C, const char* head = "Matrix:");
  static void mul(Tensor* dst, const Tensor* src, const float scale);
};

} // namespace tactics

#endif // TACTICS_MATH_MATRIX_H