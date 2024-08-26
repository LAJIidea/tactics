#ifndef TACTICS_MATH_COMMON_H
#define TACTICS_MATH_COMMON_H

#include <cstddef>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void matrix_add(float* C, const float* A, const float* B, size_t width, size_t c_stride, size_t a_stride,
                size_t b_stride, size_t height);
void matrix_sub(float* C, const float* A, const float* B, size_t width, size_t c_stride, size_t as_stride,
                size_t b_stride, size_t height);
void matrix_prod(float *C, const float* A, const float* B, size_t width, size_t c_stride, size_t a_stride,
                 size_t b_stride, size_t height);
void matrix_add_common(float* C, const float* A, const float* B, size_t width, size_t c_stride, size_t a_stride,
                       size_t b_stride, size_t height);
void matrix_sub_common(float* C, const float* A, const float* B, size_t width, size_t c_stride, size_t a_stride,
                       size_t b_stride, size_t height);
void matrix_prod_common(float* C, const float* A, const float* B, size_t width, size_t c_stride, size_t a_stride,
                        size_t b_stride, size_t height);

#ifdef __cplusplus
}
#endif

#endif // TACTICS_MATH_COMMON_H