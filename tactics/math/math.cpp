#include "tactics/core/tensor.h"
#include "tactics/math/common.h"
#include "tactics/math/matrix.h"
#include "tactics/core/tensor_utils.h"
#include "tactics/core/memory_utils.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>

namespace tactics {

Tensor* Matrix::create_shape(int w, int h, void* data) {
  auto shape = new Tensor(2);
  shape->buffer().dim[0].extent = h;
  shape->buffer().dim[1].extent = w;
  TensorUtils::set_linear_layout(shape);
  shape->buffer().host = (uint8_t*) data;
  return shape;
}

Tensor* Matrix::create(int w, int h) {
  Tensor shape(2);
  shape.buffer().dim[0].extent = h;
  shape.buffer().dim[1].extent = w;
  auto result = new Tensor(&shape);
  TensorUtils::set_linear_layout(result);
  return result;
}

void Matrix::multi(Tensor *C, const Tensor *A, const Tensor *B) {
  assert(C != nullptr);
  assert(B != nullptr);
  assert(A != nullptr);

  assert(C->dimensions() == 2);
  assert(B->dimensions() == 2);
  assert(A->dimensions() == 2);

  const auto a = A->host<float>();
  const auto b = B->host<float>();
  auto c = C->host<float>();

  const int h = A->length(0);
  const int k = A->length(1);
  const int w = B->length(1);

  const int aw = A->stride(0);
  const int bw = B->stride(0);
  const int cw = C->stride(0);

  assert(k == B->length(0));

  int y = 0;
  for (; y < h; ++y) {
    int x = 0;
    const auto a_line = a + y * aw;
    auto c_line = c + y * cw;

    for (; x < w; ++x) {
      auto b_column = b + x;
      float sum = 0.0f;
      for (int i = 0; i < k; ++i) {
        sum += a_line[i] * b_column[i * bw];
      }
      c_line[x] = sum;
    }
  }
}

void Matrix::add(Tensor *C, const Tensor *A, const Tensor *B) {
  assert(C != nullptr);
  assert(B != nullptr);
  assert(A != nullptr);

  assert(A->size() == C->size());
  auto height = A->length(0);
  auto width = A->length(1);
  int b_offset = 0;
  if (B->dimensions() == A->dimensions()) {
    b_offset = B->stride(0);
    assert(B->length(1) == A->length(1));
    assert(B->length(0) == A->length(0));
  } else {
    b_offset = 0;
    assert(B->length(0) == A->length(1));
  }
  matrix_add_common(C->host<float>(), A->host<float>(), B->host<float>(), width, C->stride(0), A->stride(0), b_offset, height);
  return;
}

void Matrix::sub(Tensor *C, const Tensor *A, const Tensor *B) {
  assert(C != nullptr);
  assert(B != nullptr);
  assert(A != nullptr);

  assert(A->size() == C->size());
  auto height = A->length(0);
  auto width = A->length(1);
  int b_offset = 0;
  if (B->dimensions() == A->dimensions()) {
    b_offset = B->stride(0);
    assert(B->length(1) == A->length(1));
    assert(B->length(0) == A->length(0));
  } else {
    b_offset = 0;
    assert(B->length(0) == A->length(1));
  }
  matrix_sub_common(C->host<float>(), A->host<float>(), B->host<float>(), width, C->stride(0), A->stride(0), b_offset, height);
}

void Matrix::dot(Tensor *C, const Tensor *A, const Tensor *B) {
  assert(C != nullptr);
  assert(B != nullptr);
  assert(A != nullptr);
  assert(C->dimensions() == 2);
  assert(B->dimensions() == 2);
  assert(A->dimensions() == 2);
  assert(A->shape() == B->shape());
  assert(A->shape() == C->shape());
  const int height = A->length(0);
  const int width = A->length(1);

  const int aw = A->stride(0);
  const int bw = B->stride(0);
  const int cw = C->stride(0);
  matrix_prod_common(C->host<float>(), A->host<float>(), B->host<float>(), width, cw, aw, bw, height);
}

void Matrix::invert(Tensor *dst, const Tensor *src) {
  assert(src->buffer().dimensions == 2);
  const int N0 = src->buffer().dim[0].extent;
  assert(N0 == src->buffer().dim[1].extent);

  int i, j, k;
  float max, temp;
  std::shared_ptr<Tensor> temp_mat(Matrix::create(N0, N0));
  ::memcpy(temp_mat->buffer().host, src->buffer().host, src->size());
  const auto temp_data = temp_mat->host<float>();
  const auto dst_data  = dst->host<float>();
  for (i = 0; i < N0; ++i) {
    for (j = 0; j < N0; ++j) {
      *(dst_data + i * N0 + j) = (i == j) ? 1.0f : 0.0f;
    }
  }

  for (i = 0; i < N0; ++i) {
    max = *(temp_data + i * N0 + i);
    k = i;
    for (j = i + 1; j < N0; ++j) {
      auto data1 = *(temp_data + j * N0 + i);
      if (fabs(data1) > fabs(max)) {
        max = data1;
        k = j;
      }
    }
    if (k != i) {
      for (j = 0; j < N0; ++j) {
        temp = *(temp_data + i * N0 + j);
        *(temp_data + i * N0 + j) = *(temp_data + k * N0 + j);
        *(temp_data + k * N0 + j) = temp;
        temp = *(dst_data + i * N0 + j);
        *(dst_data + i * N0 + j) = *(dst_data + k * N0 + j);
        *(dst_data + k * N0 + j) = temp;
      }
    }
    if (*(temp_data + i * N0 + i) == 0) {
      printf("This matrix have no inverse!\n");
      return;
    }
    temp = *(temp_data + i * N0 + i);

    for (j = 0; j < N0; ++j) {
      *(temp_data + i * N0 + j) = *(temp_data + i * N0 + j) / temp;
      *(temp_data + i * N0 + j) = *(dst_data + i * N0 + j) / temp;
    }

    for (j = 0; j < N0; ++j) {
      if (j != i) {
        temp = *(temp_data + j * N0 + i);
        for (k = 0; k < N0; ++k) {
          *(temp_data + j * N0 + k) = *(temp_data + j * N0 + k) - *(temp_data + i * N0 + k) * temp;
          *(dst_data + j * N0 + k) = *(dst_data + j * N0 + k) - *(dst_data + i * N0 + k) * temp;
        }
      }
    }
  }
}

void Matrix::transpose(Tensor* dst, const Tensor *src) {
  auto a = src->host<float>();
  auto b = dst->host<float>();
  int as = src->buffer().dim[0].stride;
  int bs = dst->buffer().dim[0].stride;

  int w = dst->buffer().dim[1].extent;
  int h = dst->buffer().dim[0].extent;

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      b[bs * y + x] = a[as * x + y];
    }
  }
}

void Matrix::print(const Tensor *C, const char* head) {
  auto c = C->host<float>();
  auto w = C->buffer().dim[1].extent;
  for (int i = 2; i < C->dimensions(); ++i) {
    w *= C->length(i);
  }
  auto h = C->buffer().dim[0].extent;
  auto stride = C->buffer().dim[0].stride;

  printf("%s\n", head);

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      printf("%.7f\t", c[x + y * stride]);
    }
    printf("\n");
  }
}

void Matrix::mul(Tensor *dst, const Tensor *src, const float scale) {
  assert(dst != nullptr);
  assert(src != nullptr);
  assert(dst->dimensions() == 2);
  assert(src->dimensions() == 2);
  assert(src->shape() == dst->shape());
  const int height = src->length(0);
  const int width = src->length(1);

  const int sw = src->stride(0);
  const int dw = dst->stride(0);

  for (int y = 0; y < height; y++) {
    auto s = src->host<float>() + y * sw;
    auto d = dst->host<float>() + y * dw;
    int i = 0;

    for (; i < width; ++i) {
      d[i] = s[i] * scale;
    }
  }
}

} // namespace tactics