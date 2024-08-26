#ifndef TACTICS_MATH_VEC_H
#define TACTICS_MATH_VEC_H

#include <algorithm>
#include <array>
#include <math.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

namespace tactics {

template <typename T, int N> struct Vec {
  using VecType = Vec<T, N>;
  std::array<T, N> value;
  VecType operator+(const VecType &lr) const {
    VecType dst;
    for (int i = 0; i < N; ++i) {
      dst.value[i] = value[i] + lr.value[i];
    }
    return dst;
  }
  VecType operator-(const VecType &lr) const {
    VecType dst;
    for (int i = 0; i < N; ++i) {
      dst.value[i] = value[i] - lr.value[i];
    }
    return dst;
  }
  Vec(std::array<T, N> &&v) { value = std::move(v); }
  VecType operator==(const VecType &lr) const {
    VecType dst;
    for (int i = 0; i < N; ++i) {
      if (value[i] == lr.value[i]) {
        dst.value[i] = 1;
      } else {
        dst.value[i] = 0;
      }
    }
    return dst;
  }
  VecType operator<(const VecType &lr) const {
    VecType dst;
    for (int i = 0; i < N; ++i) {
      if (value[i] < lr.value[i]) {
        dst.value[i] = 1;
      } else {
        dst.value[i] = 0;
      }
    }
    return dst;
  }
  VecType operator<=(const VecType &lr) const {
    VecType dst;
    for (int i = 0; i < N; ++i) {
      if (value[i] <= lr.value[i]) {
        dst.value[i] = 1;
      } else {
        dst.value[i] = 0;
      }
    }
    return dst;
  }
  VecType operator>(const VecType &lr) const {
    VecType dst;
    for (int i = 0; i < N; ++i) {
      if (value[i] > lr.value[i]) {
        dst.value[i] = 1;
      } else {
        dst.value[i] = 0;
      }
    }
    return dst;
  }
  VecType operator>=(const VecType &lr) const {
    VecType dst;
    for (int i = 0; i < N; ++i) {
      if (value[i] >= lr.value[i]) {
        dst.value[i] = 1;
      } else {
        dst.value[i] = 0;
      }
    }
    return dst;
  }
  VecType operator*(const VecType &lr) const {
    VecType dst;
    for (int i = 0; i < N; ++i) {
      dst.value[i] = value[i] * lr.value[i];
    }
    return dst;
  }
  VecType operator*(T lr) const {
    VecType dst;
    for (int i = 0; i < N; ++i) {
      dst.value[i] = value[i] * lr;
    }
    return dst;
  }
  VecType operator+=(const VecType &lr) {
    for (int i = 0; i < N; ++i) {
      value[i] = value[i] + lr.value[i];
    }
    return *this;
  }
  VecType operator-=(const VecType &lr) {
    for (int i = 0; i < N; ++i) {
      value[i] = value[i] - lr.value[i];
    }
    return *this;
  }

  VecType &operator=(const VecType &lr) {
    for (int i = 0; i < N; ++i) {
      value[i] = lr.value[i];
    }
    return *this;
  }
  VecType operator-() {
    VecType dst;
    for (int i = 0; i < N; ++i) {
      dst.value[i] = -value[i];
    }
    return dst;
  }
  Vec() {}
  Vec(const T v) {
    for (int i = 0; i < N; ++i) {
      value[i] = v;
    }
  }

  Vec(const VecType &lr) {
    for (int i = 0; i < N; ++i) {
      value[i] = lr.value[i];
    }
  }
  T operator[](size_t i) { return value[i]; }
  template <typename U> static VecType load(const U *addr) {
    VecType v;
    for (int i = 0; i < N; ++i) {
      v.value[i] = static_cast<T>(addr[i]);
    }
    return v;
  }
  template <typename U> static VecType broadcast(const U *addr) {
    VecType v;
    v.value[0] = static_cast<T>(addr[0]);
    for (int i = 1; i < N; ++i) {
      v.value[i] = v.value[0];
    }
    return v;
  }
  template <typename U> static void save(U *addr, const VecType &v) {
    for (int i = 0; i < N; ++i) {
      addr[i] = static_cast<U>(v.value[i]);
    }
  }
  static VecType max(const VecType &v1, const VecType &v2) {
    VecType dst;
    for (int i = 0; i < N; ++i) {
      dst.value[i] = std::max(v1.value[i], v2.value[i]);
    }
    return dst;
  }
  static VecType min(const VecType &v1, const VecType &v2) {
    VecType dst;
    for (int i = 0; i < N; ++i) {
      dst.value[i] = std::min(v1.value[i], v2.value[i]);
    }
    return dst;
  }
  static VecType fma(const VecType &v1, const VecType &v2, const VecType &v3) {
    return v1 + v2 * v3;
  }
  static VecType fms(const VecType &v1, const VecType &v2, const VecType &v3) {
    return v1 - v2 * v3;
  }
  static inline void transpose4(VecType &vec0, VecType &vec1, VecType &vec2,
                                VecType &vec3) {
    VecType source[4] = {vec0, vec1, vec2, vec3};
    for (int i = 0; i < N; ++i) {
      vec0.value[i] = source[i % 4].value[i >> 2];
      vec1.value[i] = source[i % 4].value[(i + N) >> 2];
      vec2.value[i] = source[i % 4].value[(i + 2 * N) >> 2];
      vec3.value[i] = source[i % 4].value[(i + 3 * N) >> 2];
    }
  }
};

template <> struct Vec<int32_t, 4> {
  using VecType = Vec<int32_t, 4>;
  using VecTypeArray = std::array<VecType, 4>;
  __m128i value;
  VecType operator+(const VecType &lr) const {
    VecType dst = {_mm_add_epi32(value, lr.value)};
    return dst;
  }
  VecType operator-(const VecType &lr) const {
    VecType dst = {_mm_sub_epi32(value, lr.value)};
    return dst;
  }
  VecType operator+=(const VecType &lr) {
    value = _mm_add_epi32(value, lr.value);
    return *this;
  }
  VecType operator-=(const VecType &lr) {
    value = _mm_sub_epi32(value, lr.value);
    return *this;
  }
  VecType operator*(const VecType &lr) const {
    VecType dst = {_mm_cvtps_epi32(
        _mm_mul_ps(_mm_cvtepi32_ps(value), _mm_cvtepi32_ps(lr.value)))};
    return dst;
  }

  VecType &operator=(const VecType &lr) {
    value = lr.value;
    return *this;
  }
  VecType operator==(const VecType &lr) const {
    __m128i one = _mm_set1_epi32(1);
    __m128i mask = _mm_cmpeq_epi32(value, lr.value);
    VecType dst = {_mm_and_si128(one, mask)};
    return dst;
  }
  VecType operator<(const VecType &lr) const {
    __m128i one = _mm_set1_epi32(1);
    __m128i mask = _mm_cmplt_epi32(value, lr.value);
    VecType dst = {_mm_and_si128(one, mask)};
    return dst;
  }
  VecType operator<=(const VecType &lr) const {
    __m128i one = _mm_set1_epi32(1);
    __m128i mask = _mm_cmpgt_epi32(value, lr.value);
    VecType dst = {_mm_andnot_si128(mask, one)};
    return dst;
  }
  VecType operator>(const VecType &lr) const {
    __m128i one = _mm_set1_epi32(1);
    __m128i mask = _mm_cmpgt_epi32(value, lr.value);
    VecType dst = {_mm_and_si128(one, mask)};
    return dst;
  }
  VecType operator>=(const VecType &lr) const {
    __m128i one = _mm_set1_epi32(1);
    __m128i mask = _mm_cmplt_epi32(value, lr.value);
    VecType dst = {_mm_andnot_si128(mask, one)};
    return dst;
  }
  VecType operator-() {
    VecType dst;
#if defined(_MSC_VER)
    dst.value = _mm_cvtps_epi32(_mm_xor_ps(
        _mm_cvtepi32_ps(value),
        _mm_set1_ps(-0.f))); // Using unary operation to SSE vec is GCC
                             // extension. We can not do this directly in MSVC.
#else
    dst.value = -value;
#endif
    return dst;
  }
  Vec() {}
  Vec(const float v) {
    int u = static_cast<int32_t>(v);
    value = _mm_set_epi32(u, u, u, u);
  }
  Vec(const int32_t v) { value = _mm_set_epi32(v, v, v, v); }
  Vec(__m128i &&v) { value = v; }
  Vec(__m128 &&v) { value = _mm_castps_si128(v); }
  Vec(const VecType &lr) { value = lr.value; }
  float operator[](size_t i) {
#if defined(_MSC_VER) // X64 native only mandatory support SSE and SSE2
                      // extension, and we can not find intrinsic function to
                      // extract element directly by index in SSE and SSE2
                      // extension.
    int32_t temp[4];
    _mm_storeu_si128((__m128i *)temp, value);
    return temp[i];
#else
    return value[i];
#endif
  }
  static VecType load(const int32_t *addr) {
    VecType v = {_mm_loadu_si128((__m128i const *)(addr))};
    return v;
  }
  static VecType broadcast(const int32_t *addr) {
    int32_t arr[4] = {*addr, 0, 0, 0};
    VecType dst = {_mm_loadu_si128((__m128i const *)(arr))};
    return dst;
  }
  static void save(int32_t *addr, const VecType &v) {
    _mm_storeu_si128((__m128i *)addr, v.value);
  }
  static VecType max(const VecType &v1, const VecType &v2) {
    VecType dst = {_mm_cvtps_epi32(
        _mm_max_ps(_mm_cvtepi32_ps(v1.value), _mm_cvtepi32_ps(v2.value)))};
    return dst;
  }
  static VecType min(const VecType &v1, const VecType &v2) {
    VecType dst = {_mm_cvtps_epi32(
        _mm_min_ps(_mm_cvtepi32_ps(v1.value), _mm_cvtepi32_ps(v2.value)))};
    return dst;
  }
  static VecType fma(const VecType &v1, const VecType &v2, const VecType &v3) {
    return v1 + v2 * v3; // TODO: use fma instruction
  }
  static VecType fms(const VecType &v1, const VecType &v2, const VecType &v3) {
    return v1 - v2 * v3; // TODO: use fma instruction
  }
  static inline void transpose4(VecType &vec0, VecType &vec1, VecType &vec2,
                                VecType &vec3) {
    __m128 tmp3, tmp2, tmp1, tmp0;
    tmp0 = _mm_unpacklo_ps(_mm_castsi128_ps(vec0.value),
                           _mm_castsi128_ps(vec1.value));
    tmp2 = _mm_unpacklo_ps(_mm_castsi128_ps(vec2.value),
                           _mm_castsi128_ps(vec3.value));
    tmp1 = _mm_unpackhi_ps(_mm_castsi128_ps(vec0.value),
                           _mm_castsi128_ps(vec1.value));
    tmp3 = _mm_unpackhi_ps(_mm_castsi128_ps(vec2.value),
                           _mm_castsi128_ps(vec3.value));
    vec0.value = _mm_castps_si128(_mm_movelh_ps(tmp0, tmp2));
    vec1.value = _mm_castps_si128(_mm_movehl_ps(tmp2, tmp0));
    vec2.value = _mm_castps_si128(_mm_movelh_ps(tmp1, tmp3));
    vec3.value = _mm_castps_si128(_mm_movehl_ps(tmp3, tmp1));
  }
};

template <> struct Vec<float, 4> {
  using VecType = Vec<float, 4>;
  using VecTypeInt32 = Vec<int32_t, 4>;
  using VecTypeArray = std::array<VecType, 4>;
  __m128 value;
  VecType operator+(const VecType &lr) const {
    VecType dst = {_mm_add_ps(value, lr.value)};
    return dst;
  }
  VecType operator-(const VecType &lr) const {
    VecType dst = {_mm_sub_ps(value, lr.value)};
    return dst;
  }
  VecType operator+=(const VecType &lr) {
    value = _mm_add_ps(value, lr.value);
    return *this;
  }
  VecType operator-=(const VecType &lr) {
    value = _mm_sub_ps(value, lr.value);
    return *this;
  }
  VecType operator*(const VecType &lr) const {
    VecType dst = {_mm_mul_ps(value, lr.value)};
    return dst;
  }
  VecType operator*(float lr) const {
    VecType dst = {_mm_mul_ps(value, _mm_set1_ps(lr))};
    return dst;
  }

  VecType &operator=(const VecType &lr) {
    value = lr.value;
    return *this;
  }
  VecType operator-() {
    VecType dst;
#if defined(_MSC_VER)
    dst.value = _mm_xor_ps(
        value,
        _mm_set1_ps(-0.f)); // Using unary operation to SSE vec is GCC
                            // extension. We can not do this directly in MSVC.
#else
    dst.value = -value;
#endif
    return dst;
  }
  VecType operator==(const VecType &lr) const {
    __m128i one = _mm_set1_epi32(1);
    __m128i mask =
        _mm_cmpeq_epi32(_mm_castps_si128(value), _mm_castps_si128(lr.value));
    VecType dst = {_mm_castsi128_ps(_mm_and_si128(one, mask))};
    return dst;
  }
  VecType operator<(const VecType &lr) const {
    __m128i one = _mm_set1_epi32(1);
    __m128i mask =
        _mm_cmplt_epi32(_mm_castps_si128(value), _mm_castps_si128(lr.value));
    VecType dst = {_mm_castsi128_ps(_mm_and_si128(one, mask))};
    return dst;
  }
  VecType operator<=(const VecType &lr) const {
    __m128i one = _mm_set1_epi32(1);
    __m128 mask = _mm_cmple_ps(value, lr.value);
    VecType dst = {
        _mm_castsi128_ps(_mm_and_si128(one, _mm_castps_si128(mask)))};
    return dst;
  }
  VecType operator>(const VecType &lr) const {
    __m128i one = _mm_set1_epi32(1);
    __m128 mask = _mm_cmpgt_ps(value, lr.value);
    VecType dst = {
        _mm_castsi128_ps(_mm_and_si128(one, _mm_castps_si128(mask)))};
    return dst;
  }
  VecType operator>=(const VecType &lr) const {
    __m128i one = _mm_set1_epi32(1);
    __m128 mask = _mm_cmpge_ps(value, lr.value);
    VecType dst = {
        _mm_castsi128_ps(_mm_and_si128(one, _mm_castps_si128(mask)))};
    return dst;
  }
  Vec() {}
  Vec(const float v) { value = _mm_set1_ps(v); }
  Vec(__m128 &&v) { value = v; }
  Vec(const VecType &lr) { value = lr.value; }
  float operator[](size_t i) {
#if defined(_MSC_VER) // X64 native only mandatory support SSE and SSE2
                      // extension, and we can not find intrinsic function to
                      // extract element directly by index in SSE and SSE2
                      // extension.
    float temp[4];
    _mm_storeu_ps(temp, value);
    return temp[i];
#else
    return value[i];
#endif
  }
  static VecType load(const float *addr) {
    VecType v = {_mm_loadu_ps(addr)};
    return v;
  }
  static VecType broadcast(const float *addr) {
    VecType dst = {_mm_load_ss(addr)};
    return dst;
  }
  static void save(float *addr, const VecType &v) {
    _mm_storeu_ps(addr, v.value);
  }
  static void save(float *addr, const VecTypeInt32 &v) {
    _mm_storeu_ps(addr, _mm_castsi128_ps(v.value));
  }
  static void save(int32_t *addr, const VecType &v) {
    _mm_storeu_si128((__m128i *)addr, _mm_castps_si128(v.value));
  }
  static VecType max(const VecType &v1, const VecType &v2) {
    VecType dst = {_mm_max_ps(v1.value, v2.value)};
    return dst;
  }
  static VecType min(const VecType &v1, const VecType &v2) {
    VecType dst = {_mm_min_ps(v1.value, v2.value)};
    return dst;
  }
  static VecType fma(const VecType &v1, const VecType &v2, const VecType &v3) {
    return v1 + v2 * v3; // TODO: use fma instruction
  }
  static VecType fms(const VecType &v1, const VecType &v2, const VecType &v3) {
    return v1 - v2 * v3; // TODO: use fma instruction
  }
  static inline void transpose4(VecType &vec0, VecType &vec1, VecType &vec2,
                                VecType &vec3) {
    __m128 tmp3, tmp2, tmp1, tmp0;
    tmp0 = _mm_unpacklo_ps((vec0.value), (vec1.value));
    tmp2 = _mm_unpacklo_ps((vec2.value), (vec3.value));
    tmp1 = _mm_unpackhi_ps((vec0.value), (vec1.value));
    tmp3 = _mm_unpackhi_ps((vec2.value), (vec3.value));
    vec0.value = _mm_movelh_ps(tmp0, tmp2);
    vec1.value = _mm_movehl_ps(tmp2, tmp0);
    vec2.value = _mm_movelh_ps(tmp1, tmp3);
    vec3.value = _mm_movehl_ps(tmp3, tmp1);
  }
};

// #endif

} // namespace tactics

#endif // TACTICS_MATH_VEC_H