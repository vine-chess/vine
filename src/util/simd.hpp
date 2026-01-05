#ifndef SIMD_HPP
#define SIMD_HPP

#include "types.hpp"
#include <cstring>
#include <smmintrin.h>

namespace util {

#if __x86_64__
#include <immintrin.h>
#if defined(__AVX512F__)
constexpr auto NATIVE_VECTOR_BYTES = 64;
#elif defined(__AVX2__)
constexpr auto NATIVE_VECTOR_BYTES = 32;
#elif defined(__SSE__)
constexpr auto NATIVE_VECTOR_BYTES = 16;
#else
constexpr auto NATIVE_VECTOR_BYTES = 0;
#endif
#elif defined(__arm__) || defined(__aarch64__)
#if defined(__ARM_NEON)
constexpr auto NATIVE_VECTOR_BYTES = 16;
#else
constexpr auto NATIVE_VECTOR_BYTES = 0;
#endif
#endif

// Useful for calculating sizes of structs aligned for simd purposes
[[nodiscard]] constexpr auto next_multiple(auto x, auto m) {
    return x + (m - x % m) % m;
}

namespace impl {

template <class T, usize N>
struct VectorHelper {
    using Type __attribute__((__vector_size__(N * sizeof(T)))) = T; // get around GCC restriction
};
} // namespace impl

constexpr auto NATIVE_VECTOR_ALIGNMENT = NATIVE_VECTOR_BYTES;

template <class T, usize N>
using SimdVector = typename impl::VectorHelper<T, N>::Type;

template <class T>
constexpr auto NATIVE_SIZE = NATIVE_VECTOR_BYTES / sizeof(T);

template <class T>
using NativeVector = SimdVector<T, NATIVE_SIZE<T>>;

template <class T = i16, usize N = NATIVE_SIZE<T>>
inline SimdVector<T, N> max(SimdVector<T, N> a, SimdVector<T, N> b) {
    return __builtin_elementwise_max(a, b);
}

template <class T = i16, usize N = NATIVE_SIZE<T>>
inline SimdVector<T, N> min(SimdVector<T, N> a, SimdVector<T, N> b) {
    return __builtin_elementwise_min(a, b);
}

template <class T = i16, usize N = NATIVE_SIZE<T>>
inline SimdVector<T, N> set1(T val) {
    std::array<T, N> vals;
    vals.fill(val);
    SimdVector<T, N> res;
    std::memcpy(&res, vals.data(), sizeof(res));
    return res;
}

template <class T = i16, usize N = NATIVE_SIZE<T>>
inline SimdVector<T, N> clamp_scalar(SimdVector<T, N> a, T lo, T hi) {
    return min<T, N>(set1<T, N>(hi), max<T, N>(set1<T, N>(lo), a));
}

template <class T = i16, usize N = NATIVE_SIZE<T>>
inline SimdVector<T, N> loadu(const T *ptr) {
    SimdVector<T, N> res;
    std::memcpy(&res, ptr, sizeof(res));
    return res;
}

template <class T = i16, usize N = NATIVE_SIZE<T>>
inline void storeu(T *ptr, SimdVector<T, N> v) {
    std::memcpy(ptr, &v, sizeof(v));
}

template <class To, class From, usize N>
inline SimdVector<To, N> convert_vector(SimdVector<From, N> v) {
    return __builtin_convertvector(v, SimdVector<To, N>);
}

template <class T, usize N>
inline SimdVector<T, N> select_vector64(SimdVector<T, N> a, SimdVector<T, N> b, auto m) {
    static_assert(sizeof(T) == 8, "T has to be a 64 bit type");

    const auto mask = __builtin_convertvector(m, SimdVector<i64, N>);

#if defined(__AVX512F__)
    if constexpr (N == 16) {
        return _mm512_mask_blend_pd(_mm512_cmpneq_epi64_mask(mask, _mm512_set1_epi64(0)), a, b);
    }
#endif
#if defined(__AVX2__)
    if constexpr (N == 8) {
        return _mm256_blendv_pd(a, b, mask);
    }
#endif
#if defined(__SSE2__)
    if constexpr (N == 4) {
        return _mm_blendv_pd(a, b, mask);
    }
#endif

    const auto ai = std::bit_cast<SimdVector<i64, N>>(a);
    const auto bi = std::bit_cast<SimdVector<i64, N>>(b);

    return std::bit_cast<SimdVector<T, N>>(ai & ~mask | bi & mask);
}

template <class T, usize N>
inline SimdVector<T, N> select_vector32(SimdVector<T, N> a, SimdVector<T, N> b, auto m) {
    static_assert(sizeof(T) == 4, "T has to be a 32 bit type");

    const auto mask = __builtin_convertvector(m, SimdVector<i32, N>);

#if defined(__AVX512F__)
    if constexpr (N == 16) {
        return _mm512_mask_blend_ps(_mm512_cmpneq_epi32_mask(mask, _mm512_set1_epi32(0)), a, b);
    }
#endif
#if defined(__AVX2__)
    if constexpr (N == 8) {
        return _mm256_blendv_ps(a, b, mask);
    }
#endif
#if defined(__SSE2__)
    if constexpr (N == 4) {
        return _mm_blendv_ps(a, b, mask);
    }
#endif

    const auto ai = std::bit_cast<SimdVector<i32, N>>(a);
    const auto bi = std::bit_cast<SimdVector<i32, N>>(b);

    return std::bit_cast<SimdVector<T, N>>(ai & ~mask | bi & mask);
}

template <class T, usize N>
inline SimdVector<T, N / 2> lower_half(SimdVector<T, N> v) {
    SimdVector<T, N / 2> res;
    std::memcpy(&res, &v, sizeof(res));
    return res;
}

template <class T, usize N>
inline SimdVector<T, N / 2> upper_half(SimdVector<T, N> v) {
    SimdVector<T, N / 2> res;
    std::memcpy(&res, reinterpret_cast<const T *>(&v) + N / 2, sizeof(res));
    return res;
}

template <class T, usize N>
inline T reduce_vector(SimdVector<T, N> v) {
    T vals[N];
    std::memcpy(vals, &v, sizeof(vals));
    T res = 0;
    for (auto val : vals) {
        res += val;
    }
    return res;
}

#ifdef __x86_64__
}
#include <immintrin.h>
namespace util {
#if defined(__AVX512F__)
inline SimdVector<i32, 16> madd_epi16(SimdVector<i16, 32> a, SimdVector<i16, 32> b) {
    return _mm512_madd_epi16(a, b);
}
inline SimdVector<f32, 16> fmadd_ps(SimdVector<f32, 16> a, SimdVector<f32, 16> b, SimdVector<f32, 16> c) {
    return _mm512_fmadd_ps(a, b, c);
}
inline f32 reduce_ps(SimdVector<f32, 16> v) {
    return _mm512_reduce_add_ps(v);
}
#endif
#if defined(__AVX2__)
inline SimdVector<i32, 8> madd_epi16(SimdVector<i16, 16> a, SimdVector<i16, 16> b) {
    return _mm256_madd_epi16(a, b);
}
inline SimdVector<f32, 8> fmadd_ps(SimdVector<f32, 8> a, SimdVector<f32, 8> b, SimdVector<f32, 8> c) {
    return _mm256_fmadd_ps(a, b, c);
}
inline f32 reduce_ps(SimdVector<f32, 8> v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(v), hi);
    __m128 shuf1 = _mm_castpd_ps(_mm_shuffle_pd(_mm_castps_pd(sum), _mm_castps_pd(sum), 1));
    sum = _mm_add_ps(sum, shuf1);
    __m128 shuf2 = _mm_movehdup_ps(sum);
    sum = _mm_add_ss(sum, shuf2);
    return _mm_cvtss_f32(sum);
}
#endif
#if defined(__SSE__)
inline SimdVector<i32, 4> madd_epi16(SimdVector<i16, 8> a, SimdVector<i16, 8> b) {
    return _mm_madd_epi16(a, b);
}
inline SimdVector<f32, 4> fmadd_ps(SimdVector<f32, 4> a, SimdVector<f32, 4> b, SimdVector<f32, 4> c) {
    return _mm_fmadd_ps(a, b, c);
}
inline f32 reduce_ps(SimdVector<f32, 4> v) {
    __m128 shuf1 = _mm_castpd_ps(_mm_shuffle_pd(_mm_castps_pd(v), _mm_castps_pd(v), 1));
    __m128 sum = _mm_add_ps(v, shuf1);
    __m128 shuf2 = _mm_movehdup_ps(sum);
    sum = _mm_add_ss(sum, shuf2);
    return _mm_cvtss_f32(sum);
}
#endif

#endif

#if defined(__arm__) || defined(__aarch64__)
#if defined(__ARM_NEON)
#include <arm_neon.h>

inline SimdVector<f32, 4> fmadd_ps(SimdVector<f32, 4> a, SimdVector<f32, 4> b, SimdVector<f32, 4> c) {
    return vfmaq_f32(c, b, a); // thanks to @kelseyde for helping me debug this, was the wrong order
}

inline SimdVector<i32, 4> madd_epi16(SimdVector<i16, 8> a, SimdVector<i16, 8> b) {
    int32x4_t mul_low = vmull_s16(vget_low_s16(a), vget_low_s16(b));
    int32x4_t mul_high = vmull_s16(vget_high_s16(a), vget_high_s16(b));

    return vaddq_s32(mul_low, mul_high);
}

#endif
#endif

} // namespace util

#endif // SIMD_HPP
