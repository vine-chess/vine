#ifndef SIMD_HPP
#define SIMD_HPP

#include "types.hpp"
#include <cstring>

namespace util {

#if __x86_64__
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
[[nodiscard]] constexpr auto nextMultiple(auto x, auto m) {
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

template <usize N = NATIVE_SIZE<i16>>
inline SimdVector<i16, N> max_epi16(SimdVector<i16, N> a, SimdVector<i16, N> b) {
    return __builtin_elementwise_max(a, b);
}

template <usize N = NATIVE_SIZE<i16>>
inline SimdVector<i16, N> min_epi16(SimdVector<i16, N> a, SimdVector<i16, N> b) {
    return __builtin_elementwise_min(a, b);
}

template <usize N = NATIVE_SIZE<i16>>
inline SimdVector<i16, N> loadu_epi16(const i16 *ptr) {
    SimdVector<i16, N> res;
    std::memcpy(&res, ptr, sizeof(res));
    return res;
}

template <usize N = NATIVE_SIZE<i16>>
inline SimdVector<i16, N> set1_epi16(i16 val) {
    std::array<i16, N> vals;
    vals.fill(val);
    SimdVector<i16, N> res;
    std::memcpy(&res, vals.data(), sizeof(res));
    return res;
}

template <class To, class From, usize N>
inline SimdVector<To, N> convert_vector(SimdVector<From, N> v) {
    return __builtin_convertvector(v, SimdVector<To, N>);
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
#endif
#if defined(__AVX2__)
inline SimdVector<i32, 8> madd_epi16(SimdVector<i16, 16> a, SimdVector<i16, 16> b) {
    return _mm256_madd_epi16(a, b);
}
#endif
#if defined(__SSE__)
inline SimdVector<i32, 4> madd_epi16(SimdVector<i16, 8> a, SimdVector<i16, 8> b) {
    return _mm_madd_epi16(a, b);
}
#endif

#endif

#if defined(__arm__) || defined(__aarch64__)
#if defined(__ARM_NEON)
#include <arm_neon.h>

inline SimdVector<i32, 4> madd_epi16(SimdVector<i16, 8> a, SimdVector<i16, 8> b) {
    int32x4_t mul_low = vmull_s16(vget_low_s16(a), vget_low_s16(b));
    int32x4_t mul_high = vmull_s16(vget_high_s16(a), vget_high_s16(b));

    return vaddq_s32(mul_low, mul_high);
}

#endif
#endif

} // namespace util

#endif // SIMD_HPP
