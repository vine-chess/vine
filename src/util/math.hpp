#ifndef MATH
#define MATH

#include "types.hpp"

#include <bit>
#include <cmath>

namespace util {

namespace math {

inline f64 sigmoid(f64 x) {
    return 1.0 / (1.0 + std::exp(-x));
}

inline f64 inverse_sigmoid(f64 x) {
    return -std::log(1.0 / x - 1.0);
}

// uses the fact that inverse_sigmoid(x) == ln(x) - ln(1 - x)
// and a normal fast ln approximation
inline f32 fast_inverse_sigmoid(f32 x) {
    f32 z = 1.0f - x;
    i32 ix = std::bit_cast<i32>(x);
    i32 iz = std::bit_cast<i32>(z);
    constexpr f32 factor = 8.262958294867817e-08; // ln(2) / 2^23
    return static_cast<f32>(ix - iz) * factor;
}

inline f64 fast_inverse_sigmoid(f64 x) {
    f64 z = 1.0f - x;
    i64 ix = std::bit_cast<i64>(x);
    i64 iz = std::bit_cast<i64>(z);
    constexpr f64 scale = 1.539095918623324e-16; // ln(2) / 2^52
    return static_cast<f64>(ix - iz) * scale;
}

inline f32 fast_exp(f32 x) {
    constexpr f32 scale = 12102203.161561485; // 2^23 / ln(2)
    constexpr i32 bias = 1065353216;
    return std::bit_cast<f32>(static_cast<i32>(x * scale) + bias);
}

inline f32 fast_log_newton(f32 x) {
    i32 ix = std::bit_cast<i32>(x);
    constexpr f32 scale = 8.262958294867817e-08;
    constexpr i32 bias = 1065353216;
    f32 y = static_cast<f32>(ix - bias) * scale;
    f32 inverse_exp = fast_exp(-y);
    return y + x * inverse_exp - 1.0f;
}
inline f32 fast_inverse_sigmoid_newton(f32 x) {
    return fast_log_newton(x) - fast_log_newton(1.0f - x);
}

} // namespace math

} // namespace util

#endif // MATH
