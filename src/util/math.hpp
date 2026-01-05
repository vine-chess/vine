#ifndef MATH
#define MATH

#include "types.hpp"

#include <cmath>

namespace util {

namespace math {

inline f64 sigmoid(f64 x) {
    return 1.0 / (1.0 + std::exp(-x));
}

inline f64 inverse_sigmoid(f64 x) {
    return std::log(1.0 / (1.0 - x) - 1.0);
}

inline f32 fast_exp(f32 x) {
    constexpr f32 scale = 12102203.161561485;
    constexpr i32 bias = 127 * (1 << 23);
    return std::bit_cast<f32>(static_cast<i32>(x * scale) + bias);
};

} // namespace math

} // namespace util

#endif // MATH
