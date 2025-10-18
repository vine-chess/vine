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

} // namespace math

} // namespace util

#endif // MATH
