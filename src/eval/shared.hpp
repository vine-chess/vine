#ifndef EVAL_SHARED_HPP
#define EVAL_SHARED_HPP

#include "../util/types.hpp"
#define MIXER_VALUE_NETWORK
#define MIXER_VALUE_DIM 16

#ifdef __CUDACC__
#define VINE_HOST __host__
#define VINE_DEVICE __device__
#else
#define VINE_HOST
#define VINE_DEVICE
#endif
#define VINE_HOST_DEVICE VINE_HOST VINE_DEVICE

namespace network::value {

#ifdef MIXER_VALUE_NETWORK

#ifndef MIXER_VALUE_DIM
#error "need to define MIXER_VALUE_DIM"
#endif

constexpr usize MIXER_D = MIXER_VALUE_DIM;
constexpr usize MIXER_SIZE = MIXER_D * MIXER_D;
constexpr usize MIXER_FEATURE_COUNT = 2 * 2 * 2 * 6 * 64;

namespace shared::mixer {

enum class MixSide {
    Left,
    Right,
};

[[nodiscard]] VINE_HOST_DEVICE constexpr usize feature_class(const usize defended, const usize threatened,
                                                            const usize opposite_color, const usize piece_idx) {
    return (((defended * 2 + threatened) * 2 + opposite_color) * 6 + piece_idx);
}

[[nodiscard]] VINE_HOST_DEVICE constexpr usize feature_index(const usize feature_class, const usize sq,
                                                            const usize flip) {
    return feature_class * 64 + (sq ^ flip);
}

[[nodiscard]] VINE_HOST_DEVICE constexpr usize feature_flip(const bool black_perspective, const bool king_on_right) {
    return (0b111000 * black_perspective) ^ (0b000111 * king_on_right);
}

[[nodiscard]] VINE_HOST_DEVICE constexpr usize matrix_index(const usize row, const usize col) {
    return col * MIXER_D + row; // took way too goddamn long to realize bullet was spitting out column major
}

[[nodiscard]] VINE_HOST_DEVICE constexpr f32 crelu(f32 value) {
    if (value < 0.0f) {
        value = 0.0f;
    }
    if (value > 1.0f) {
        value = 1.0f;
    }
    return value;
}

} // namespace shared::mixer

#endif

} // namespace network::value

#undef VINE_HOST_DEVICE
#undef VINE_DEVICE
#undef VINE_HOST

#endif // EVAL_SHARED_HPP
