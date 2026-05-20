#ifndef EVAL_VALUE_MIXER_SHARED_HPP
#define EVAL_VALUE_MIXER_SHARED_HPP

#include "../../../util/types.hpp"
#include "../../compat.hpp"

#ifndef MIXER_VALUE_D1
#define MIXER_VALUE_D1 16
#endif
#ifndef MIXER_VALUE_D2
#define MIXER_VALUE_D2 16
#endif
#ifndef MIXER_VALUE_INNER1
#define MIXER_VALUE_INNER1 MIXER_VALUE_D1
#endif
#ifndef MIXER_VALUE_INNER2
#define MIXER_VALUE_INNER2 MIXER_VALUE_D2
#endif
#ifndef MIXER_VALUE_NUM_LAYERS
#define MIXER_VALUE_NUM_LAYERS 2
#endif

#ifndef MIXER_DO_DOWN_PROJ
#define MIXER_DO_DOWN_PROJ (MIXER_VALUE_INNER1 != MIXER_VALUE_D1 || MIXER_VALUE_INNER2 != MIXER_VALUE_D2)
#endif

namespace network::value {

constexpr usize MIXER_D1 = MIXER_VALUE_D1;
constexpr usize MIXER_D2 = MIXER_VALUE_D2;
constexpr usize MIXER_INNER1 = MIXER_VALUE_INNER1;
constexpr usize MIXER_INNER2 = MIXER_VALUE_INNER2;
constexpr usize MIXER_NUM_LAYERS = MIXER_VALUE_NUM_LAYERS;
constexpr usize MIXER_SIZE = MIXER_D1 * MIXER_D2;
constexpr usize MIXER_TILE_SIZE = 16 * 16;
constexpr usize MIXER_FEATURE_COUNT = 2 * 2 * 2 * 6 * 64;

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

[[nodiscard]] VINE_HOST_DEVICE constexpr usize matrix_index(const usize row, const usize col, const usize num_rows) {
    return col * num_rows + row;
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

struct CudaMixerLayer {
    f32 wl_up[MIXER_D1 * MIXER_INNER1]{};
#if MIXER_DO_DOWN_PROJ
    f32 wl_down[MIXER_INNER1 * MIXER_D1]{};
#endif
    f32 wr_up[MIXER_INNER2 * MIXER_D2]{};
#if MIXER_DO_DOWN_PROJ
    f32 wr_down[MIXER_D2 * MIXER_INNER2]{};
#endif
};

struct CudaValueNetwork {
    i16 ft_weights[MIXER_FEATURE_COUNT * MIXER_SIZE]{};
    i16 ft_biases[MIXER_SIZE]{};
    CudaMixerLayer layers[MIXER_NUM_LAYERS]{};
    f32 value_weights[MIXER_SIZE]{};
    f32 value_bias = 0.0f;
};

void export_cuda_network(CudaValueNetwork &dst);

} // namespace network::value

#endif // EVAL_VALUE_MIXER_SHARED_HPP
