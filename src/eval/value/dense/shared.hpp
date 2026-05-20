#ifndef EVAL_VALUE_DENSE_SHARED_HPP
#define EVAL_VALUE_DENSE_SHARED_HPP

#include "../../../util/types.hpp"
#include "../../compat.hpp"

namespace network::value {

[[nodiscard]] VINE_HOST_DEVICE inline f32 hard_sigmoid(f32 value) {
    f32 scaled = value / 6.0f + 0.5f;
    if (scaled < 0.0f) {
        return 0.0f;
    }
    if (scaled > 1.0f) {
        return 1.0f;
    }
    return scaled;
}

[[nodiscard]] VINE_HOST_DEVICE inline f32 hard_silu(f32 value) {
    return value * hard_sigmoid(value);
}

constexpr i16 QB = 64;
constexpr usize L1_SIZE = 4096;
constexpr usize L2_SIZE = 16;
constexpr usize L3_SIZE = 128;

constexpr usize FT_WEIGHT_COUNT = 2 * 2 * 2 * 6 * 64 * 4096;
constexpr usize FT_BIAS_COUNT = 4096;
constexpr usize L1_WEIGHT_COUNT = (4096 / 2) * 16;
constexpr usize L1_BIAS_COUNT = 16;
constexpr usize L2_WEIGHT_COUNT = 16 * (128 * 2);
constexpr usize L2_BIAS_COUNT = 128 * 2;
constexpr usize L3_WEIGHT_COUNT = 128;

struct CudaValueNetwork {
    i16 ft_weights[FT_WEIGHT_COUNT]{};
    i16 ft_biases[FT_BIAS_COUNT]{};
    i8 l1_weights[L1_WEIGHT_COUNT]{};
    f32 l1_biases[L1_BIAS_COUNT]{};
    f32 l2_weights[L2_WEIGHT_COUNT]{};
    f32 l2_biases[L2_BIAS_COUNT]{};
    f32 l3_weights[L3_WEIGHT_COUNT]{};
    f32 l3_bias = 0.0f;
};

void export_cuda_network(CudaValueNetwork &dst);

} // namespace network::value

#endif // EVAL_VALUE_DENSE_SHARED_HPP
