#ifndef VALUE_NETWORK_CUDA_HPP
#define VALUE_NETWORK_CUDA_HPP

#include "../chess/board_state.hpp"
#include "../util/types.hpp"

#include <array>

namespace network::value::cuda_detail {

constexpr usize FT_WEIGHT_COUNT = 2 * 2 * 2 * 6 * 64 * 4096;
constexpr usize FT_BIAS_COUNT = 4096;
constexpr usize L1_WEIGHT_COUNT = (4096 / 2) * 16;
constexpr usize L1_BIAS_COUNT = 16;
constexpr usize L2_WEIGHT_COUNT = 16 * (128 * 2);
constexpr usize L2_BIAS_COUNT = 128 * 2;
constexpr usize L3_WEIGHT_COUNT = 128;

struct CudaValueNetwork {
    std::array<i16, FT_WEIGHT_COUNT> ft_weights{};
    std::array<i16, FT_BIAS_COUNT> ft_biases{};
    std::array<i8, L1_WEIGHT_COUNT> l1_weights{};
    std::array<f32, L1_BIAS_COUNT> l1_biases{};
    std::array<f32, L2_WEIGHT_COUNT> l2_weights{};
    std::array<f32, L2_BIAS_COUNT> l2_biases{};
    std::array<f32, L3_WEIGHT_COUNT> l3_weights{};
    f32 l3_bias = 0.0f;
};

void export_cuda_network(CudaValueNetwork &dst);

} // namespace network::value::cuda_detail

namespace network::value {

struct CudaBoardInput {
    ColoredPiece pieces[64]{};
    u8 side_to_move = 0;
};

} // namespace network::value

#endif // VALUE_NETWORK_CUDA_HPP
