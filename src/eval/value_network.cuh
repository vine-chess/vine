#ifndef VALUE_NETWORK_CUDA_HPP
#define VALUE_NETWORK_CUDA_HPP

#include "compressed_mailbox.hpp"
#include "shared.hpp"

#include "../chess/board_state.hpp"
#include "../util/types.hpp"

#include <array>

namespace network::value {

namespace cuda_detail {

#ifdef MIXER_VALUE_NETWORK

struct CudaValueNetwork {
    std::array<i16, MIXER_FEATURE_COUNT * MIXER_SIZE> ft_weights{};
    std::array<i16, MIXER_SIZE> ft_biases{};
    std::array<f32, MIXER_D * MIXER_D> wl1{};
    std::array<f32, MIXER_D * MIXER_D> wr1{};
    std::array<f32, MIXER_D * MIXER_D> wl2{};
    std::array<f32, MIXER_D * MIXER_D> wr2{};
    std::array<f32, MIXER_SIZE> value_weights{};
    f32 value_bias = 0.0f;
};

#else

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

#endif

void export_cuda_network(CudaValueNetwork &dst);

} // namespace cuda_detail

struct CudaBoardInput {
    cuda_common::CompressedMailbox pieces{};
    u8 side_to_move = 0;
};

} // namespace network::value

#endif // VALUE_NETWORK_CUDA_HPP
