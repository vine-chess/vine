#ifndef POLICY_NETWORK_CUDA_HPP
#define POLICY_NETWORK_CUDA_HPP

#include "compressed_mailbox.hpp"

#include "../chess/board_state.hpp"
#include "../util/types.hpp"

#include <array>

namespace network::policy {

constexpr i16 Q = 128;
constexpr usize L1_SIZE = 4096;
constexpr usize OUTPUT_SIZE = 3920;

} // namespace network::policy

namespace network::policy::cuda_detail {

constexpr usize FT_WEIGHT_COUNT = 2 * 2 * 2 * 6 * 64 * L1_SIZE;
constexpr usize FT_BIAS_COUNT = L1_SIZE;
constexpr usize L1_WEIGHT_COUNT = OUTPUT_SIZE * (L1_SIZE / 2);
constexpr usize L1_BIAS_COUNT = OUTPUT_SIZE;

struct CudaPolicyNetwork {
    std::array<i8, FT_WEIGHT_COUNT> ft_weights{};
    std::array<i8, FT_BIAS_COUNT> ft_biases{};
    std::array<i8, L1_WEIGHT_COUNT> l1_weights{};
    std::array<i8, L1_BIAS_COUNT> l1_biases{};
};

void export_cuda_network(CudaPolicyNetwork &dst);

struct CudaPolicyInput {
    network::cuda_common::CompressedMailbox pieces{};
    u32 move_offset = 0;
    u8 move_count = 0;
    u8 side_to_move = 0;
    u16 padding = 0;
};

} // namespace network::policy::cuda_detail

namespace network::policy {

using cuda_detail::CudaPolicyInput;

} // namespace network::policy

#endif // POLICY_NETWORK_CUDA_HPP
