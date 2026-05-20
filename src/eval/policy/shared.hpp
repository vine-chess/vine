#ifndef EVAL_POLICY_SHARED_HPP
#define EVAL_POLICY_SHARED_HPP

#include "../board.hpp"

#include "../../util/types.hpp"

#include <array>

namespace network::policy {

constexpr i16 Q = 128;
constexpr usize L1_SIZE = 4096;
constexpr usize OUTPUT_SIZE = 3920;

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

bool cuda_available();
void evaluate_many(const CudaPolicyInput *inputs, const u16 *move_indices, f32 *outputs, usize position_count);

} // namespace network::policy

#endif // EVAL_POLICY_SHARED_HPP
