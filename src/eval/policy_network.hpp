#ifndef POLICY_NETWORK_HPP
#define POLICY_NETWORK_HPP

#include "../chess/board_state.hpp"
#include "../util/multi_array.hpp"
#include "../util/simd.hpp"
#include <array>

namespace network::policy {

constexpr i16 Q = 128;
constexpr usize L1_SIZE = 256;
constexpr usize OUTPUT_SIZE = 1880;
constexpr usize VECTOR_SIZE = std::min<usize>(L1_SIZE, util::NATIVE_SIZE<i16>);

using i8Vec = util::SimdVector<i8, VECTOR_SIZE>;
using i16Vec = util::SimdVector<i16, VECTOR_SIZE>;

struct alignas(util::NATIVE_VECTOR_ALIGNMENT) PolicyNetwork {
    union {
        util::MultiArray<i8Vec, 2, 6, 64, L1_SIZE / VECTOR_SIZE> ft_weights_vec;
        util::MultiArray<i8, 2, 6, 64, L1_SIZE> ft_weights;
    };
    union {
        std::array<i8, L1_SIZE> ft_biases;
        std::array<i8Vec, L1_SIZE / VECTOR_SIZE> ft_biases_vec;
    };
    union {
        util::MultiArray<i8Vec, OUTPUT_SIZE, L1_SIZE / VECTOR_SIZE> l1_weights_vec;
        util::MultiArray<i8, OUTPUT_SIZE, L1_SIZE> l1_weights;
    };
    std::array<i8, OUTPUT_SIZE> l1_biases;
};

class PolicyContext {
  public:
    // Build the feature accumulator for the given position (one-time per node)
    PolicyContext(const BoardState& state);

    // Raw score (logit) for a specific move in the position
    [[nodiscard]] f32 logit(Move move) const;

  private:
    Color stm_;
    std::array<i16Vec, L1_SIZE / VECTOR_SIZE> feature_accumulator_{};
};

} // namespace network::policy

#endif // POLICY_NETWORK_HPP