#ifndef POLICY_NETWORK_HPP
#define POLICY_NETWORK_HPP

#include "../chess/board_state.hpp"
#include "../util/multi_array.hpp"
#include "../util/simd.hpp"
#include <array>
#include <span>

namespace network::policy {

constexpr i16 Q = 128;
constexpr usize L1_0_SIZE = 2048;
constexpr usize L1_1_SIZE = 16;
constexpr usize OUTPUT_SIZE = 3920;
constexpr usize VECTOR_SIZE = std::min<usize>(L1_1_SIZE, util::NATIVE_SIZE<i16>);

using i8Vec = util::SimdVector<i8, VECTOR_SIZE>;
using i16Vec = util::SimdVector<i16, VECTOR_SIZE>;

struct alignas(util::NATIVE_VECTOR_ALIGNMENT) PolicyNetwork {
    union {
        util::MultiArray<i8Vec, 2, 2, 2, 6, 64, L1_0_SIZE / VECTOR_SIZE> ft_weights_vec;
        util::MultiArray<i8, 2, 2, 2, 6, 64, L1_0_SIZE> ft_weights;
    };
    union {
        std::array<i8, L1_0_SIZE> ft_biases;
        std::array<i8Vec, L1_0_SIZE / VECTOR_SIZE> ft_biases_vec;
    };

    union {
        util::MultiArray<i8Vec, OUTPUT_SIZE, L1_0_SIZE / 2 / VECTOR_SIZE> l1_0_weights_vec;
        util::MultiArray<i8, OUTPUT_SIZE, L1_0_SIZE / 2> l1_0_weights;
    };
    std::array<i8, OUTPUT_SIZE> l1_0_biases;

    union {
        util::MultiArray<i8Vec, L1_1_SIZE / VECTOR_SIZE, L1_0_SIZE / 2 / VECTOR_SIZE> l1_1_weights_vec;
        util::MultiArray<i8, L1_1_SIZE, L1_0_SIZE / 2> l1_1_weights;
    };
    union {
        std::array<i8, L1_1_SIZE> l1_1_biases;
        std::array<i8, L1_1_SIZE / VECTOR_SIZE> l1_1_biases_vec;
    };

    union {
        util::MultiArray<i8Vec, OUTPUT_SIZE, L1_1_SIZE / VECTOR_SIZE> l2_weights_vec;
        util::MultiArray<i8, OUTPUT_SIZE, L1_1_SIZE> l2_weights;
    };
    std::array<i8, OUTPUT_SIZE> l2_biases;
};

class PolicyContext {
  public:
    // Build the feature accumulator for the given position (one-time per node)
    PolicyContext(const BoardState &state);

    // Raw score (logit) for a specific move in the position
    [[nodiscard]] f32 logit(Move move, PieceType moving_piece) const;

  private:
    Color stm_;
    Square king_sq_;
    std::array<util::SimdVector<i16, VECTOR_SIZE>, L1_0_SIZE / 2 / VECTOR_SIZE> activated_acc_{};
};

} // namespace network::policy

#endif // POLICY_NETWORK_HPP
