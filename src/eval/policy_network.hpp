#ifndef POLICY_NETWORK_HPP
#define POLICY_NETWORK_HPP

#include "../chess/board_state.hpp"
#include "../util/multi_array.hpp"
#include "../util/simd.hpp"
#include <array>

namespace network::policy {

constexpr i16   Q            = 128;
constexpr usize L1_SIZE      = 512;
constexpr usize L1_2_SIZE    = 16;
constexpr usize OUTPUT_SIZE  = 1880;

constexpr usize VECTOR_SIZE      = std::min<usize>(L1_SIZE,   util::NATIVE_SIZE<i16>);
constexpr usize VECTOR_SIZE_L1_2 = std::min<usize>(L1_2_SIZE, util::NATIVE_SIZE<i16>);

using i8Vec    = util::SimdVector<i8,  VECTOR_SIZE>;
using i16Vec   = util::SimdVector<i16, VECTOR_SIZE>;
using i8VecH2  = util::SimdVector<i8,  VECTOR_SIZE_L1_2>;
using i16VecH2 = util::SimdVector<i16, VECTOR_SIZE_L1_2>;

struct alignas(util::NATIVE_VECTOR_ALIGNMENT) PolicyNetwork {
    // ---- l0 ----
    union {
        util::MultiArray<i8,    2, 6, 64, L1_SIZE>               ft_weights;     // "l0w"
        util::MultiArray<i8Vec, 2, 6, 64, L1_SIZE / VECTOR_SIZE> ft_weights_vec;
    };
    union {
        std::array<i8,    L1_SIZE>               ft_biases;      // "l0b"
        std::array<i8Vec, L1_SIZE / VECTOR_SIZE> ft_biases_vec;
    };

    // ---- l1_1 ----
    union {
        util::MultiArray<i8,    OUTPUT_SIZE, L1_SIZE>               l1_1_weights;     // "l1_1w"
        util::MultiArray<i8Vec, OUTPUT_SIZE, L1_SIZE / VECTOR_SIZE> l1_1_weights_vec;
    };
    std::array<i8, OUTPUT_SIZE> l1_1_biases; // "l1_1b"

    // ---- l1_2 ----
    union {
        util::MultiArray<i8,    L1_2_SIZE, L1_SIZE>               l1_2_weights;     // "l1_2w"
        util::MultiArray<i8Vec, L1_2_SIZE, L1_SIZE / VECTOR_SIZE> l1_2_weights_vec;
    };
    std::array<i8, L1_2_SIZE> l1_2_biases; // "l1_2b"

    // ---- l2 ----
    union {
        util::MultiArray<i8,      OUTPUT_SIZE, L1_2_SIZE>                     l2_weights;     // "l2w"
        util::MultiArray<i8VecH2, OUTPUT_SIZE, L1_2_SIZE / VECTOR_SIZE_L1_2> l2_weights_vec;
    };
    std::array<i8, OUTPUT_SIZE> l2_biases; // "l2b"
};

class PolicyContext {
  public:
    // Build the feature accumulator for the given position (one-time per node)
    PolicyContext(const BoardState &state);

    // Raw score (logit) for a specific move in the position
    [[nodiscard]] f32 logit(Move move) const;

  private:
    Color stm_;
    std::array<i16Vec, L1_SIZE / VECTOR_SIZE> feature_accumulator_{};
};

} // namespace network::policy

#endif // POLICY_NETWORK_HPP