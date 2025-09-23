#ifndef VALUE_NETWORK_HPP
#define VALUE_NETWORK_HPP

#include "../chess/board_state.hpp"
#include "../util/multi_array.hpp"
#include "../util/simd.hpp"
#include <array>

namespace network::value {

constexpr i16 QA = 255;
constexpr i16 QB = 64;
constexpr usize L1_SIZE = 1024;
constexpr usize VECTOR_SIZE = std::min<usize>(L1_SIZE, util::NATIVE_SIZE<i16>);
constexpr i16 EVAL_SCALE = 400;

using i16Vec = util::SimdVector<i16, VECTOR_SIZE>;

struct alignas(util::NATIVE_VECTOR_ALIGNMENT) ValueNetwork {
    union {
        util::MultiArray<i16Vec, 2, 6, 64, L1_SIZE / VECTOR_SIZE> ft_weights_vec;
        util::MultiArray<i16, 2, 6, 64, L1_SIZE> ft_weights;
    };
    std::array<i16, L1_SIZE> ft_biases;
    union {
        util::MultiArray<i16Vec, L1_SIZE / VECTOR_SIZE> l1_weights_vec;
        util::MultiArray<i16, L1_SIZE> l1_weights;
    };
    std::array<i16, 1> l1_biases;
};

f64 evaluate(const BoardState &state);

} // namespace network::value

#endif // VALUE_NETWORK_HPP
