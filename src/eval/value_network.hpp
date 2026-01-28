#ifndef VALUE_NETWORK_HPP
#define VALUE_NETWORK_HPP

#include "../chess/board_state.hpp"
#include "../util/multi_array.hpp"
#include "../util/simd.hpp"
#include <algorithm>

namespace network::value {

constexpr i16 QA = 255;
constexpr i16 QB = 64;
constexpr usize L1_SIZE = 1024;
constexpr usize L2_SIZE = 16;
constexpr usize L3_SIZE = 128;
constexpr usize L4_SIZE = 16;
constexpr usize VECTOR_SIZE = util::NATIVE_SIZE<i16>;
constexpr usize L2_REG_SIZE = std::min(util::NATIVE_SIZE<i16>, L2_SIZE);
constexpr usize L3_REG_SIZE = std::min(util::NATIVE_SIZE<f32>, L3_SIZE);
constexpr usize L4_REG_SIZE = std::min(util::NATIVE_SIZE<f32>, L4_SIZE);

constexpr i16 EVAL_SCALE = 400;

using i16Vec = util::SimdVector<i16, VECTOR_SIZE>;
using i8Vec = util::SimdVector<i8, VECTOR_SIZE>;

struct alignas(util::NATIVE_VECTOR_ALIGNMENT) ValueNetwork {
    union {
        util::MultiArray<i16Vec, 2, 2, 2, 6, 64, L1_SIZE / VECTOR_SIZE> ft_weights_vec;
        util::MultiArray<i16, 2, 2, 2, 6, 64, L1_SIZE> ft_weights;
    };
    util::MultiArray<i16, L1_SIZE> ft_biases;

    union {
        util::MultiArray<util::SimdVector<i8, L2_REG_SIZE>, L1_SIZE / 2, L2_SIZE / L2_REG_SIZE> l1_weights_vec;
        util::MultiArray<i8, L1_SIZE / 2, L2_SIZE> l1_weights;
    };
    util::MultiArray<f32, L2_SIZE> l1_biases;

    union {
        util::MultiArray<util::SimdVector<f32, L3_REG_SIZE>, L2_SIZE, L3_SIZE * 2 / L3_REG_SIZE> l2_weights_vec;
        util::MultiArray<f32, L2_SIZE, L3_SIZE * 2> l2_weights;
    };
    util::MultiArray<f32, L3_SIZE * 2> l2_biases;

    union {
        util::MultiArray<util::SimdVector<f32, L4_REG_SIZE>, L3_SIZE, L4_SIZE / L4_REG_SIZE> l3_weights_vec;
        util::MultiArray<f32, L3_SIZE, L4_SIZE> l3_weights;
    };
    util::MultiArray<f32, L4_SIZE> l3_biases;

    union {
        util::MultiArray<util::SimdVector<f32, L4_REG_SIZE>, L4_SIZE / L4_REG_SIZE> l4_weights_vec;
        util::MultiArray<f32, L4_SIZE> l4_weights;
    };
    util::MultiArray<f32, 1> l4_biases;
};

f64 evaluate(const BoardState &state);

} // namespace network::value

#endif // VALUE_NETWORK_HPP
