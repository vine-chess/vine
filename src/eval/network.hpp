#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "../chess/board_state.hpp"
#include "../util/multi_array.hpp"
#include "../util/simd.hpp"
#include <array>

constexpr static auto QA = 64;
constexpr static auto QB = 255;
constexpr static auto SCALE = 400;
constexpr static auto L1_SIZE = 16;
constexpr static auto VECTOR_SIZE = std::min<usize>(L1_SIZE, util::NATIVE_SIZE<i16>);
using i16Vec = util::SimdVector<i16, VECTOR_SIZE>;

i32 evaluate(const BoardState &state);

struct Network {
    union {
        alignas(util::NATIVE_VECTOR_ALIGNMENT) util::MultiArray<i16, 768, L1_SIZE / VECTOR_SIZE> ft_weights_vec;
        alignas(util::NATIVE_VECTOR_ALIGNMENT) util::MultiArray<i16, 768, L1_SIZE> ft_weights;
    };
    alignas(util::NATIVE_VECTOR_ALIGNMENT) std::array<i16, L1_SIZE> ft_biases;
    alignas(util::NATIVE_VECTOR_ALIGNMENT) util::MultiArray<i16, 2, L1_SIZE> l1_weights;
    alignas(util::NATIVE_VECTOR_ALIGNMENT) std::array<i16, 1> l1_biases;
};

#endif // NETWORK_HPP
