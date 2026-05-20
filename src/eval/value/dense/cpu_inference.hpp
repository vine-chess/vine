#ifndef EVAL_VALUE_DENSE_CPU_INFERENCE_HPP
#define EVAL_VALUE_DENSE_CPU_INFERENCE_HPP

#include "cpu.hpp"
#include "../../../chess/board_state.hpp"

#include <algorithm>
#include <array>
#include <cstring>

namespace network::value {

const extern ValueNetwork *const network;

namespace detail {

[[nodiscard]] const util::MultiArray<i16Vec, L1_SIZE / VECTOR_SIZE> &feature(Square sq, PieceType piece,
                                                                             Color piece_color, Color perspective,
                                                                             Square king_sq, Bitboard threats,
                                                                             Bitboard defences) {
    usize flip = 0b111000 * perspective ^ 0b000111 * (king_sq.file() >= File::E);
    return network
        ->ft_weights_vec[defences.is_set(sq)][threats.is_set(sq)][piece_color != perspective][piece - 1][sq ^ flip];
}

} // namespace detail

void export_cuda_network(CudaValueNetwork &dst) {
    std::ranges::copy(network->ft_weights.flat_span(), dst.ft_weights);
    std::ranges::copy(network->ft_biases, dst.ft_biases);
    std::ranges::copy(network->l1_weights.flat_span(), dst.l1_weights);
    std::ranges::copy(network->l1_biases, dst.l1_biases);
    std::ranges::copy(network->l2_weights.flat_span(), dst.l2_weights);
    std::ranges::copy(network->l2_biases, dst.l2_biases);
    std::ranges::copy(network->l3_weights, dst.l3_weights);
    dst.l3_bias = network->l3_biases[0];
}

f64 evaluate(const BoardState &state) {
    std::array<i16Vec, L1_SIZE / VECTOR_SIZE> accumulator;
    std::memcpy(accumulator.data(), network->ft_biases.data(), sizeof(accumulator));

    const auto stm = state.side_to_move;
    const auto king_sq = state.king(stm).lsb();

    const std::array<Bitboard, 2> threats = {state.pinned_threats_by(Color::WHITE),
                                             state.pinned_threats_by(Color::BLACK)};

    for (PieceType piece = PieceType::PAWN; piece <= PieceType::KING; piece = PieceType(piece + 1)) {
        for (auto sq : state.piece_bbs[piece - 1] & state.occupancy(stm)) {
            const auto feat = detail::feature(sq, piece, stm, stm, king_sq, threats[~stm], threats[stm]);
            for (usize i = 0; i < L1_SIZE / VECTOR_SIZE; ++i) {
                accumulator[i] += feat[i];
            }
        }

        for (auto sq : state.piece_bbs[piece - 1] & state.occupancy(~stm)) {
            const auto feat = detail::feature(sq, piece, ~stm, stm, king_sq, threats[stm], threats[~stm]);
            for (usize i = 0; i < L1_SIZE / VECTOR_SIZE; ++i) {
                accumulator[i] += feat[i];
            }
        }
    }

    const f32 dequantisation_constant = 1.0 / (QA * QA * QB);

    const i16 *l1 = reinterpret_cast<const i16 *>(accumulator.data());

    std::array<i32, L2_SIZE> l2_int{};
    for (usize i = 0; i < L1_SIZE / 2 / L2_REG_SIZE; ++i) {
        auto left = util::loadu<i16, L2_REG_SIZE>(l1 + L2_REG_SIZE * i);
        auto right = util::loadu<i16, L2_REG_SIZE>(l1 + L2_REG_SIZE * i + L1_SIZE / 2);

        left = util::clamp_scalar<i16, L2_REG_SIZE>(left, 0, QA);
        right = util::clamp_scalar<i16, L2_REG_SIZE>(right, 0, QA);

        const auto left_widened = util::convert_vector<u16, i16, L2_REG_SIZE>(left);
        const auto right_widened = util::convert_vector<u16, i16, L2_REG_SIZE>(right);

        const auto activated = left_widened * right_widened;

        for (usize j = 0; j < L2_REG_SIZE; ++j) {
            const auto idx = i * L2_REG_SIZE + j;
            for (usize k = 0; k < L2_SIZE; ++k) {
                l2_int[k] += activated[j] * network->l1_weights[idx][k];
            }
        }
    }

    std::array<f32, L2_SIZE> l2;
    for (usize i = 0; i < L2_SIZE; ++i) {
        l2[i] = l2_int[i] * dequantisation_constant + network->l1_biases[i];
    }

    for (usize i = 0; i < L2_SIZE / L2_REG_SIZE; ++i) {
        auto v = util::loadu<f32, L2_REG_SIZE>(l2.data() + L2_REG_SIZE * i);
        const auto scaled = util::fma<f32, L2_REG_SIZE>(v, util::set1<f32, L2_REG_SIZE>(1.0f / 6.0f),
                                                        util::set1<f32, L2_REG_SIZE>(0.5f));
        v *= util::clamp_scalar<f32, L2_REG_SIZE>(scaled, 0, 1);
        util::storeu<f32, L2_REG_SIZE>(l2.data() + L2_REG_SIZE * i, v);
    }

    std::array<f32, L3_SIZE> l3{};
    for (usize i = 0; i < L3_SIZE / L3_REG_SIZE; ++i) {
        auto v = util::loadu<f32, L3_REG_SIZE>(network->l2_biases.data() + L3_REG_SIZE * i);
        auto g = util::loadu<f32, L3_REG_SIZE>(network->l2_biases.data() + L3_REG_SIZE * i + L3_SIZE);

        for (usize j = 0; j < L2_SIZE; ++j) {
            const auto l2_val = util::set1<f32, L3_REG_SIZE>(l2[j]);
            const auto w1 = network->l2_weights_vec[j][i];
            const auto w2 = network->l2_weights_vec[j][i + L3_SIZE / L3_REG_SIZE];
            v = util::fma<f32, L3_REG_SIZE>(l2_val, w1, v);
            g = util::fma<f32, L3_REG_SIZE>(l2_val, w2, g);
        }

        const auto g_scaled = util::fma<f32, L3_REG_SIZE>(g, util::set1<f32, L3_REG_SIZE>(1.0f / 6.0f),
                                                          util::set1<f32, L3_REG_SIZE>(0.5f));
        v *= util::clamp_scalar<f32, L3_REG_SIZE>(g_scaled, 0, 1);

        const auto l3_val = util::loadu<f32, L3_REG_SIZE>(l3.data() + L3_REG_SIZE * i);
        util::storeu<f32, L3_REG_SIZE>(l3.data() + L3_REG_SIZE * i,
                                       util::fma<f32, L3_REG_SIZE>(v, network->l3_weights_vec[i], l3_val));
    }

    f32 final_sum = network->l3_biases[0];
    for (usize i = 0; i < L3_SIZE; ++i) {
        final_sum += l3[i];
    }
    return final_sum;
}

} // namespace network::value

#endif // EVAL_VALUE_DENSE_CPU_INFERENCE_HPP
