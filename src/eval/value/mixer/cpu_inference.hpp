#ifndef EVAL_VALUE_MIXER_CPU_INFERENCE_HPP
#define EVAL_VALUE_MIXER_CPU_INFERENCE_HPP

#include "cpu.hpp"
#include "../../../chess/board_state.hpp"
#include "../shared.hpp"

#include <algorithm>
#include <array>

namespace network::value {

const extern ValueNetwork *const network;

namespace detail {

using MixerVector = std::array<f32, MIXER_SIZE>;

[[nodiscard]] usize feature_index(Square sq, PieceType piece, Color piece_color, Color perspective, Square king_sq,
                                  Bitboard threats, Bitboard defences) {
    const usize flip = feature_flip(perspective == Color::BLACK, king_sq.file() >= File::E);
    const usize defended = defences.is_set(sq);
    const usize threatened = threats.is_set(sq);
    const usize opposite_color = piece_color != perspective;
    const usize fc = network::value::feature_class(defended, threatened, opposite_color, piece - 1);
    return network::value::feature_index(fc, static_cast<u8>(sq), flip);
}

void add_feature(std::array<i32, MIXER_SIZE> &accumulator, const usize feature_idx) {
    for (usize i = 0; i < MIXER_SIZE; ++i) {
        accumulator[i] += network->ft_weights[feature_idx][i];
    }
}

void apply_left_mix(MixerVector &x, const MixerLayer &layer) {
    std::array<f32, MIXER_INNER1 * MIXER_D2> up{};
    for (usize col = 0; col < MIXER_D2; ++col) {
        for (usize k = 0; k < MIXER_D1; ++k) {
            const f32 x_val = x[col * MIXER_D1 + k];
            for (usize row = 0; row < MIXER_INNER1; ++row) {
                up[col * MIXER_INNER1 + row] += layer.wl_up[k][row] * x_val;
            }
        }
    }
#if MIXER_DO_DOWN_PROJ
    for (auto &v : up)
        v = crelu(v);
    for (usize col = 0; col < MIXER_D2; ++col) {
        for (usize k = 0; k < MIXER_INNER1; ++k) {
            const f32 up_val = up[col * MIXER_INNER1 + k];
            for (usize row = 0; row < MIXER_D1; ++row) {
                x[col * MIXER_D1 + row] += layer.wl_down[k][row] * up_val;
            }
        }
    }
#else
    for (usize i = 0; i < MIXER_SIZE; ++i) {
        x[i] += crelu(up[i]);
    }
#endif
}

void apply_right_mix(MixerVector &x, const MixerLayer &layer) {
    std::array<f32, MIXER_D1 * MIXER_INNER2> up{};
    for (usize col = 0; col < MIXER_INNER2; ++col) {
        for (usize k = 0; k < MIXER_D2; ++k) {
            const f32 w_val = layer.wr_up[col][k];
            for (usize row = 0; row < MIXER_D1; ++row) {
                up[col * MIXER_D1 + row] += x[k * MIXER_D1 + row] * w_val;
            }
        }
    }
#if MIXER_DO_DOWN_PROJ
    for (auto &v : up)
        v = crelu(v);
    for (usize col = 0; col < MIXER_D2; ++col) {
        for (usize k = 0; k < MIXER_INNER2; ++k) {
            const f32 w_val = layer.wr_down[col][k];
            for (usize row = 0; row < MIXER_D1; ++row) {
                x[col * MIXER_D1 + row] += up[k * MIXER_D1 + row] * w_val;
            }
        }
    }
#else
    for (usize i = 0; i < MIXER_SIZE; ++i) {
        x[i] += crelu(up[i]);
    }
#endif
}

} // namespace detail

void export_cuda_network(CudaValueNetwork &dst) {
    std::ranges::copy(network->ft_weights.flat_span(), dst.ft_weights);
    std::ranges::copy(network->ft_biases, dst.ft_biases);
    for (usize i = 0; i < MIXER_NUM_LAYERS; ++i) {
        const auto &src_layer = network->layers[i];
        auto &dst_layer = dst.layers[i];

        std::ranges::copy(src_layer.wl_up.flat_span(), dst_layer.wl_up);
        std::ranges::copy(src_layer.wr_up.flat_span(), dst_layer.wr_up);
#if MIXER_DO_DOWN_PROJ
        std::ranges::copy(src_layer.wl_down.flat_span(), dst_layer.wl_down);
        std::ranges::copy(src_layer.wr_down.flat_span(), dst_layer.wr_down);
#endif
    }
    std::ranges::copy(network->value_weights, dst.value_weights);
    dst.value_bias = network->value_bias;
}

f64 evaluate(const BoardState &state) {
    std::array<i32, MIXER_SIZE> accumulator{};
    for (usize i = 0; i < MIXER_SIZE; ++i) {
        accumulator[i] = network->ft_biases[i];
    }

    const auto stm = state.side_to_move;
    const auto king_sq = state.king(stm).lsb();
    const std::array<Bitboard, 2> threats = {state.pinned_threats_by(Color::WHITE),
                                             state.pinned_threats_by(Color::BLACK)};

    // Accumulate features for both sides, viewed from side-to-move's perspective
    for (PieceType piece = PieceType::PAWN; piece <= PieceType::KING; piece = PieceType(piece + 1)) {
        // Our pieces
        for (auto sq : state.piece_bbs[piece - 1] & state.occupancy(stm)) {
            detail::add_feature(accumulator,
                                detail::feature_index(sq, piece, stm, stm, king_sq, threats[~stm], threats[stm]));
        }

        // Opponent pieces
        for (auto sq : state.piece_bbs[piece - 1] & state.occupancy(~stm)) {
            detail::add_feature(accumulator,
                                detail::feature_index(sq, piece, ~stm, stm, king_sq, threats[stm], threats[~stm]));
        }
    }

    // Clamp to [0, 1] (quantized)
    std::array<f32, MIXER_SIZE> x{};
    for (usize i = 0; i < MIXER_SIZE; ++i) {
        x[i] = static_cast<f32>(std::clamp(accumulator[i], 0, static_cast<i32>(QA))) / static_cast<f32>(QA);
    }

    // Run the mixer layers
    for (usize i = 0; i < MIXER_NUM_LAYERS; ++i) {
        detail::apply_left_mix(x, network->layers[i]);
        detail::apply_right_mix(x, network->layers[i]);
    }

    // Final output projection
    f32 value = network->value_bias;
    for (usize i = 0; i < MIXER_SIZE; ++i) {
        value += x[i] * network->value_weights[i];
    }
    return value;
}

} // namespace network::value

#endif // EVAL_VALUE_MIXER_CPU_INFERENCE_HPP
