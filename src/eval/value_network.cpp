#include "value_network.hpp"

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

f64 evaluate(const BoardState &state) {
    std::array<i16Vec, L1_SIZE / VECTOR_SIZE> accumulator;
    std::memcpy(accumulator.data(), network->ft_biases.data(), sizeof(accumulator));

    const auto stm = state.side_to_move;
    const auto king_sq = state.king(stm).lsb();

    const std::array<Bitboard, 2> threats = {state.threats_by(Color::WHITE), state.threats_by(Color::BLACK)};

    // Accumulate features for both sides, viewed from side-to-move's perspective
    for (PieceType piece = PieceType::PAWN; piece <= PieceType::KING; piece = PieceType(piece + 1)) {
        // Our pieces
        for (auto sq : state.piece_bbs[piece - 1] & state.occupancy(stm)) {
            const auto feat = detail::feature(sq, piece, stm, stm, king_sq, threats[~stm], threats[stm]);
            for (usize i = 0; i < L1_SIZE / VECTOR_SIZE; ++i) {
                accumulator[i] += feat[i];
            }
        }

        // Opponent pieces
        for (auto sq : state.piece_bbs[piece - 1] & state.occupancy(~stm)) {
            const auto feat = detail::feature(sq, piece, ~stm, stm, king_sq, threats[stm], threats[~stm]);
            for (usize i = 0; i < L1_SIZE / VECTOR_SIZE; ++i) {
                accumulator[i] += feat[i];
            }
        }
    }

    const f32 dequantisation_constant = 1.0 / (QA * QA * QB);

    std::array<i16, L1_SIZE> l1;
    std::memcpy(l1.data(), accumulator.data(), sizeof(l1));

    // activate l1
    std::array<i32, L1_SIZE / 2> l1_activated;
    for (usize i = 0; i < L1_SIZE / 2; ++i) {
        l1_activated[i] = std::clamp<i32>(l1[i], 0, QA) * std::clamp<i32>(l1[i + L1_SIZE / 2], 0, QA);
    }
    // l1 -> l2 matmul
    std::array<f32, L2_SIZE> l2;
    std::memcpy(l2.data(), &network->l1_biases, sizeof(l2));
    for (usize j = 0; j < L2_SIZE; ++j) {
        for (usize i = 0; i < L1_SIZE / 2; ++i) {
            l2[j] += l1_activated[i] * network->l1_weights[j][i] * dequantisation_constant;
        }
    }

    // activate l2
    for (usize i = 0; i < L2_SIZE; ++i) {
        l2[i] = std::clamp<f32>(l2[i], 0, 1);
        l2[i] *= l2[i];
    }
    // l2 -> l3 matmul
    std::array<f32, L3_SIZE> l3;
    std::memcpy(l3.data(), &network->l2_biases, sizeof(l3));
    for (usize j = 0; j < L3_SIZE; ++j) {
        for (usize i = 0; i < L2_SIZE; ++i) {
            l3[j] += l2[i] * network->l2_weights[j][i];
        }
    }

    // activate l3
    for (usize i = 0; i < L3_SIZE; ++i) {
        l3[i] = std::clamp<f32>(l3[i], 0, 1);
        l3[i] *= l3[i];
    }
    // l3 -> out matmul
    f32 res = network->l3_biases[0];
    for (usize i = 0; i < L3_SIZE; ++i) {
        res += l3[i] * network->l3_weights[i];
    }

    return res;
}

} // namespace network::value
