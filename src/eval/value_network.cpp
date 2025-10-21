#include "value_network.hpp"

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

    const i16 *l1 = reinterpret_cast<const i16 *>(accumulator.data());

    // activate l1
    std::array<i32, L1_SIZE / 2> l1_activated;
    for (usize i = 0; i < L1_SIZE / 2; ++i) {
        l1_activated[i] = std::clamp<i32>(l1[i], 0, QA) * std::clamp<i32>(l1[i + L1_SIZE / 2], 0, QA);
    }

    std::array<i32, L2_SIZE> l2i{};
    // l1 -> l2 matmul
    for (usize i = 0; i < L2_SIZE; ++i) {
        for (usize j = 0; j < L1_SIZE / 2; ++j) {
            l2i[i] += l1_activated[j] * network->l1_weights[i][j];
        }
    }
    std::array<f32, L2_SIZE> l2;
    for (usize i = 0; i < L2_SIZE; ++i) {
        l2[i] = l2i[i] * dequantisation_constant + network->l1_biases[i];
    }

    // activate l2
    for (usize i = 0; i < L2_SIZE / L2_REG_SIZE; ++i) {
        auto v = util::loadu<f32, L2_REG_SIZE>(l2.data() + L2_REG_SIZE * i);
        v = util::clamp_scalar<f32, L2_REG_SIZE>(v, 0, 1);
        v *= v;
        util::storeu(l2.data() + L2_REG_SIZE * i, v);
    }

    auto res = util::set1<f32, L3_REG_SIZE>(0);
    for (usize i = 0; i < L3_SIZE / L3_REG_SIZE; ++i) {
        auto v = util::loadu<f32, L3_REG_SIZE>(network->l2_biases.data() + L3_REG_SIZE * i);

        // l2 -> l3 matmul
        for (usize j = 0; j < L2_SIZE; ++j) {
            const auto l2_val = util::set1<f32, L3_REG_SIZE>(l2[j]);
            const auto w = network->l2_weights_vec[j][i];
            v = util::fmadd_ps(l2_val, w, v);
        }

        // activate l3
        v = util::clamp_scalar<f32, L3_REG_SIZE>(v, 0, 1);
        v *= v;

        // l3 -> out matmul
        res = util::fmadd_ps(v, network->l3_weights_vec[i], res);
    }

    return util::reduce_ps(res) + network->l3_biases[0];
}

} // namespace network::value
