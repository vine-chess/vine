#include "value_network.hpp"

#include <array>
#include <cstring>
#include <iostream>

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

    util::SimdVector<i32, VECTOR_SIZE / 2> sum{};
    const auto zero = util::set1_epi16<VECTOR_SIZE>(0);
    const auto one = util::set1_epi16<VECTOR_SIZE>(QA);

    for (usize i = 0; i < L1_SIZE / VECTOR_SIZE; ++i) {
        const auto clamped = util::min_epi16<VECTOR_SIZE>(util::max_epi16<VECTOR_SIZE>(accumulator[i], zero), one);
        sum += util::madd_epi16(clamped, clamped * network->l1_weights_vec[i]);
    }

    const i32 dot = util::reduce_vector<i32, VECTOR_SIZE / 2>(sum);
    const i32 bias = network->l1_biases[0];
    return (dot / static_cast<f64>(QA) + bias) / static_cast<f64>(QA * QB);
}

} // namespace network::value
