#include "policy_network.hpp"
#include "../chess/move_gen.hpp"

#include "../third_party/incbin.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>

namespace network::policy {

const extern PolicyNetwork *const network;

namespace detail {

[[nodiscard]] const util::MultiArray<i8Vec, L1_0_SIZE / VECTOR_SIZE> &feature(Square sq, PieceType piece,
                                                                              Color piece_color, Color perspective,
                                                                              Bitboard threats, Bitboard defences,
                                                                              Square king_sq) {
    const usize flip = (0b111000 * perspective) ^ (king_sq.file() >= File::E ? 0b000111 : 0);
    return network
        ->ft_weights_vec[defences.is_set(sq)][threats.is_set(sq)][piece_color != perspective][piece - 1][sq ^ flip];
}

constexpr static std::array<std::array<Bitboard, 6>, 64> DESTINATIONS = [] {
    std::array<std::array<Bitboard, 6>, 64> arr{};
    for (i32 sq = 0; sq < 64; ++sq) {
        const auto sq_bb = Bitboard(Square(sq));
        arr[sq][0] = sq_bb.shift<UP, 0>() | sq_bb.shift<UP, LEFT>() | sq_bb.shift<UP, RIGHT>();
        arr[sq][1] = KNIGHT_MOVES[sq];
        arr[sq][2] = BISHOP_RAYS[sq];
        arr[sq][3] = ROOK_RAYS[sq];
        arr[sq][4] = QUEEN_RAYS[sq];
        arr[sq][5] = KING_MOVES[sq];
    }
    return arr;
}();

constexpr std::array<std::array<usize, 65>, 6> OFFSETS = [] {
    std::array<std::array<usize, 65>, 6> off{};
    usize cur = 0;
    for (i32 p = 0; p < 6; ++p) {
        for (i32 sq = 0; sq < 64; ++sq) {
            off[p][sq] = cur;
            cur += static_cast<usize>(DESTINATIONS[sq][p].pop_count());
        }
        off[p][64] = cur; // Start of the promotion buckets
    }
    return off;
}();

[[nodiscard]] i32 promo_kind_id(PieceType pt) {
    return static_cast<i32>(pt) - 2;
}

[[nodiscard]] usize move_output_idx(Color stm, Move move, PieceType moving_piece, Square king_sq) {
    const usize flipper = (stm == Color::BLACK ? 0b111000 : 0) ^ (king_sq.file() >= File::E ? 0b000111 : 0);
    const Square from = move.from() ^ flipper;
    const Square to = move.to() ^ flipper;
    constexpr usize PROMO_STRIDE = 22;
    if (move.is_promo()) {
        const i32 promo_id = 2 * from.file() + to.file();
        const i32 kind = promo_kind_id(move.promo_type());
        return OFFSETS[5][64] + static_cast<usize>(kind * PROMO_STRIDE + promo_id);
    } else if (move.is_castling()) {
        const bool is_kingside = move.from() < move.to();
        const bool is_hm = king_sq.file() >= File::E;
        return OFFSETS[5][64] + PROMO_STRIDE * 4 + (is_kingside ^ is_hm);
    } else if (moving_piece == PieceType::PAWN && (move.from() ^ move.to()) == 16) {
        return OFFSETS[5][64] + PROMO_STRIDE * 4 + 2 + from.file();
    } else {
        const u64 all = static_cast<u64>(DESTINATIONS[from][moving_piece - 1]);
        const u64 below = all & (to.to_bb() - 1);
        return OFFSETS[moving_piece - 1][from] + static_cast<usize>(std::popcount(below));
    }
}

} // namespace detail

PolicyContext::PolicyContext(const BoardState &state)
    : stm_(state.side_to_move), king_sq_(state.king(state.side_to_move).lsb()) {
    std::array<i16Vec, L1_0_SIZE / VECTOR_SIZE> feature_accumulator_;
    for (usize i = 0; i < L1_0_SIZE / VECTOR_SIZE; ++i) {
        feature_accumulator_[i] = util::convert_vector<i16, i8, VECTOR_SIZE>(network->ft_biases_vec[i]);
    }

    // Accumulate features for both sides, viewed from side-to-move's perspective
    for (PieceType piece = PieceType::PAWN; piece <= PieceType::KING; piece = PieceType(piece + 1)) {
        // Our pieces
        const std::array<Bitboard, 2> threats = {state.threats_by(Color::WHITE), state.threats_by(Color::BLACK)};
        for (auto sq : state.piece_bbs[piece - 1] & state.occupancy(stm_)) {
            for (usize i = 0; i < L1_0_SIZE / VECTOR_SIZE; ++i) {
                feature_accumulator_[i] += util::convert_vector<i16, i8, VECTOR_SIZE>(
                    detail::feature(sq, piece, stm_, stm_, threats[~stm_], threats[stm_], king_sq_)[i]);
            }
        }
        // Opponent pieces
        for (auto sq : state.piece_bbs[piece - 1] & state.occupancy(~stm_)) {
            for (usize i = 0; i < L1_0_SIZE / VECTOR_SIZE; ++i) {
                feature_accumulator_[i] += util::convert_vector<i16, i8, VECTOR_SIZE>(
                    detail::feature(sq, piece, ~stm_, stm_, threats[~stm_], threats[stm_], king_sq_)[i]);
            }
        }
    }
    for (usize i = 0; i < L1_0_SIZE / 2 / VECTOR_SIZE; ++i) {
        const auto first_clamped = util::clamp_scalar<i16, VECTOR_SIZE>(feature_accumulator_[i], 0, Q);
        const auto second_clamped =
            util::clamp_scalar<i16, VECTOR_SIZE>(feature_accumulator_[i + L1_0_SIZE / 2 / VECTOR_SIZE], 0, Q);
        activated_acc_[i] = first_clamped * second_clamped;
    }
}

f32 PolicyContext::logit(Move move, PieceType moving_piece) const {
    const usize idx = detail::move_output_idx(stm_, move, moving_piece, king_sq_);

    // HL -> Output
    util::SimdVector<i32, VECTOR_SIZE / 2> hl_sum{};
    for (usize i = 0; i < L1_0_SIZE / 2 / VECTOR_SIZE; ++i) {
        hl_sum += util::madd_epi16(activated_acc_[i],
                                   util::convert_vector<i16, i8, VECTOR_SIZE>(network->l1_0_weights_vec[idx][i]));
    }

    const i32 hl_dot = util::reduce_vector<i32, VECTOR_SIZE / 2>(hl_sum);
    const i32 hl_bias = network->l1_0_biases[idx];
    const f32 hl_out =
        ((static_cast<f32>(hl_dot) / static_cast<f32>(Q * Q)) + static_cast<f32>(hl_bias)) / static_cast<f32>(Q);


    const f32Vec dequantize_vector = util::set1<f32, VECTOR_SIZE>(1.0 / (Q * Q));
    // HL -> Sub HL
    std::array<util::SimdVector<f32, VECTOR_SIZE>, L1_1_SIZE / VECTOR_SIZE> sub_hl_sum{};
    for (usize i = 0; i < L1_1_SIZE / VECTOR_SIZE; ++i) {
        for (usize j = 0; j < L1_0_SIZE / 2 / VECTOR_SIZE; ++j) {
            sub_hl_sum[i] +=
                util::convert_vector<f32, i16, VECTOR_SIZE>(activated_acc_[j]) * dequantize_vector * network->l1_1_weights_vec[i][j];
        }
        sub_hl_sum[i] += network->l1_1_biases_vec[i];
        sub_hl_sum[i] = util::clamp_scalar<f32, VECTOR_SIZE>(sub_hl_sum[i], 0.f, 1.f);
    }

    // Sub HL -> Output
    util::SimdVector<f32, VECTOR_SIZE> sub_hl_output{};
    for (usize i = 0; i < L1_1_SIZE / VECTOR_SIZE; ++i) {
        sub_hl_output += sub_hl_sum[i] * network->l2_weights_vec[idx][i];
    }

    const f32 sub_hl_dot = util::reduce_vector<f32, VECTOR_SIZE>(sub_hl_output);
    const f32 sub_hl_bias = network->l2_biases[idx];
    const f32 sub_hl_out = sub_hl_dot + sub_hl_bias;

    return hl_out + sub_hl_out;
}

} // namespace network::policy
