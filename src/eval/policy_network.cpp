#include "policy_network.hpp"

#include "../third_party/incbin.h"
#include <array>
#include <cstring>

INCBIN(POLICYNETWORK, POLICYFILE);

namespace network::policy {

const auto network = reinterpret_cast<const PolicyNetwork *>(gPOLICYNETWORKData);

const util::MultiArray<i8, L1_SIZE> &feature(Square sq, PieceType piece, Color piece_color, Color perspective) {
    usize flip = 0b111000 * perspective;
    return network->ft_weights[piece_color != perspective][piece - 1][sq ^ flip];
}

constexpr std::array<Bitboard, 64> ALL_DESTINATIONS = [] {
    std::array<Bitboard, 64> arr{};
    for (i32 sq = 0; sq < 64; ++sq) {
        arr[sq] = ROOK_RAYS[sq] | BISHOP_RAYS[sq] | KNIGHT_MOVES[sq] | KING_MOVES[sq];
    }
    return arr;
}();

constexpr std::array<usize, 65> OFFSETS = [] {
    std::array<usize, 65> off{};
    usize cur = 0;
    for (i32 sq = 0; sq < 64; ++sq) {
        off[sq] = cur;
        cur += static_cast<usize>(ALL_DESTINATIONS[sq].pop_count());
    }
    off[64] = cur; // Start of the promotion buckets
    return off;
}();

[[nodiscard]] i32 promo_kind_id(PieceType pt) {
    return static_cast<i32>(pt) - 2;
}

[[nodiscard]] usize move_output_idx(Color stm, Move move) {
    const i32 flipper = stm == Color::BLACK ? 56 : 0;
    if (move.is_promo()) {
        constexpr usize PROMO_STRIDE = 22;
        const i32 promo_id = 2 * move.from().file() + move.to().file();
        const i32 kind = promo_kind_id(move.promo_type());
        return OFFSETS[64] + static_cast<usize>(kind * PROMO_STRIDE + promo_id);
    } else {
        const Square from = move.from() ^ flipper;
        const Square to = move.to() ^ flipper;
        const u64 all = static_cast<u64>(ALL_DESTINATIONS[from]);
        const u64 below = to == 0 ? 0ull : (all & (to.to_bb() - 1ull));
        return OFFSETS[from] + static_cast<usize>(std::popcount(below));
    }
}

std::array<i16, L1_SIZE> accumulate_policy(const BoardState &state) {
    std::array<i16, L1_SIZE> accumulator{};
    for (usize i = 0; i < L1_SIZE; ++i) {
        accumulator[i] += network->ft_biases[i];
    }

    const auto stm = state.side_to_move;
    // add all the features
    for (PieceType piece = PieceType::PAWN; piece <= PieceType::KING; piece = PieceType(piece + 1)) {
        for (auto sq : state.piece_bbs[piece - 1] & state.occupancy(stm)) {
            for (usize i = 0; i < L1_SIZE; ++i) {
                accumulator[i] += feature(sq, piece, stm, stm)[i];
            }
        }
        for (auto sq : state.piece_bbs[piece - 1] & state.occupancy(~stm)) {
            for (usize i = 0; i < L1_SIZE; ++i) {
                accumulator[i] += feature(sq, piece, ~stm, stm)[i];
            }
        }
    }

    return accumulator;
}

f64 evaluate_move(const std::array<i16, L1_SIZE> &accumulator, const BoardState &state, Move move) {
    const usize idx = move_output_idx(state.side_to_move, move);
    i32 output = network->l1_biases[idx];
    for (usize i = 0; i < L1_SIZE; i++) {
        const i32 clamped = std::clamp<i32>(accumulator[i], 0, Q);
        output += clamped * network->l1_weights[idx][i];
    }

    return static_cast<f64>(output) / static_cast<f64>(Q * Q);
}

} // namespace network
