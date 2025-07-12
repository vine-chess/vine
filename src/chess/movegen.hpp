#ifndef MOVEGEN_HPP
#define MOVEGEN_HPP

#include "../util/static_vector.hpp"
#include "bitboard.hpp"
#include "board_state.hpp"
#include <array>

using MoveList = util::StaticVector<Move, 218>;

constexpr static auto KNIGHT_MOVES = []() {
    std::array<Bitboard, 64> res;
    for (int i = 0; i < 64; ++i) {
        const auto sq = Bitboard(1ull << i);
        const auto forward_back = sq.shift_masked<2, 0>() | sq.shift_masked<-2, 0>();
        const auto left_right = sq.shift_masked<0, 2>() | sq.shift_masked<0, -2>();
        res[i] = forward_back.shift_masked<0, 1>() | forward_back.shift_masked<0, -1>() |
                 left_right.shift_masked<1, 0>() | left_right.shift_masked<-1, 0>();
    }
    return res;
}();

constexpr static auto KING_MOVES = []() {
    std::array<Bitboard, 64> res;
    for (int i = 0; i < 64; ++i) {
        res[i] = 1ull << i;
        res[i] |= res[i].shift_masked<1, 0>();
        res[i] |= res[i].shift_masked<-1, 0>();
        res[i] |= res[i].shift_masked<0, 1>();
        res[i] |= res[i].shift_masked<0, -1>();
        res[i] ^= 1ull << i;
    }
    return res;
}();

inline void pawn_moves(const BoardState &board, MoveList &move_list,
                       Bitboard allowed_destinations = Bitboard::ALL_SET) {
    const auto forward = board.side_to_move == Color::WHITE ? 1 : -1;

    const auto occ = board.occupancy();

    const auto pawns = board.pawns(board.side_to_move);

    const Bitboard allowed_double_push_rank = 0xffull << 8 * (board.side_to_move == Color::WHITE ? 3 : 4);
    const Bitboard promo_ranks = 0xff'00'00'00'00'00'00'ff;

    const auto one_forward = pawns << 8 * forward & allowed_destinations & ~occ;
    const auto two_forward = one_forward << 8 * forward & allowed_destinations & ~occ & allowed_double_push_rank;
    const auto left_captures = one_forward.shift_masked<0, -1>() & board.occupancy(~board.side_to_move);
    const auto right_captures = one_forward.shift_masked<0, 1>() & board.occupancy(~board.side_to_move);

    for (auto sq : one_forward & ~promo_ranks) {
        move_list.emplace_back(sq - forward * 8, sq);
    }
    for (auto sq : two_forward) {
        move_list.emplace_back(sq - forward * 16, sq);
    }
    for (auto sq : one_forward &promo_ranks) {
        move_list.emplace_back(sq - forward * 8, sq, MoveFlag::PROMO_KNIGHT);
        move_list.emplace_back(sq - forward * 8, sq, MoveFlag::PROMO_BISHOP);
        move_list.emplace_back(sq - forward * 8, sq, MoveFlag::PROMO_ROOK);
        move_list.emplace_back(sq - forward * 8, sq, MoveFlag::PROMO_QUEEN);
    }
    for (auto sq : left_captures & ~promo_ranks) {
        move_list.emplace_back(sq - forward * 8 + 1, sq, MoveFlag::CAPTURE_BIT);
    }
    for (auto sq : right_captures & ~promo_ranks) {
        move_list.emplace_back(sq - forward * 8 - 1, sq, MoveFlag::CAPTURE_BIT);
    }
    for (auto sq : left_captures &promo_ranks) {
        move_list.emplace_back(sq - forward * 8 + 1, sq, MoveFlag::PROMO_KNIGHT_CAPTURE);
        move_list.emplace_back(sq - forward * 8 + 1, sq, MoveFlag::PROMO_BISHOP_CAPTURE);
        move_list.emplace_back(sq - forward * 8 + 1, sq, MoveFlag::PROMO_ROOK_CAPTURE);
        move_list.emplace_back(sq - forward * 8 + 1, sq, MoveFlag::PROMO_QUEEN_CAPTURE);
    }
    for (auto sq : right_captures &promo_ranks) {
        move_list.emplace_back(sq - forward * 8 - 1, sq, MoveFlag::PROMO_KNIGHT_CAPTURE);
        move_list.emplace_back(sq - forward * 8 - 1, sq, MoveFlag::PROMO_BISHOP_CAPTURE);
        move_list.emplace_back(sq - forward * 8 - 1, sq, MoveFlag::PROMO_ROOK_CAPTURE);
        move_list.emplace_back(sq - forward * 8 - 1, sq, MoveFlag::PROMO_QUEEN_CAPTURE);
    }
}

inline void knight_moves(const BoardState &board, MoveList &move_list,
                         Bitboard allowed_destionations = Bitboard::ALL_SET) {
    for (auto knight : board.knights(board.side_to_move)) {
        for (auto destination : KNIGHT_MOVES[knight] & allowed_destionations & ~board.occupancy(board.side_to_move)) {
            move_list.emplace_back(knight, destination,
                                   board.occupancy(~board.side_to_move).is_set(destination) ? MoveFlag::CAPTURE_BIT
                                                                                            : MoveFlag::NORMAL);
        }
    }
}

#endif // MOVEGEN_HPP
