#include "move_gen.hpp"
#include "bitboard.hpp"
#include "magics.hpp"
#include <ios>
#include <iostream>
constexpr static auto KNIGHT_MOVES = []() {
    std::array<Bitboard, 64> res;
    for (int i = 0; i < 64; ++i) {
        const auto sq = Bitboard(Square(i));
        const auto forward_back = sq.shift<UP * 2, 0>() | sq.shift<DOWN * 2, 0>();
        const auto left_right = sq.shift<0, LEFT * 2>() | sq.shift<0, RIGHT * 2>();
        res[i] = forward_back.shift<0, LEFT>() | forward_back.shift<0, RIGHT>() | left_right.shift<UP, 0>() |
                 left_right.shift<DOWN, 0>();
    }
    return res;
}();

constexpr static auto KING_MOVES = []() {
    std::array<Bitboard, 64> res;
    for (int i = 0; i < 64; ++i) {
        res[i] = Bitboard(Square(i));
        res[i] |= res[i].shift<UP, 0>();
        res[i] |= res[i].shift<DOWN, 0>();
        res[i] |= res[i].shift<0, RIGHT>();
        res[i] |= res[i].shift<0, LEFT>();
        res[i] ^= Bitboard(Square(i));
    }
    return res;
}();

void pawn_moves(const BoardState &board, MoveList &move_list, Bitboard allowed_destinations) {
    const auto forward = board.side_to_move == Color::WHITE ? 1 : -1;

    const auto occ = board.occupancy();
    const auto pawns = board.pawns(board.side_to_move);

    const Bitboard allowed_double_push_rank =
        Bitboard::rank_mask(board.side_to_move == Color::WHITE ? Rank::FOURTH : Rank::FIFTH);
    const Bitboard promo_ranks = Bitboard::rank_mask(Rank::FIRST) | Bitboard::rank_mask(Rank::EIGHTH);

    const auto horizontal_pins = board.ortho_pins & (board.ortho_pins << 1);
    const auto vertical_pins = board.ortho_pins & board.ortho_pins.shift<UP, 0>();
    const auto one_forward = pawns.rotl(8 * forward);
    const auto pushable = one_forward & ~(board.diag_pins | horizontal_pins).rotl(8 * forward) & ~occ;
    const auto two_forward = pushable.rotl(8 * forward) & ~occ & allowed_double_push_rank;
    const auto left_captures = one_forward.shift<0, LEFT>() & board.occupancy(~board.side_to_move) & ~board.ortho_pins;
    const auto right_captures =
        one_forward.shift<0, RIGHT>() & board.occupancy(~board.side_to_move) & ~board.ortho_pins;

    for (auto sq : pushable & ~promo_ranks & allowed_destinations) {
        move_list.emplace_back(sq - forward * 8, sq);
    }

    for (auto sq : (two_forward & allowed_destinations)) {
        move_list.emplace_back(sq - forward * 16, sq);
    }

    for (auto sq : (pushable & promo_ranks & allowed_destinations)) {
        move_list.emplace_back(sq - forward * 8, sq, MoveFlag::PROMO_KNIGHT);
        move_list.emplace_back(sq - forward * 8, sq, MoveFlag::PROMO_BISHOP);
        move_list.emplace_back(sq - forward * 8, sq, MoveFlag::PROMO_ROOK);
        move_list.emplace_back(sq - forward * 8, sq, MoveFlag::PROMO_QUEEN);
    }

    for (auto sq : (left_captures & ~promo_ranks & allowed_destinations)) {
        move_list.emplace_back(sq - forward * 8 + 1, sq, MoveFlag::CAPTURE_BIT);
    }

    for (auto sq : (right_captures & ~promo_ranks & allowed_destinations)) {
        move_list.emplace_back(sq - forward * 8 - 1, sq, MoveFlag::CAPTURE_BIT);
    }

    for (auto sq : (left_captures & promo_ranks & allowed_destinations)) {
        move_list.emplace_back(sq - forward * 8 + 1, sq, MoveFlag::PROMO_KNIGHT_CAPTURE);
        move_list.emplace_back(sq - forward * 8 + 1, sq, MoveFlag::PROMO_BISHOP_CAPTURE);
        move_list.emplace_back(sq - forward * 8 + 1, sq, MoveFlag::PROMO_ROOK_CAPTURE);
        move_list.emplace_back(sq - forward * 8 + 1, sq, MoveFlag::PROMO_QUEEN_CAPTURE);
    }

    for (auto sq : (right_captures & promo_ranks & allowed_destinations)) {
        move_list.emplace_back(sq - forward * 8 - 1, sq, MoveFlag::PROMO_KNIGHT_CAPTURE);
        move_list.emplace_back(sq - forward * 8 - 1, sq, MoveFlag::PROMO_BISHOP_CAPTURE);
        move_list.emplace_back(sq - forward * 8 - 1, sq, MoveFlag::PROMO_ROOK_CAPTURE);
        move_list.emplace_back(sq - forward * 8 - 1, sq, MoveFlag::PROMO_QUEEN_CAPTURE);
    }
}

void knight_moves(const BoardState &board, MoveList &move_list, Bitboard allowed_destinations) {
    const auto occ = board.occupancy();
    const auto us = board.occupancy(board.side_to_move);
    const auto them = board.occupancy(~board.side_to_move);

    for (auto from : board.knights(board.side_to_move)) {
        const auto legal = KNIGHT_MOVES[from] & allowed_destinations;
        for (auto to : legal & ~us) {
            move_list.emplace_back(from, to, them.is_set(to) ? MoveFlag::CAPTURE_BIT : MoveFlag::NORMAL);
        }
    }
}

void slider_moves(const BoardState &board, MoveList &move_list, Bitboard allowed_destinations) {
    const auto occ = board.occupancy();
    const auto us = board.occupancy(board.side_to_move);
    const auto them = board.occupancy(~board.side_to_move);

    for (auto from : board.queens(board.side_to_move) | board.bishops(board.side_to_move)) {
        const auto legal = get_bishop_attacks(from, occ) & allowed_destinations;
        for (auto to : legal & ~us) {
            move_list.emplace_back(from, to, them.is_set(to) ? MoveFlag::CAPTURE_BIT : MoveFlag::NORMAL);
        }
    }

    for (auto from : board.queens(board.side_to_move) | board.rooks(board.side_to_move)) {
        const auto legal = get_rook_attacks(from, occ) & allowed_destinations;
        for (auto to : legal & ~us) {
            move_list.emplace_back(from, to, them.is_set(to) ? MoveFlag::CAPTURE_BIT : MoveFlag::NORMAL);
        }
    }
}

void king_moves(const BoardState &board, MoveList &move_list, Bitboard allowed_destinations) {
    constexpr static auto KING_SUPERPIECE = []() {
        std::array<std::array<Bitboard, 3>, 64> res;
        for (int i = 0; i < 64; ++i) {
            for (int j = 0; j < 3; ++j) {
                res[i][j] = KING_MOVES[i];
            }
            for (auto from : res[i][0]) {
                res[i][0] |= KNIGHT_MOVES[from];
            }
            for (auto from : res[i][1]) {
                res[i][1] |= BISHOP_RAYS[from];
            }
            for (auto from : res[i][2]) {
                res[i][2] |= ROOK_RAYS[from];
            }
        }
        return res;
    }();
    const auto occ = board.occupancy();
    const auto us = board.occupancy(board.side_to_move);
    const auto them = board.occupancy(~board.side_to_move);
    const auto king = board.king(board.side_to_move);
    const auto king_sq = king.lsb();

    for (auto knight : KING_SUPERPIECE[king_sq][0] & board.knights(~board.side_to_move)) {
        allowed_destinations &= ~KNIGHT_MOVES[knight];
    }
    // std::cout << (u64)KING_SUPERPIECE[king_sq][1] << '\n';
    for (auto bishop : KING_SUPERPIECE[king_sq][1] & them & (board.bishops() | board.queens())) {
        // std::cout << (u64)get_bishop_attacks(bishop, occ ^ king) << '\n';
        allowed_destinations &= ~get_bishop_attacks(bishop, occ ^ king);
    }
    for (auto rook : KING_SUPERPIECE[king_sq][2] & them & (board.rooks() | board.queens())) {
        allowed_destinations &= ~get_rook_attacks(rook, occ ^ king);
    }
    const auto legal = KING_MOVES[king_sq] & allowed_destinations;

    for (auto to : legal & ~us) {
        move_list.emplace_back(king_sq, to, them.is_set(to) ? MoveFlag::CAPTURE_BIT : MoveFlag::NORMAL);
    }
}

void generate_moves(const BoardState &board, MoveList &move_list) {
    Bitboard allowed = Bitboard::ALL_SET;
    if (board.checkers != 0) {
        if (board.checkers.pop_count() > 1) {
            king_moves(board, move_list);
            return;
        }
        const auto checker = board.checkers;
        const auto king = board.king(board.side_to_move);
        allowed = RAY_BETWEEN[checker.lsb()][king.lsb()] | checker;
    }
    pawn_moves(board, move_list, allowed);
    knight_moves(board, move_list, allowed);
    slider_moves(board, move_list, allowed);
    king_moves(board, move_list, allowed);
}
