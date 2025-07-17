#include "move_gen.hpp"
#include "bitboard.hpp"
#include "magics.hpp"
#include <ios>
#include <iostream>

Bitboard compute_pawn_attacks(Bitboard pawns, Color side_to_move) {
    Bitboard forward = pawns.rotl(side_to_move == Color::WHITE ? 8 : -8);
    return forward.shift<0, LEFT>() | forward.shift<0, RIGHT>();
}

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
    const auto left_captures =
        one_forward.shift<0, LEFT>() & board.occupancy(~board.side_to_move) & ~(board.ortho_pins.shift<0, RIGHT>());
    const auto right_captures =
        one_forward.shift<0, RIGHT>() & board.occupancy(~board.side_to_move) & ~(board.ortho_pins.shift<0, LEFT>());

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

    for (auto from : board.knights(board.side_to_move) & ~(board.ortho_pins | board.diag_pins)) {
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

    for (auto from : (board.queens(board.side_to_move) | board.bishops(board.side_to_move)) & ~board.ortho_pins) {
        const auto legal = get_bishop_attacks(from, occ) & allowed_destinations;
        for (auto to : legal & ~us) {
            const auto from_pinned = board.diag_pins.is_set(from);
            const auto to_pinned = board.diag_pins.is_set(to);
            move_list.push_back_conditional(Move(from, to, them.is_set(to) ? MoveFlag::CAPTURE_BIT : MoveFlag::NORMAL),
                                            !from_pinned || to_pinned);
        }
    }

    for (auto from : (board.queens(board.side_to_move) | board.rooks(board.side_to_move)) & ~board.diag_pins) {
        const auto legal = get_rook_attacks(from, occ) & allowed_destinations;
        for (auto to : legal & ~us) {
            const auto from_pinned = board.ortho_pins.is_set(from);
            const auto to_pinned = board.ortho_pins.is_set(to);
            move_list.push_back_conditional(Move(from, to, them.is_set(to) ? MoveFlag::CAPTURE_BIT : MoveFlag::NORMAL),
                                            !from_pinned || to_pinned);
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
            if (i < 8) {
                for (int j = 0; j < 8; ++j) {
                    if (i <= 2 and 2 <= j) {
                        res[i][0] |= KNIGHT_MOVES[j];
                        res[i][1] |= BISHOP_RAYS[j];
                        res[i][2] |= ROOK_RAYS[j];
                    }
                    if (i <= 6 and 6 <= j) {
                        res[i][0] |= KNIGHT_MOVES[j];
                        res[i][1] |= BISHOP_RAYS[j];
                        res[i][2] |= ROOK_RAYS[j];
                    }
                }
            }
            if (i >= 56) {
                for (int j = 56; j < 64; ++j) {
                    if (i <= 58 and 58 <= j) {
                        res[i][0] |= KNIGHT_MOVES[j];
                        res[i][1] |= BISHOP_RAYS[j];
                        res[i][2] |= ROOK_RAYS[j];
                    }
                    if (i <= 62 and 62 <= j) {
                        res[i][0] |= KNIGHT_MOVES[j];
                        res[i][1] |= BISHOP_RAYS[j];
                        res[i][2] |= ROOK_RAYS[j];
                    }
                }
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
        allowed_destinations &= ~KNIGHT_MOVES[knight] | knight.to_bb();
    }
    // std::cout << (u64)KING_SUPERPIECE[king_sq][1] << '\n';
    for (auto bishop : KING_SUPERPIECE[king_sq][1] & them & (board.bishops() | board.queens())) {
        // std::cout << (u64)get_bishop_attacks(bishop, occ ^ king) << '\n';
        allowed_destinations &= ~get_bishop_attacks(bishop, occ ^ king);
    }
    for (auto rook : KING_SUPERPIECE[king_sq][2] & them & (board.rooks() | board.queens())) {
        allowed_destinations &= ~get_rook_attacks(rook, occ ^ king);
    }
    allowed_destinations &= ~KING_MOVES[board.king(~board.side_to_move).lsb()];
    allowed_destinations &= ~compute_pawn_attacks(board.pawns(~board.side_to_move), ~board.side_to_move);

    const auto legal = KING_MOVES[king_sq] & allowed_destinations;

    for (auto to : legal & ~us) {
        move_list.emplace_back(king_sq, to, them.is_set(to) ? MoveFlag::CAPTURE_BIT : MoveFlag::NORMAL);
    }

    if (board.checkers == 0) {
        // TODO: fix for FRC
        constexpr std::array<Bitboard, 2> KINGSIDE_OCCUPANCY_SQUARES = {0x60ull, 0x6000000000000000ull};
        constexpr std::array<Bitboard, 2> QUEENSIDE_OCCUPANCY_SQUARES = {0xeull, 0xe00000000000000ull};
        constexpr std::array<Bitboard, 2> KINGSIDE_ATTACKED_SQUARES = {0x60ull, 0x6000000000000000ull};
        constexpr std::array<Bitboard, 2> QUEENSIDE_ATTACKED_SQUARES = {0xcull, 0xc00000000000000ull};

        if (board.castle_rights.can_kingside_castle(board.side_to_move) &&
            (KINGSIDE_OCCUPANCY_SQUARES[board.side_to_move] & occ) == 0 &&
            (allowed_destinations & KINGSIDE_ATTACKED_SQUARES[board.side_to_move]) ==
                KINGSIDE_ATTACKED_SQUARES[board.side_to_move]) {
            move_list.emplace_back(king_sq, board.castle_rights.kingside_rook_sq(board.side_to_move), MoveFlag::CASTLE);
        }

        if (board.castle_rights.can_queenside_castle(board.side_to_move) &&
            (QUEENSIDE_OCCUPANCY_SQUARES[board.side_to_move] & occ) == 0 &&
            (allowed_destinations & QUEENSIDE_ATTACKED_SQUARES[board.side_to_move]) ==
                QUEENSIDE_ATTACKED_SQUARES[board.side_to_move]) {
            move_list.emplace_back(king_sq, board.castle_rights.queenside_rook_sq(board.side_to_move),
                                   MoveFlag::CASTLE);
        }
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
        // std::cout << (u64)allowed << '\n';
    }
    pawn_moves(board, move_list, allowed);
    knight_moves(board, move_list, allowed);
    slider_moves(board, move_list, allowed);
    king_moves(board, move_list);
}
