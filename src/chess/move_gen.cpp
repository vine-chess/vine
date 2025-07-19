#include "move_gen.hpp"
#include "bitboard.hpp"
#include "magics.hpp"
#include <ios>
#include <iostream>

Bitboard compute_pawn_attacks(Bitboard pawns, Color side_to_move) {
    Bitboard forward = pawns.rotl(side_to_move == Color::WHITE ? 8 : -8);
    return forward.shift<0, LEFT>() | forward.shift<0, RIGHT>();
}

void pawn_moves(const BoardState &state, MoveList &move_list, Bitboard allowed_destinations) {
    const auto forward = state.side_to_move == Color::WHITE ? 1 : -1;

    const auto occ = state.occupancy();
    const auto pawns = state.pawns(state.side_to_move);

    const Bitboard allowed_double_push_rank =
        Bitboard::rank_mask(state.side_to_move == Color::WHITE ? Rank::FOURTH : Rank::FIFTH);
    const Bitboard promo_ranks = Bitboard::rank_mask(Rank::FIRST) | Bitboard::rank_mask(Rank::EIGHTH);

    const auto horizontal_pins = state.ortho_pins & (state.ortho_pins << 1);
    const auto vertical_pins = state.ortho_pins & state.ortho_pins.shift<UP, 0>();
    const auto one_forward = pawns.rotl(8 * forward);
    const auto pushable = one_forward & ~(state.diag_pins | horizontal_pins).rotl(8 * forward) & ~occ;
    const auto two_forward = pushable.rotl(8 * forward) & ~occ & allowed_double_push_rank;
    const auto left_captures =
        one_forward.shift<0, LEFT>() & state.occupancy(~state.side_to_move) & ~(state.ortho_pins.shift<0, RIGHT>());
    const auto right_captures =
        one_forward.shift<0, RIGHT>() & state.occupancy(~state.side_to_move) & ~(state.ortho_pins.shift<0, LEFT>());

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

    const auto ep_target_bb = Bitboard(state.en_passant_sq) & ~vertical_pins;
    auto ep_targeting_pawns = (ep_target_bb.shift<0, LEFT>() | ep_target_bb.shift<0, RIGHT>()).rotl(-forward * 8);

    if (state.diag_pins.is_set(state.en_passant_sq)) {
        ep_targeting_pawns &= state.diag_pins;
    }
}

void knight_moves(const BoardState &state, MoveList &move_list, Bitboard allowed_destinations) {
    const auto occ = state.occupancy();
    const auto us = state.occupancy(state.side_to_move);
    const auto them = state.occupancy(~state.side_to_move);

    for (auto from : state.knights(state.side_to_move) & ~(state.ortho_pins | state.diag_pins)) {
        const auto legal = KNIGHT_MOVES[from] & allowed_destinations;
        for (auto to : legal & ~us) {
            move_list.emplace_back(from, to, them.is_set(to) ? MoveFlag::CAPTURE_BIT : MoveFlag::NORMAL);
        }
    }
}

void slider_moves(const BoardState &state, MoveList &move_list, Bitboard allowed_destinations) {
    const auto occ = state.occupancy();
    const auto us = state.occupancy(state.side_to_move);
    const auto them = state.occupancy(~state.side_to_move);

    for (auto from : (state.queens(state.side_to_move) | state.bishops(state.side_to_move)) & ~state.ortho_pins) {
        const auto legal = get_bishop_attacks(from, occ) & allowed_destinations;
        for (auto to : legal & ~us) {
            const auto from_pinned = state.diag_pins.is_set(from);
            const auto to_pinned = state.diag_pins.is_set(to);
            move_list.push_back_conditional(Move(from, to, them.is_set(to) ? MoveFlag::CAPTURE_BIT : MoveFlag::NORMAL),
                                            !from_pinned || to_pinned);
        }
    }

    for (auto from : (state.queens(state.side_to_move) | state.rooks(state.side_to_move)) & ~state.diag_pins) {
        const auto legal = get_rook_attacks(from, occ) & allowed_destinations;
        for (auto to : legal & ~us) {
            const auto from_pinned = state.ortho_pins.is_set(from);
            const auto to_pinned = state.ortho_pins.is_set(to);
            move_list.push_back_conditional(Move(from, to, them.is_set(to) ? MoveFlag::CAPTURE_BIT : MoveFlag::NORMAL),
                                            !from_pinned || to_pinned);
        }
    }
}

void king_moves(const BoardState &state, MoveList &move_list, Bitboard allowed_destinations) {
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
                const auto lo1 = i < 2 ? i : 2;
                const auto hi1 = i < 2 ? 2 : i;
                const auto lo2 = i < 6 ? i : 6;
                const auto hi2 = i < 6 ? 6 : i;
                for (int j = 0; j < 8; ++j) {
                    if (lo1 <= j && j <= hi1) {
                        res[i][0] |= KNIGHT_MOVES[j];
                        res[i][1] |= BISHOP_RAYS[j];
                        res[i][2] |= ROOK_RAYS[j];
                    }
                    if (lo2 <= j && j <= hi2) {
                        res[i][0] |= KNIGHT_MOVES[j];
                        res[i][1] |= BISHOP_RAYS[j];
                        res[i][2] |= ROOK_RAYS[j];
                    }
                }
            }
            if (i >= 56) {
                const auto lo1 = i < 58 ? i : 58;
                const auto hi1 = i < 58 ? 58 : i;
                const auto lo2 = i < 62 ? i : 62;
                const auto hi2 = i < 62 ? 62 : i;
                for (int j = 56; j < 64; ++j) {
                    if (lo1 <= j && j <= hi1) {
                        res[i][0] |= KNIGHT_MOVES[j];
                        res[i][1] |= BISHOP_RAYS[j];
                        res[i][2] |= ROOK_RAYS[j];
                    }
                    if (lo2 <= j && j <= hi2) {
                        res[i][0] |= KNIGHT_MOVES[j];
                        res[i][1] |= BISHOP_RAYS[j];
                        res[i][2] |= ROOK_RAYS[j];
                    }
                }
            }
        }
        return res;
    }();
    const auto occ = state.occupancy();
    const auto us = state.occupancy(state.side_to_move);
    const auto them = state.occupancy(~state.side_to_move);
    const auto king = state.king(state.side_to_move);
    const auto king_sq = king.lsb();

    for (auto knight : KING_SUPERPIECE[king_sq][0] & state.knights(~state.side_to_move)) {
        allowed_destinations &= ~KNIGHT_MOVES[knight] | knight.to_bb();
    }
    // std::cout << (u64)KING_SUPERPIECE[king_sq][1] << '\n';
    for (auto bishop : KING_SUPERPIECE[king_sq][1] & them & (state.bishops() | state.queens())) {
        // std::cout << (u64)get_bishop_attacks(bishop, occ ^ king) << '\n';
        allowed_destinations &= ~get_bishop_attacks(bishop, occ ^ king);
    }
    for (auto rook : KING_SUPERPIECE[king_sq][2] & them & (state.rooks() | state.queens())) {
        allowed_destinations &= ~get_rook_attacks(rook, occ ^ king);
    }
    allowed_destinations &= ~KING_MOVES[state.king(~state.side_to_move).lsb()];
    allowed_destinations &= ~compute_pawn_attacks(state.pawns(~state.side_to_move), ~state.side_to_move);

    const auto legal = KING_MOVES[king_sq] & allowed_destinations;

    for (auto to : legal & ~us) {
        move_list.emplace_back(king_sq, to, them.is_set(to) ? MoveFlag::CAPTURE_BIT : MoveFlag::NORMAL);
    }

    if (state.checkers == 0) {
        const auto kingside_rook = state.castle_rights.kingside_rook(state.side_to_move);
        const auto queenside_rook = state.castle_rights.queenside_rook(state.side_to_move);
        const auto kingside_king = state.castle_rights.kingside_king_dest(state.side_to_move);
        const auto queenside_king = state.castle_rights.queenside_king_dest(state.side_to_move);
        const auto kingside_occupancy_mask = ROOK_RAY_BETWEEN[king_sq][kingside_rook] | kingside_king.to_bb() |
                                             state.castle_rights.kingside_rook_dest(state.side_to_move).to_bb();
        const auto queenside_occupancy_mask = ROOK_RAY_BETWEEN[king_sq][queenside_rook] | queenside_king.to_bb() |
                                              state.castle_rights.queenside_rook_dest(state.side_to_move).to_bb();
        const auto kingside_path_mask = ROOK_RAY_BETWEEN[king_sq][kingside_king] | kingside_king.to_bb();
        const auto queenside_path_mask = ROOK_RAY_BETWEEN[king_sq][queenside_king] | queenside_king.to_bb();

        if (state.castle_rights.can_kingside_castle(state.side_to_move) &&
            allowed_destinations.has_all_squares_set(kingside_path_mask) &&
            !occ.has_any_squares_set(kingside_occupancy_mask)) {
            move_list.emplace_back(king_sq, kingside_rook, MoveFlag::CASTLE);
        }

        if (state.castle_rights.can_queenside_castle(state.side_to_move) &&
            allowed_destinations.has_all_squares_set(queenside_path_mask) &&
            !occ.has_any_squares_set(queenside_occupancy_mask)) {
            move_list.emplace_back(king_sq, queenside_rook, MoveFlag::CASTLE);
        }
    }
}

void generate_moves(const BoardState &state, MoveList &move_list) {
    Bitboard allowed = Bitboard::ALL_SET;
    if (state.checkers != 0) {
        if (state.checkers.pop_count() > 1) {
            king_moves(state, move_list);
            return;
        }

        const auto checker = state.checkers;
        allowed = RAY_BETWEEN[checker.lsb()][state.king(state.side_to_move).lsb()] | checker;
    }

    pawn_moves(state, move_list, allowed);
    knight_moves(state, move_list, allowed);
    slider_moves(state, move_list, allowed);
    king_moves(state, move_list);
}
