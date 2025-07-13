#ifndef MOVE_GEN_HPP
#define MOVE_GEN_HPP

#include "../util/static_vector.hpp"
#include "bitboard.hpp"
#include "board_state.hpp"
#include <array>

using MoveList = util::StaticVector<Move, 218>;

Bitboard compute_bishop_attacks(Square sq, Bitboard occ);
Bitboard compute_rook_attacks(Square sq, Bitboard occ);

void pawn_moves(const BoardState &board, MoveList &move_list, Bitboard allowed_destinations = Bitboard::ALL_SET);
void knight_moves(const BoardState &board, MoveList &move_list, Bitboard allowed_destionations = Bitboard::ALL_SET);
void slider_moves(const BoardState &board, MoveList &move_list, Bitboard allowed_destionations = Bitboard::ALL_SET);
void king_moves(const BoardState &board, MoveList &move_list, Bitboard allowed_destionations = Bitboard::ALL_SET);
void generate_moves(const BoardState &board, MoveList &move_list);

#endif // MOVE_GEN_HPP