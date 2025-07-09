#include "board_state.hpp"
#include <cassert>
#include <iostream>

void BoardState::place_piece(PieceType piece_type, Square sq, Color color) {
    piece_type_on_square[sq] = piece_type;
    piece_bbs[piece_type - 1].set(sq);
    side_bbs[color].set(sq);
}

void BoardState::remove_piece(PieceType piece_type, Square sq, Color color) {
    piece_type_on_square[sq] = NONE;
    piece_bbs[piece_type - 1].unset(sq);
    side_bbs[color].unset(sq);
}

Bitboard BoardState::occupancy() const {
    return side_bbs[WHITE] | side_bbs[BLACK];
}

Bitboard BoardState::occupancy(Color color) const {
    return side_bbs[color];
}

Bitboard BoardState::pawns() const {
    return piece_bbs[PAWN];
}

Bitboard BoardState::pawns(Color color) const {
    return piece_bbs[PAWN] & side_bbs[color];
}

Bitboard BoardState::knights() const {
    return piece_bbs[KNIGHT];
}

Bitboard BoardState::knights(Color color) const {
    return piece_bbs[KNIGHT] & side_bbs[color];
}

Bitboard BoardState::bishops() const {
    return piece_bbs[BISHOP];
}

Bitboard BoardState::bishops(Color color) const {
    return piece_bbs[BISHOP] & side_bbs[color];
}

Bitboard BoardState::rooks() const {
    return piece_bbs[ROOK];
}

Bitboard BoardState::rooks(Color color) const {
    return piece_bbs[ROOK] & side_bbs[color];
}

Bitboard BoardState::queens() const {
    return piece_bbs[QUEEN];
}

Bitboard BoardState::queens(Color color) const {
    return piece_bbs[QUEEN] & side_bbs[color];
}

Bitboard BoardState::kings() const {
    return piece_bbs[KING];
}

Bitboard BoardState::king(Color color) const {
    return piece_bbs[KING] & side_bbs[color];
}

PieceType BoardState::get_piece_type(Square sq) const {
    return piece_type_on_square[sq];
}

Color BoardState::get_piece_color(Square sq) const {
    assert(piece_type_on_square[sq] != NONE);
    return side_bbs[WHITE].is_set(sq) ? WHITE : BLACK;
}