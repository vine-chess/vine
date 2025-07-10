#include "board_state.hpp"
#include <cassert>
#include <iostream>

void BoardState::place_piece(PieceType piece_type, Square sq, Color color) {
    piece_type_on_sq[sq] = piece_type;
    piece_bbs[piece_type - 1].set(sq);
    side_bbs[color].set(sq);
    hash_key ^= zobrist::pieces[piece_type - 1][color][sq];
}

void BoardState::remove_piece(PieceType piece_type, Square sq, Color color) {
    piece_type_on_sq[sq] = PieceType::NONE;
    piece_bbs[piece_type - 1].unset(sq);
    side_bbs[color].unset(sq);
    hash_key ^= zobrist::pieces[piece_type - 1][color][sq];
}

void BoardState::set_en_passant_sq(Square sq) {
    if (en_passant_sq.is_valid()) {
        hash_key ^= zobrist::en_passant[sq.file()];
    }
    en_passant_sq = sq;
    hash_key ^= zobrist::en_passant[sq.file()];
}

Bitboard BoardState::occupancy() const {
    return side_bbs[WHITE] | side_bbs[BLACK];
}

Bitboard BoardState::occupancy(Color color) const {
    return side_bbs[color];
}

Bitboard BoardState::pawns() const {
    return piece_bbs[PieceType::PAWN];
}

Bitboard BoardState::pawns(Color color) const {
    return piece_bbs[PieceType::PAWN] & side_bbs[color];
}

Bitboard BoardState::knights() const {
    return piece_bbs[PieceType::KNIGHT];
}

Bitboard BoardState::knights(Color color) const {
    return piece_bbs[PieceType::KNIGHT] & side_bbs[color];
}

Bitboard BoardState::bishops() const {
    return piece_bbs[PieceType::BISHOP];
}

Bitboard BoardState::bishops(Color color) const {
    return piece_bbs[PieceType::BISHOP] & side_bbs[color];
}

Bitboard BoardState::rooks() const {
    return piece_bbs[PieceType::ROOK];
}

Bitboard BoardState::rooks(Color color) const {
    return piece_bbs[PieceType::ROOK] & side_bbs[color];
}

Bitboard BoardState::queens() const {
    return piece_bbs[PieceType::QUEEN];
}

Bitboard BoardState::queens(Color color) const {
    return piece_bbs[PieceType::QUEEN] & side_bbs[color];
}

Bitboard BoardState::kings() const {
    return piece_bbs[PieceType::KING];
}

Bitboard BoardState::king(Color color) const {
    return piece_bbs[PieceType::KING] & side_bbs[color];
}

PieceType BoardState::get_piece_type(Square sq) const {
    return piece_type_on_sq[sq];
}

Color BoardState::get_piece_color(Square sq) const {
    assert(piece_type_on_sq[sq] != PieceType::NONE);
    return side_bbs[WHITE].is_set(sq) ? WHITE : BLACK;
}
