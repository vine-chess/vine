#include "board_state.hpp"
#include "move.hpp"

#include <cassert>

void BoardState::make_move(Move move) {
    if (move.is_castling()) {
        remove_piece(PieceType::KING, move.from(), side_to_move);
        remove_piece(PieceType::ROOK, move.to(), side_to_move);
        place_piece(PieceType::KING, move.king_castling_to(), side_to_move);
        place_piece(PieceType::ROOK, move.rook_castling_to(), side_to_move);
        return;
    }

    PieceType from_type = piece_type_on_sq[move.from()];
    PieceType to_type = piece_type_on_sq[move.from()];
    if (move.is_capture()) {
        Square target_square = move.to();
        if (move.is_ep()) {
            target_square = Square{move.from().rank(), move.to().file()};
        }
        remove_piece(piece_type_on_sq[target_square], target_square, ~side_to_move);
    }
    if (move.is_promo()) {
        to_type = move.promo_type();
    }
    remove_piece(from_type, move.from(), side_to_move);
    place_piece(to_type, move.to(), side_to_move);
    side_to_move = ~side_to_move;
}

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
    return side_bbs[Color::WHITE] | side_bbs[Color::BLACK];
}

Bitboard BoardState::occupancy(Color color) const {
    return side_bbs[color];
}

Bitboard BoardState::pawns() const {
    return piece_bbs[PieceType::PAWN - 1];
}

Bitboard BoardState::pawns(Color color) const {
    return piece_bbs[PieceType::PAWN - 1] & side_bbs[color];
}

Bitboard BoardState::knights() const {
    return piece_bbs[PieceType::KNIGHT - 1];
}

Bitboard BoardState::knights(Color color) const {
    return piece_bbs[PieceType::KNIGHT - 1] & side_bbs[color];
}

Bitboard BoardState::bishops() const {
    return piece_bbs[PieceType::BISHOP - 1];
}

Bitboard BoardState::bishops(Color color) const {
    return piece_bbs[PieceType::BISHOP - 1] & side_bbs[color];
}

Bitboard BoardState::rooks() const {
    return piece_bbs[PieceType::ROOK - 1];
}

Bitboard BoardState::rooks(Color color) const {
    return piece_bbs[PieceType::ROOK - 1] & side_bbs[color];
}

Bitboard BoardState::queens() const {
    return piece_bbs[PieceType::QUEEN - 1];
}

Bitboard BoardState::queens(Color color) const {
    return piece_bbs[PieceType::QUEEN - 1] & side_bbs[color];
}

Bitboard BoardState::kings() const {
    return piece_bbs[PieceType::KING - 1];
}

Bitboard BoardState::king(Color color) const {
    return piece_bbs[PieceType::KING - 1] & side_bbs[color];
}

PieceType BoardState::get_piece_type(Square sq) const {
    return piece_type_on_sq[sq];
}

Color BoardState::get_piece_color(Square sq) const {
    assert(piece_type_on_sq[sq] != PieceType::NONE);
    return side_bbs[Color::WHITE].is_set(sq) ? Color::WHITE : Color::BLACK;
}
