#include "../uci/uci.hpp"
#include "move_gen.hpp"
#include "board_state.hpp"

#include <cassert>
#include <string>

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

void BoardState::compute_masks() {
    const auto our_king = king(side_to_move).lsb();
    const auto ortho = rooks(~side_to_move) | queens(~side_to_move);
    const auto diag = bishops(~side_to_move) | queens(~side_to_move);

    checkers = diag_pins = ortho_pins = 0;
    const auto occ = occupancy();
    for (auto potential_pinner : (BISHOP_RAYS[our_king] & diag)) {
        const auto ray = RAY_BETWEEN[our_king][potential_pinner];
        const auto blockers = occ & ray;
        checkers |= blockers.pop_count() == 0 ? potential_pinner.to_bb() : 0;
        diag_pins |= blockers.pop_count() == 1 ? ray | potential_pinner.to_bb() : 0;
    }
    for (auto potential_pinner : (ROOK_RAYS[our_king] & ortho)) {
        const auto ray = RAY_BETWEEN[our_king][potential_pinner];
        const auto blockers = occ & ray;

        checkers |= blockers.pop_count() == 0 ? potential_pinner.to_bb() : 0;
        ortho_pins |= blockers.pop_count() == 1 ? ray | potential_pinner.to_bb() : 0;
    }
    checkers |= knights(~side_to_move) & KNIGHT_MOVES[our_king];
    checkers |= king(~side_to_move) & KING_MOVES[our_king];
    checkers |= pawns(~side_to_move) & PAWN_ATTACKS[our_king][side_to_move];
}

std::string BoardState::to_fen() const {
    std::string res;

    for (i32 i = 8; i-- > 0;) {
        i32 empty = 0;
        for (i32 j = 0; j < 8; ++j) {
            const auto sq = Square(8 * i + j);
            const auto pt = get_piece_type(sq);
            if (pt == PieceType::NONE) {
                ++empty;
                continue;
            }

            if (empty) {
                res += '0' + empty;
                empty = 0;
            }

            const auto col = get_piece_color(sq);
            res += pt.to_char(col);
        }

        if (empty) {
            res += '0' + empty;
            empty = 0;
        }

        if (i)
            res += '/';
    }

    res += ' ';
    res += side_to_move == Color::WHITE ? 'w' : 'b';
    res += ' ';

    if (castle_rights.to_mask() != 0) {
        if (std::get<bool>(uci::options.get("UCI_Chess960")->value_as_variant())) {
            if (castle_rights.can_kingside_castle(Color::WHITE)) {
                res += castle_rights.kingside_rook(Color::WHITE).file().to_char();
            }
            if (castle_rights.can_queenside_castle(Color::WHITE)) {
                res += castle_rights.queenside_rook(Color::WHITE).file().to_char();
            }
            if (castle_rights.can_kingside_castle(Color::BLACK)) {
                res += castle_rights.kingside_rook(Color::BLACK).file().to_char();
            }
            if (castle_rights.can_queenside_castle(Color::BLACK)) {
                res += castle_rights.queenside_rook(Color::BLACK).file().to_char();
            }
        } else {
            if (castle_rights.can_kingside_castle(Color::WHITE)) {
                res += 'K';
            }
            if (castle_rights.can_queenside_castle(Color::WHITE)) {
                res += 'Q';
            }
            if (castle_rights.can_kingside_castle(Color::BLACK)) {
                res += 'k';
            }
            if (castle_rights.can_queenside_castle(Color::BLACK)) {
                res += 'q';
            }
        }
    } else {
        res += '-';
    }

    res += ' ';
    if (en_passant_sq != Square::NO_SQUARE) {
        res += en_passant_sq.file().to_char();
        res += en_passant_sq.rank().to_char();
    } else {
        res += '-';
    }

    res += ' ';
    res += std::to_string(fifty_moves_clock);
    res += ' ';
    res += '1';
    return res;
}
