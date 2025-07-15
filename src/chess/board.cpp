#include "board.hpp"
#include <iostream>
#include <sstream>

constexpr std::string_view STARTPOS_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

[[nodiscard]] char get_piece_ch(const BoardState &state, Square sq) {
    if (!state.occupancy().is_set(sq))
        return ' ';
    return state.get_piece_type(sq).to_char(state.get_piece_color(sq));
}

Board::Board(std::string_view fen) {
    state_history_.emplace_back();
    std::istringstream stream((std::string(fen)));

    std::string position;
    stream >> position;

    u8 square = Square::A8;
    for (const char &ch : position) {
        if (ch == '/') {
            square = square - 16 + (square % 8);
            continue;
        }

        if (std::isdigit(ch)) {
            square += ch - '0';
            continue;
        }

        state().place_piece(PieceType::from_char(ch), square, std::islower(ch) ? Color::BLACK : Color::WHITE);
        square++;
    }

    char side_to_move;
    stream >> side_to_move;
    state().side_to_move = side_to_move == 'w' ? Color::WHITE : Color::BLACK;

    std::string castle_rights;
    stream >> castle_rights;
    for (const char &ch : castle_rights) {
        if (ch == 'K')
            state().castle_rights.set_kingside_castle(Color::WHITE, true);
        else if (ch == 'Q')
            state().castle_rights.set_queenside_castle(Color::WHITE, true);
        else if (ch == 'k')
            state().castle_rights.set_kingside_castle(Color::BLACK, true);
        else if (ch == 'q')
            state().castle_rights.set_queenside_castle(Color::BLACK, true);
    }

    std::string en_passant;
    stream >> en_passant;

    if (en_passant != "-") {
        state().set_en_passant_sq(Square::from_string(en_passant));
    }

    stream >> state().fifty_moves_clock;
    state().compute_masks();
}

Board::Board() : Board(STARTPOS_FEN) {}

BoardState &Board::state() {
    return state_history_.back();
}

const BoardState &Board::state() const {
    return state_history_.back();
}

void Board::make_move(Move move) {
    state_history_.push_back(state());

    if (move.is_castling()) {
        state().remove_piece(PieceType::KING, move.from(), state().side_to_move);
        state().remove_piece(PieceType::ROOK, move.to(), state().side_to_move);
        state().place_piece(PieceType::KING, move.king_castling_to(), state().side_to_move);
        state().place_piece(PieceType::ROOK, move.rook_castling_to(), state().side_to_move);
        return;
    }

    const auto from_type = state().get_piece_type(move.from());
    auto to_type = state().get_piece_type(move.from());

    if (move.is_capture()) {
        Square target_square = move.to();
        if (move.is_ep()) {
            target_square = Square{move.from().rank(), move.to().file()};
        }
        state().remove_piece(state().get_piece_type(target_square), target_square, ~state().side_to_move);
    }

    if (move.is_promo()) {
        to_type = move.promo_type();
    }

    state().remove_piece(from_type, move.from(), state().side_to_move);
    state().place_piece(to_type, move.to(), state().side_to_move);
    state().side_to_move = ~state().side_to_move;
    state().compute_masks();
}

void Board::undo_move() {
    state_history_.pop_back();
}

std::ostream &operator<<(std::ostream &out, const Board &board) {
    for (int rank = 7; rank >= 0; rank--) {
        out << rank + 1 << ' ';
        for (int file = 0; file < 8; file++) {
            const auto square = Square(Rank(rank), File(file));
            out << get_piece_ch(board.state(), square);
            if (file < 7)
                out << ' ';
        }
        out << std::endl;
    }

    out << "  ";
    for (int file = 0; file < 8; file++)
        out << static_cast<char>('a' + file) << ' ';

    return out;
}
