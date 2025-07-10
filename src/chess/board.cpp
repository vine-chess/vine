#include "board.hpp"
#include <iostream>
#include <sstream>

constexpr std::string_view STARTPOS_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBKQBNR w KQkq - 0 1";

[[nodiscard]] char get_piece_ch(const BoardState &state, Square sq) {
    if (!state.occupancy().is_set(sq))
        return ' ';
    return state.get_piece_type(sq).to_char(state.get_piece_color(sq));
}

Board::Board(std::string_view fen) {
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

        state_.place_piece(PieceType::from_char(ch), square, std::islower(ch) ? Color::BLACK : Color::WHITE);
        square++;
    }

    char side_to_move;
    stream >> side_to_move;
    state_.side_to_move = side_to_move == 'w' ? Color::WHITE : Color::BLACK;

    std::string castle_rights;
    stream >> castle_rights;
    for (const char &ch : castle_rights) {
        if (ch == 'K')
            state_.castle_rights.set_kingside_castle(Color::WHITE, true);
        else if (ch == 'Q')
            state_.castle_rights.set_queenside_castle(Color::WHITE, true);
        else if (ch == 'k')
            state_.castle_rights.set_kingside_castle(Color::BLACK, true);
        else if (ch == 'q')
            state_.castle_rights.set_queenside_castle(Color::BLACK, true);
    }

    std::string en_passant;
    stream >> en_passant;

    if (en_passant != "-") {
        state_.set_en_passant_sq(Square(static_cast<u8>(en_passant[1] - '1'), static_cast<u8>(en_passant[0] - 'a')));
    }

    stream >> state_.fifty_moves_clock;
}

Board::Board() : Board(STARTPOS_FEN) {}

BoardState &Board::state() {
    return state_;
}

std::ostream &operator<<(std::ostream &out, const Board &board) {
    for (int rank = 7; rank >= 0; rank--) {
        out << rank + 1 << ' ';
        for (int file = 0; file < 8; file++) {
            const auto square = Square(rank, file);
            out << get_piece_ch(board.state_, square);
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
