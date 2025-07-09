#include "board.hpp"
#include <sstream>
#include <iostream>

constexpr std::string_view STARTPOS_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBKQBNR w KQkq - 0 1";

[[nodiscard]] char get_piece_ch(const BoardState &state, Square sq) {
    if (!state.occupancy().is_set(sq)) return ' ';
    return PIECE_TYPE_TO_CHAR[state.get_piece_color(sq)][state.get_piece_type(sq)];
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

        const auto piece_color = std::islower(ch) ? Color::BLACK : Color::WHITE;
        const auto piece_type = CHAR_TO_PIECE_TYPE.at(std::tolower(ch));
        state_.place_piece(piece_type, square, piece_color);

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
        state_.en_passant_sq = Square(static_cast<u8>(en_passant[1] - '1'), static_cast<u8>(en_passant[0] - 'a'));
    }

    stream >> state_.fifty_moves_clock;
}

Board::Board() : Board(STARTPOS_FEN) {}

void Board::print() {
    for (int rank = 7; rank >= 0; rank--) {
        std::cout << rank + 1 << ' ';
        for (int file = 0; file < 8; file++) {
            const auto square = Square(rank, file);
            std::cout << get_piece_ch(state_, square);
            if (file < 7) std::cout << ' ';
        }
        std::cout << std::endl;
    }
    std::cout << "  ";
    for (int file = 0; file < 8; file++)
        std::cout << static_cast<char>('a' + file) << ' ';
    std::cout << std::endl;
}