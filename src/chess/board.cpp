#include "board.hpp"

#include "move_gen.hpp"
#include "zobrist.hpp"

#include <cstdlib>
#include <iostream>
#include <sstream>

[[nodiscard]] char get_piece_ch(const BoardState &state, Square sq) {
    if (!state.occupancy().is_set(sq))
        return ' ';
    return state.get_piece_type(sq).to_char(state.get_piece_color(sq));
}

Board::Board(std::string_view fen) {
    history_.reserve(2048);
    history_.emplace_back();
    std::istringstream stream((std::string(fen)));

    std::string position;
    stream >> position;

    i32 square = Square::A8;
    for (const char &ch : position) {
        if (ch == '/') {
            square = square - 16 + square % 8;
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
    if (state().side_to_move == Color::BLACK) {
        state().hash_key ^= zobrist::side_to_move;
    }

    std::string castle_data;
    stream >> castle_data;

    if (castle_data != "-") {
        for (const char ch : castle_data) {
            if (ch == 'K') {
                state().castle_rights.set_kingside_rook_file(Color::WHITE, File::H);
            } else if (ch == 'Q') {
                state().castle_rights.set_queenside_rook_file(Color::WHITE, File::A);
            } else if (ch == 'k') {
                state().castle_rights.set_kingside_rook_file(Color::BLACK, File::H);
            } else if (ch == 'q') {
                state().castle_rights.set_queenside_rook_file(Color::BLACK, File::A);
            } else {
                const auto color = std::isupper(ch) ? Color::WHITE : Color::BLACK;
                const auto rook_file = File::from_char(ch);
                if (rook_file > state().king(color).lsb().file()) {
                    state().castle_rights.set_kingside_rook_file(color, rook_file);
                } else {
                    state().castle_rights.set_queenside_rook_file(color, rook_file);
                }
            }
        }
    }
    state().hash_key ^= zobrist::castle_rights[state().castle_rights.to_mask()];

    std::string en_passant;
    stream >> en_passant;

    if (en_passant != "-") {
        state().en_passant_sq = Square::from_string(en_passant);
        state().hash_key ^= zobrist::en_passant[state().en_passant_sq.file()];
    }

    int hmc;
    stream >> hmc;
    state().fifty_moves_clock = static_cast<u8>(hmc);
    state().compute_masks();
}

Board::Board(const BoardState &board_state) {
    history_.reserve(2048);
    history_.push_back(board_state);
}

BoardState &Board::state() {
    return history_.back();
}

const BoardState &Board::state() const {
    return history_.back();
}

const BoardState &Board::prev_state() const {
    return history_.at(history_.size() - 2);
}

Board::History &Board::history() {
    return history_;
}

const Board::History &Board::history() const {
    return history_;
}

bool Board::has_threefold_repetition() const {
    const u16 maximum_distance = std::min<u32>(state().fifty_moves_clock, history_.size());

    u16 times_seen = 1;
    for (i32 i = 3; i <= maximum_distance; i += 2) {
        if (state().hash_key == history_[history_.size() - i].hash_key && ++times_seen == 3) {
            return true;
        }
    }

    return false;
}

bool Board::is_fifty_move_draw() const {
    if (state().fifty_moves_clock < 100) {
        return false;
    }

    if (state().checkers == 0) {
        return true;
    }

    MoveList moves;
    generate_moves(state(), moves);
    return !moves.empty();
}

Move Board::create_move(std::string_view uci_move) const {
    MoveList moves;
    generate_moves(state(), moves);

    for (const auto move : moves) {
        if (move.to_string() == uci_move) {
            return move;
        }
    }

    throw std::runtime_error("cannot create illegal move");
}

void Board::make_move(Move move) {
    history_.push_back(state());

    const u8 old_castle_rights_mask = state().castle_rights.to_mask();

    state().fifty_moves_clock += 1;
    if (state().en_passant_sq != Square::NO_SQUARE) {
        state().hash_key ^= zobrist::en_passant[state().en_passant_sq.file()];
    }
    state().en_passant_sq = Square::NO_SQUARE;

    if (move.is_castling()) {
        state().remove_piece(PieceType::KING, move.from(), state().side_to_move);
        state().remove_piece(PieceType::ROOK, move.to(), state().side_to_move);
        state().place_piece(PieceType::KING, move.king_castling_to(), state().side_to_move);
        state().place_piece(PieceType::ROOK, move.rook_castling_to(), state().side_to_move);
        state().castle_rights.set_kingside_rook_file(state().side_to_move, File::NO_FILE);
        state().castle_rights.set_queenside_rook_file(state().side_to_move, File::NO_FILE);
        state().side_to_move = ~state().side_to_move;
        state().compute_masks();
        return;
    }

    const auto from_type = state().get_piece_type(move.from());
    auto to_type = state().get_piece_type(move.from());

    if (move.is_capture()) {
        state().fifty_moves_clock = 0;

        Square target_square = move.to();
        if (move.is_ep()) {
            target_square = Square{move.from().rank(), move.to().file()};
        }

        if (move.to() == state().castle_rights.kingside_rook(~state().side_to_move)) {
            state().castle_rights.set_kingside_rook_file(~state().side_to_move, File::NO_FILE);
        } else if (move.to() == state().castle_rights.queenside_rook(~state().side_to_move)) {
            state().castle_rights.set_queenside_rook_file(~state().side_to_move, File::NO_FILE);
        }

        state().remove_piece(state().get_piece_type(target_square), target_square, ~state().side_to_move);
    }

    if (move.is_promo()) {
        to_type = move.promo_type();
    }

    state().remove_piece(from_type, move.from(), state().side_to_move);
    state().place_piece(to_type, move.to(), state().side_to_move);

    if (from_type == PieceType::PAWN) {
        state().fifty_moves_clock = 0;
        if ((move.from() ^ move.to()) == 16) {
            state().en_passant_sq = (move.from() + move.to()) / 2;
            state().hash_key ^= zobrist::en_passant[state().en_passant_sq.file()];
        }
    } else if (from_type == PieceType::KING) {
        state().castle_rights.set_kingside_rook_file(state().side_to_move, File::NO_FILE);
        state().castle_rights.set_queenside_rook_file(state().side_to_move, File::NO_FILE);
    }

    if (move.from() == state().castle_rights.kingside_rook(state().side_to_move)) {
        state().castle_rights.set_kingside_rook_file(state().side_to_move, File::NO_FILE);
    } else if (move.from() == state().castle_rights.queenside_rook(state().side_to_move)) {
        state().castle_rights.set_queenside_rook_file(state().side_to_move, File::NO_FILE);
    }

    state().hash_key ^= zobrist::castle_rights[old_castle_rights_mask ^ state().castle_rights.to_mask()];
    state().hash_key ^= zobrist::side_to_move;
    state().side_to_move = ~state().side_to_move;
    state().compute_masks();
}

void Board::undo_move() {
    history_.pop_back();
}

void Board::undo_n_moves(usize n) {
    while (n--) {
        history_.pop_back();
    }
}

std::ostream &operator<<(std::ostream &out, const BoardState &board) {
    for (int rank = 7; rank >= 0; rank--) {
        out << rank + 1 << ' ';
        for (int file = 0; file < 8; file++) {
            const auto square = Square(Rank(rank), File(file));
            out << get_piece_ch(board, square);
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

std::ostream &operator<<(std::ostream &out, const Board &board) {
    return out << board.state();
}
