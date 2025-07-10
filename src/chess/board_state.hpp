#ifndef BOARD_STATE_HPP
#define BOARD_STATE_HPP

#include "../util/types.hpp"

#include "bitboard.hpp"
#include "castle_rights.hpp"
#include "move.hpp"
#include "zobrist.hpp"

struct BoardState {
    void place_piece(PieceType piece_type, Square sq, Color color);
    void remove_piece(PieceType piece_type, Square sq, Color color);
    void set_en_passant_sq(Square sq);

    [[nodiscard]] Bitboard occupancy() const;
    [[nodiscard]] Bitboard occupancy(Color color) const;
    [[nodiscard]] Bitboard pawns() const;
    [[nodiscard]] Bitboard pawns(Color color) const;
    [[nodiscard]] Bitboard knights() const;
    [[nodiscard]] Bitboard knights(Color color) const;
    [[nodiscard]] Bitboard bishops() const;
    [[nodiscard]] Bitboard bishops(Color color) const;
    [[nodiscard]] Bitboard rooks() const;
    [[nodiscard]] Bitboard rooks(Color color) const;
    [[nodiscard]] Bitboard queens() const;
    [[nodiscard]] Bitboard queens(Color color) const;
    [[nodiscard]] Bitboard kings() const;
    [[nodiscard]] Bitboard king(Color color) const;

    [[nodiscard]] PieceType get_piece_type(Square sq) const;
    [[nodiscard]] Color get_piece_color(Square sq) const;

    void make_move(Move move);

    std::array<Bitboard, 6> piece_bbs{};
    std::array<Bitboard, 2> side_bbs{};
    std::array<PieceType, 64> piece_type_on_sq{};
    Color side_to_move{Color::WHITE};
    Square en_passant_sq{};
    CastleRights castle_rights{};
    u8 fifty_moves_clock = 0;
    HashKey hash_key{};
};

#endif // BOARD_STATE_HPP
