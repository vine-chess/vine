#ifndef BOARD_STATE_HPP
#define BOARD_STATE_HPP

#include "../util/types.hpp"

#include "bitboard.hpp"
#include "castle_rights.hpp"
#include "move.hpp"
#include "zobrist.hpp"
#include <ostream>

class ColoredPiece {
  public:
    constexpr ColoredPiece() = default;
    constexpr ColoredPiece(PieceType piece_type, Color color)
        : raw_(static_cast<u8>(piece_type) << COLOR_BITS | static_cast<u8>(color)) {}

    [[nodiscard]] constexpr static ColoredPiece none() {
        return ColoredPiece();
    }

    [[nodiscard]] constexpr PieceType piece_type() const {
        return PieceType{static_cast<u8>(raw_ >> COLOR_BITS)};
    }

    [[nodiscard]] constexpr Color color() const {
        return Color{static_cast<u8>(raw_ & COLOR_MASK)};
    }

    [[nodiscard]] constexpr bool operator==(ColoredPiece const &other) const = default;

    [[nodiscard]] constexpr operator u8() const {
        return raw_;
    }

  private:
    static constexpr u8 COLOR_BITS = 1;
    static constexpr u8 COLOR_MASK = 0b1;

    u8 raw_ = 0;
};

struct BoardState {
    void place_piece(PieceType piece_type, Square sq, Color color);
    void remove_piece(PieceType piece_type, Square sq, Color color);

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

    [[nodiscard]] Bitboard threats_by(Color color) const;
    [[nodiscard]] Bitboard pinned_threats_by(Color color) const;

    std::array<Bitboard, 6> piece_bbs{};
    std::array<Bitboard, 2> side_bbs{};
    std::array<ColoredPiece, 64> piece_type_on_sq{};
    Color side_to_move{Color::WHITE};
    Square en_passant_sq{};
    CastleRights castle_rights{};
    u8 fifty_moves_clock = 0;
    HashKey hash_key{};
    Bitboard ortho_pins{};
    Bitboard diag_pins{};
    Bitboard checkers{};

    void compute_masks();
    [[nodiscard]] std::string to_fen() const;
};

inline bool operator==(const BoardState &lhs, const BoardState &rhs) noexcept {
    return lhs.piece_bbs == rhs.piece_bbs && lhs.side_bbs == rhs.side_bbs &&
           lhs.piece_type_on_sq == rhs.piece_type_on_sq && lhs.side_to_move == rhs.side_to_move &&
           lhs.en_passant_sq == rhs.en_passant_sq && lhs.fifty_moves_clock == rhs.fifty_moves_clock &&
           lhs.ortho_pins == rhs.ortho_pins && lhs.diag_pins == rhs.diag_pins && lhs.checkers == rhs.checkers &&
           lhs.castle_rights.to_mask() == rhs.castle_rights.to_mask();
}

inline bool operator!=(const BoardState &lhs, const BoardState &rhs) noexcept {
    return !(lhs == rhs);
}

#endif // BOARD_STATE_HPP
