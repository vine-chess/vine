#pragma once

#include <array>
#include <cstdint>
#include <unordered_map>

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;
using f32 = float;
using f64 = double;
using usize = std::size_t;

class Square {
  public:
    // clang-format off
    enum {
        A1, B1, C1, D1, E1, F1, G1, H1,
        A2, B2, C2, D2, E2, F2, G2, H2,
        A3, B3, C3, D3, E3, F3, G3, H3,
        A4, B4, C4, D4, E4, F4, G4, H4,
        A5, B5, C5, D5, E5, F5, G5, H5,
        A6, B6, C6, D6, E6, F6, G6, H6,
        A7, B7, C7, D7, E7, F7, G7, H7,
        A8, B8, C8, D8, E8, F8, G8, H8,
        NO_SQUARE
    };
    // clang-format on

    constexpr Square() {}
    constexpr Square(u8 sq) : raw_(sq) {}
    constexpr Square(u8 rank, u8 file) : raw_(rank * 8 + file) {}

    [[nodiscard]] constexpr bool is_valid() {
        return raw_ < 64;
    }

    [[nodiscard]] constexpr int rank() const {
        return raw_ >> 3;
    }

    [[nodiscard]] constexpr int file() const {
        return raw_ & 7;
    }

    [[nodiscard]] constexpr operator u8() const {
        return raw_;
    }

    constexpr Square operator++(int) {
        const Square tmp = *this;
        raw_++;
        return tmp;
    }

    constexpr Square &operator++() {
        ++raw_;
        return *this;
    }

  private:
    u8 raw_;
};

enum Color : u8 {
    WHITE,
    BLACK
};

enum PieceType : u8 {
    NONE,
    PAWN,
    KNIGHT,
    BISHOP,
    ROOK,
    QUEEN,
    KING
};

const std::unordered_map<char, PieceType> CHAR_TO_PIECE_TYPE = {
    {'p', PieceType::PAWN}, {'n', PieceType::KNIGHT}, {'b', PieceType::BISHOP},
    {'r', PieceType::ROOK}, {'q', PieceType::QUEEN},  {'k', PieceType::KING},
};

constexpr std::array<std::array<char, 7>, 2> PIECE_TYPE_TO_CHAR = {{
    // WHITE
    {' ', 'P', 'N', 'B', 'R', 'Q', 'K'},
    // BLACK
    {' ', 'p', 'n', 'b', 'r', 'q', 'k'},
}};
