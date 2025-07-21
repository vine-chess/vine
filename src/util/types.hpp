#pragma once

#include <array>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <exception>
#include <ostream>
#include <stdexcept>

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

class Rank {
  public:
    enum RankEnum {
        FIRST,
        SECOND,
        THIRD,
        FOURTH,
        FIFTH,
        SIXTH,
        SEVENTH,
        EIGHTH,
        NO_RANK
    };

    constexpr Rank() : raw_(NO_RANK) {}
    constexpr explicit Rank(u8 r) : raw_(r) {}
    constexpr Rank(RankEnum r) : raw_(r) {}

    [[nodiscard]] constexpr operator u8() const {
        return raw_;
    }

    [[nodiscard]] constexpr static Rank from_char(char ch) {
        return Rank(ch - '1');
    }
    [[nodiscard]] constexpr char to_char() const {
        return raw_ + '1';
    }

    constexpr Rank operator++(int) {
        const Rank tmp = *this;
        raw_++;
        return tmp;
    }

    constexpr Rank &operator++() {
        ++raw_;
        return *this;
    }

  private:
    u8 raw_;
};

class File {
  public:
    enum FileEnum {
        A,
        B,
        C,
        D,
        E,
        F,
        G,
        H,
        NO_FILE
    };

    constexpr File() : raw_(NO_FILE) {}
    constexpr explicit File(u8 f) : raw_(f) {}
    constexpr File(FileEnum f) : raw_(f) {}

    [[nodiscard]] constexpr operator u8() const {
        return raw_;
    }

    [[nodiscard]] constexpr static File from_char(char ch) {
        return File(std::tolower(ch) - 'a');
    }

    [[nodiscard]] constexpr char to_char() const {
        return raw_ + 'a';
    }

    constexpr File operator++(int) {
        const File tmp = *this;
        raw_++;
        return tmp;
    }

    constexpr File &operator++() {
        ++raw_;
        return *this;
    }

  private:
    u8 raw_;
};

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

    constexpr Square() : raw_(NO_SQUARE) {}
    constexpr Square(u8 sq) : raw_(sq) {}
    constexpr Square(Rank rank, File file) : raw_(rank * 8 + file) {}
    [[nodiscard]] constexpr static Square from_string(std::string_view sv) {
        return Square(Rank::from_char(sv[1]), File::from_char(sv[0]));
    }

    [[nodiscard]] constexpr bool is_valid() const {
        return raw_ < 64;
    }

    [[nodiscard]] constexpr u64 to_bb() const {
        if (!is_valid()) {
            assert(false);
        }
        return 1ull << *this;
    }

    [[nodiscard]] constexpr Rank rank() const {
        return Rank(raw_ >> 3);
    }

    [[nodiscard]] constexpr File file() const {
        return File(raw_ & 7);
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

inline std::ostream &operator<<(std::ostream &os, Square sq) {
    os << sq.file().to_char() << sq.rank().to_char();
    return os;
}

class Color {
  public:
    enum ColorEnum : u8 {
        WHITE,
        BLACK,
        NO_COLOR = 2
    };

    constexpr Color() : raw_(NO_COLOR) {}
    constexpr explicit Color(u8 i) : raw_(i) {}
    constexpr Color(ColorEnum c) : raw_(c) {}

    [[nodiscard]] constexpr bool operator==(const Color &other) const {
        return raw_ == other.raw_;
    }

    [[nodiscard]] constexpr bool operator==(ColorEnum c) const {
        return raw_ == c;
    }

    [[nodiscard]] constexpr bool operator!=(const Color &other) const {
        return raw_ != other.raw_;
    }

    [[nodiscard]] constexpr operator u8() const {
        return raw_;
    }

    [[nodiscard]] constexpr Color operator~() const {
        return Color(raw_ ^ 1);
    }

  private:
    u8 raw_;
};

class PieceType {
  public:
    enum PieceTypeEnum {
        NONE,
        PAWN,
        KNIGHT,
        BISHOP,
        ROOK,
        QUEEN,
        KING,
    };
    constexpr PieceType() {}
    constexpr explicit PieceType(u8 i) : raw_(i) {}

    constexpr PieceType(PieceTypeEnum pte) : raw_(pte) {}

    [[nodiscard]] constexpr static PieceType from_char(char ch) {
        return CHAR_TO_PIECE[std::tolower(ch)];
    }

    [[nodiscard]] constexpr char to_char(Color col) const {
        return col == Color::WHITE ? std::toupper(PIECE_TO_CHAR[raw_]) : PIECE_TO_CHAR[raw_];
    }

    [[nodiscard]] constexpr char to_char() const {
        return PIECE_TO_CHAR[raw_];
    }

    [[nodiscard]] constexpr operator u8() const {
        return raw_;
    }

  private:
    constexpr static auto PIECE_TO_CHAR = " pnbrqk";
    constexpr static auto CHAR_TO_PIECE = []() {
        std::array<PieceTypeEnum, 256> res{};
        res['p'] = PieceTypeEnum::PAWN;
        res['n'] = PieceTypeEnum::KNIGHT;
        res['b'] = PieceTypeEnum::BISHOP;
        res['r'] = PieceTypeEnum::ROOK;
        res['q'] = PieceTypeEnum::QUEEN;
        res['k'] = PieceTypeEnum::KING;
        return res;
    }();
    u8 raw_ = NONE;
};
