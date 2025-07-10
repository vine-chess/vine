#ifndef MOVE_HPP
#define MOVE_HPP

#include "../util/types.hpp"

class MoveFlag {
  public:
    enum MoveFlagEnum {
        NORMAL = 0b0000,
        CASTLE = 0b0001,
        CAPTURE_BIT = 0b0100,
        EN_PASSANT = 0b0110,
        PROMOTION_BIT = 0b1000,
        PROMO_KNIGHT = (0b1000 | (static_cast<u8>(PieceType::KNIGHT) - 2)),
        PROMO_BISHOP = (0b1000 | (static_cast<u8>(PieceType::BISHOP) - 2)),
        PROMO_ROOK = (0b1000 | (static_cast<u8>(PieceType::ROOK) - 2)),
        PROMO_QUEEN = (0b1000 | (static_cast<u8>(PieceType::QUEEN) - 2)),
        PROMO_CAPTURE = 0b1100,
        PROMO_KNIGHT_CAPTURE = (0b1100 | (static_cast<u8>(PieceType::KNIGHT) - 2)),
        PROMO_BISHOP_CAPTURE = (0b1100 | (static_cast<u8>(PieceType::BISHOP) - 2)),
        PROMO_ROOK_CAPTURE = (0b1100 | (static_cast<u8>(PieceType::ROOK) - 2)),
        PROMO_QUEEN_CAPTURE = (0b1100 | (static_cast<u8>(PieceType::QUEEN) - 2)),
    };
    constexpr MoveFlag() : raw_{NORMAL} {}
    explicit constexpr MoveFlag(u8 raw) : raw_(raw) {}
    constexpr MoveFlag(MoveFlagEnum e) : raw_(e) {}

    [[nodiscard]] constexpr operator u8() const {
        return raw_;
    }

  private:
    u8 raw_;
};

class Move {
    [[nodiscard]] constexpr explicit Move(Square from, Square to, MoveFlag flag = MoveFlag::NORMAL) : raw_{static_cast<u16>(flag << 12 | to << 6 | from)} {
    }

    [[nodiscard]] constexpr Square from() const {
        return Square{static_cast<u8>(raw_ & 0b111111)};
    }

    [[nodiscard]] constexpr Square to() const {
        return Square{static_cast<u8>(raw_ >> 6 & 0b111111)};
    }

    [[nodiscard]] constexpr MoveFlag flag() const {
        return MoveFlag{raw_flag()};
    }

    [[nodiscard]] constexpr bool is_capture() const {
        return (raw_flag() & static_cast<u8>(MoveFlag::CAPTURE_BIT)) != 0;
    }

    [[nodiscard]] constexpr bool is_promo() const {
        return (raw_flag() & static_cast<u8>(MoveFlag::PROMOTION_BIT)) != 0;
    }

    [[nodiscard]] constexpr bool is_ep() const {
        return flag() == MoveFlag::EN_PASSANT;
    }

    [[nodiscard]] constexpr bool is_castling() const {
        return flag() == MoveFlag::CASTLE;
    }

    [[nodiscard]] constexpr bool is_null() const {
        return raw_ == 0;
    }

    [[nodiscard]] constexpr bool operator==(Move const &) const = default;

  private:
    [[nodiscard]] constexpr u8 raw_flag() const {
        return static_cast<u8>(raw_ >> 12);
    }
    u16 raw_;
};

#endif // MOVE_HPP
