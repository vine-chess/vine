#pragma once

#include "../util/types.hpp"

#include <bit>

class Bitboard {
  public:
    constexpr Bitboard() : raw_{} {}
    constexpr Bitboard(u64 bb) : raw_{bb} {}

    [[nodiscard]] constexpr Square lsb() const {
        return std::countr_zero(raw_);
    }

    constexpr void clear_lsb() {
        raw_ &= raw_ - 1;
    }

    [[nodiscard]] constexpr bool is_set(Square sq) const {
        return (raw_ >> static_cast<u8>(sq)) & 1;
    }

    constexpr void set(Square sq) {
        raw_ |= (u64{1} << static_cast<u8>(sq));
    }

    constexpr void unset(Square sq) {
        raw_ &= ~(u64{1} << static_cast<u8>(sq));
    }

    [[nodiscard]] constexpr bool empty() {
        return raw_ == 0;
    }

    [[nodiscard]] constexpr Bitboard operator+(Bitboard other) const {
        return Bitboard{raw_ + other.raw_};
    }

    constexpr Bitboard &operator+=(Bitboard other) {
        raw_ += other.raw_;
        return *this;
    }

    [[nodiscard]] constexpr Bitboard operator-(Bitboard other) const {
        return Bitboard{raw_ - other.raw_};
    }

    constexpr Bitboard &operator-=(Bitboard other) {
        raw_ -= other.raw_;
        return *this;
    }

    [[nodiscard]] constexpr Bitboard operator&(Bitboard other) const {
        return Bitboard{raw_ & other.raw_};
    }

    constexpr Bitboard &operator&=(Bitboard other) {
        raw_ &= other.raw_;
        return *this;
    }

    [[nodiscard]] constexpr Bitboard operator~() const {
        return Bitboard{~raw_};
    }

    [[nodiscard]] constexpr Bitboard operator|(Bitboard other) const {
        return Bitboard{raw_ | other.raw_};
    }

    constexpr Bitboard &operator|=(Bitboard other) {
        raw_ |= other.raw_;
        return *this;
    }

    [[nodiscard]] constexpr Bitboard operator<<(Bitboard other) const {
        return Bitboard{raw_ << other.raw_};
    }

    constexpr Bitboard &operator<<=(Bitboard other) {
        raw_ <<= other.raw_;
        return *this;
    }

    [[nodiscard]] constexpr Bitboard operator>>(Bitboard other) const {
        return Bitboard{raw_ >> other.raw_};
    }

    constexpr Bitboard &operator>>=(Bitboard other) {
        raw_ >>= other.raw_;
        return *this;
    }

    [[nodiscard]] constexpr bool operator==(Bitboard const &other) const = default;

    [[nodiscard]] constexpr explicit operator u64() const {
        return raw_;
    }

    class Iterator {
      public:
        u64 state_;

        constexpr Square operator*() const {
            return Square{static_cast<u8>(std::countr_zero(state_))};
        }

        constexpr Iterator &operator++() {
            state_ &= state_ - 1;
            return *this;
        }

        [[nodiscard]] constexpr bool operator==(Iterator const &other) const = default;

      private:
        friend class Bitboard;

        constexpr Iterator(Bitboard bb) : state_{static_cast<u64>(bb)} {}
    };

    [[nodiscard]] constexpr Iterator begin() {
        return Iterator{*this};
    }

    [[nodiscard]] constexpr Iterator begin() const {
        return Iterator{*this};
    }

    [[nodiscard]] constexpr Iterator end() {
        return Iterator{0};
    }

    [[nodiscard]] constexpr Iterator end() const {
        return Iterator{0};
    }

  private:
    u64 raw_;
};
