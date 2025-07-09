#pragma once

#include "../util/types.hpp"

#include <bit>

class BitBoard {
  public:
    constexpr BitBoard() : raw_{} {}
    constexpr BitBoard(u64 bb) : raw_{bb} {}

    [[nodiscard]] constexpr u8 get_lsb() const {
        return std::countr_zero(raw_);
    }

    constexpr void clear_lsb() {
        raw_ &= raw_ - 1;
    }

    [[nodiscard]] constexpr bool empty() {
        return raw_ == 0;
    }

    [[nodiscard]] constexpr BitBoard operator+(BitBoard other) const {
        return BitBoard{raw_ + other.raw_};
    }

    constexpr BitBoard &operator+=(BitBoard other) {
        raw_ += other.raw_;
        return *this;
    }

    [[nodiscard]] constexpr BitBoard operator-(BitBoard other) const {
        return BitBoard{raw_ - other.raw_};
    }

    constexpr BitBoard &operator-=(BitBoard other) {
        raw_ -= other.raw_;
        return *this;
    }

    [[nodiscard]] constexpr BitBoard operator|(BitBoard other) const {
        return BitBoard{raw_ | other.raw_};
    }

    constexpr BitBoard &operator|=(BitBoard other) {
        raw_ |= other.raw_;
        return *this;
    }

    [[nodiscard]] constexpr BitBoard operator<<(BitBoard other) const {
        return BitBoard{raw_ << other.raw_};
    }

    constexpr BitBoard &operator<<=(BitBoard other) {
        raw_ <<= other.raw_;
        return *this;
    }

    [[nodiscard]] constexpr BitBoard operator>>(BitBoard other) const {
        return BitBoard{raw_ >> other.raw_};
    }

    constexpr BitBoard &operator>>=(BitBoard other) {
        raw_ >>= other.raw_;
        return *this;
    }

    [[nodiscard]] constexpr bool operator==(BitBoard const &other) const = default;

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
        friend class BitBoard;

        constexpr Iterator(BitBoard bb) : state_{static_cast<u64>(bb)} {}
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
