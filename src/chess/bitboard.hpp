#pragma once

#include "../util/types.hpp"

#include <bit>
#include <iostream>
#include <type_traits>
#include <utility>

class Bitboard {
  public:
    constexpr Bitboard() : raw_{} {}
    constexpr Bitboard(u64 bb) : raw_{bb} {}

    constexpr static u64 ALL_SET = 0xffffffffffffffff;

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

    [[nodiscard]] constexpr Bitboard operator^(Bitboard other) const {
        return Bitboard{raw_ ^ other.raw_};
    }

    constexpr Bitboard &operator^=(Bitboard other) {
        raw_ ^= other.raw_;
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

    class RankDifference {
      public:
        [[nodiscard]] constexpr RankDifference operator*(int i) const {
            return RankDifference{raw_ * i};
        }
        constexpr explicit operator int() const {
            return raw_;
        }

        int raw_;
    };

    class FileDifference {
      public:
        [[nodiscard]] constexpr FileDifference operator*(int i) const {
            return FileDifference{raw_ * i};
        }
        constexpr explicit operator int() const {
            return raw_;
        }
        int raw_;
    };

    constexpr static auto UP = RankDifference{1};
    constexpr static auto DOWN = RankDifference{-1};
    constexpr static auto LEFT = FileDifference{-1};
    constexpr static auto RIGHT = FileDifference{1};

    template <auto rank_diff, auto file_diff>
        requires((std::is_same_v<decltype(rank_diff), RankDifference> ||
                  (std::is_same_v<decltype(rank_diff), int> && rank_diff == 0)) &&
                 (std::is_same_v<decltype(file_diff), FileDifference> ||
                  (std::is_same_v<decltype(file_diff), int> and file_diff == 0)))
    [[nodiscard]] constexpr Bitboard shift_masked() const {
        constexpr auto rank = static_cast<int>(rank_diff);
        constexpr auto file = static_cast<int>(file_diff);

        constexpr auto FILE_MASK = ((file > 0 ? 0xff << file : 0xff >> -file) & 0xff) * (-1ull / 0xff);
        auto res = *this;
        res = file > 0 ? res << file : res >> -file;
        res = rank > 0 ? res << rank * 8 : res >> -rank * 8;
        return res & FILE_MASK;
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
