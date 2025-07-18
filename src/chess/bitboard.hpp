#pragma once

#include "../util/types.hpp"

#include <bit>
#include <type_traits>

class Bitboard {
  public:
    constexpr Bitboard() : raw_{} {}
    constexpr Bitboard(u64 bb) : raw_{bb} {}
    constexpr explicit Bitboard(Square sq) : raw_{1ull << sq} {}

    constexpr static u64 ALL_SET = 0xffffffffffffffff;
    constexpr static u64 FIRST_RANK = 0xff;

    constexpr static Bitboard rank_mask(Rank which) {
        return FIRST_RANK << 8 * which;
    }

    constexpr static Bitboard file_mask(File which) {
        return ALL_SET / FIRST_RANK << which;
    }

    [[nodiscard]] constexpr u16 pop_count() const {
        return std::popcount(raw_);
    }

    [[nodiscard]] constexpr Square lsb() const {
        return std::countr_zero(raw_);
    }

    [[nodiscard]] constexpr Square msb() const {
        return raw_ ? 63 - std::countl_zero(raw_) : 64;
    }

    [[nodiscard]] constexpr Bitboard rotl(i8 shift) const {
        return std::rotl(raw_, shift);
    }

    [[nodiscard]] constexpr Bitboard rotr(i8 shift) const {
        return std::rotr(raw_, shift);
    }

    constexpr void clear_lsb() {
        raw_ &= raw_ - 1;
    }

    [[nodiscard]] constexpr bool is_set(Square sq) const {
        return (raw_ >> static_cast<u8>(sq)) & 1;
    }

    [[nodiscard]] constexpr bool has_all_squares_set(Bitboard bb) const {
        return (bb & raw_) == bb;
    }
    
    [[nodiscard]] constexpr bool has_any_squares_set(Bitboard bb) const {
        return (bb & raw_) != 0;
    }

    constexpr void set(Square sq) {
        raw_ |= (u64{1} << static_cast<u8>(sq));
    }

    constexpr void unset(Square sq) {
        raw_ &= ~(u64{1} << static_cast<u8>(sq));
    }

    [[nodiscard]] constexpr Bitboard reverse_bits() const {
        auto x = raw_;
        x = ((x >> 1) & 0x5555555555555555ULL) | ((x & 0x5555555555555555ULL) << 1);
        x = ((x >> 2) & 0x3333333333333333ULL) | ((x & 0x3333333333333333ULL) << 2);
        x = ((x >> 4) & 0x0F0F0F0F0F0F0F0FULL) | ((x & 0x0F0F0F0F0F0F0F0FULL) << 4);
        x = ((x >> 8) & 0x00FF00FF00FF00FFULL) | ((x & 0x00FF00FF00FF00FFULL) << 8);
        x = ((x >> 16) & 0x0000FFFF0000FFFFULL) | ((x & 0x0000FFFF0000FFFFULL) << 16);
        x = (x >> 32) | (x << 32);
        return Bitboard{x};
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
        requires((std::is_same_v<std::remove_cvref_t<decltype(rank_diff)>, RankDifference> ||
                  (std::is_same_v<std::remove_cvref_t<decltype(rank_diff)>, int> && rank_diff == 0)) &&
                 (std::is_same_v<std::remove_cvref_t<decltype(file_diff)>, FileDifference> ||
                  (std::is_same_v<std::remove_cvref_t<decltype(file_diff)>, int> and file_diff == 0)))
    [[nodiscard]] constexpr Bitboard shift() const {
        constexpr auto rank = static_cast<int>(rank_diff);
        constexpr auto file = static_cast<int>(file_diff);

        constexpr auto FILE_MASK = (std::rotl<u64>(FIRST_RANK, file) & FIRST_RANK) * (ALL_SET / FIRST_RANK);
        auto res = *this;
        if constexpr (file > 0) {
            res <<= file;
        } else {
            res >>= -file;
        }
        if constexpr (rank > 0) {
            res <<= rank * 8;
        } else {
            res >>= -rank * 8;
        }
        return res & FILE_MASK;
    }

    template <auto rank_diff, auto file_diff>
    [[nodiscard]] static constexpr Bitboard get_ray(Square sq) {
        if (sq == Square::NO_SQUARE) {
            return 0;
        }
        auto res = Bitboard(sq);
        for (int i = 0; i < 7; ++i) {
            res |= res.shift<rank_diff, file_diff>();
        }
        return res ^ Bitboard(sq);
    }

    template <auto rank_diff, auto file_diff>
    static constexpr auto ray_precomp = []() {
        std::array<u64, 65> ray_precomp_res{};
        for (int i = 0; i < 65; ++i) {
            ray_precomp_res[i] = static_cast<u64>(get_ray<rank_diff, file_diff>(i));
        }
        return ray_precomp_res;
    }();
    template <auto rank_diff, auto file_diff>
    [[nodiscard]] static constexpr Bitboard get_ray_precomputed(Square sq) {
        return ray_precomp<rank_diff, file_diff>[sq];
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
