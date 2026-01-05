#ifndef NODE_INDEX_HPP
#define NODE_INDEX_HPP

#include "../util/types.hpp"

namespace search {
class HalfIndex {
  public:
    static constexpr u8 LOWER = 0;
    static constexpr u8 UPPER = 1;

    constexpr HalfIndex(u8 v = LOWER) noexcept : value_(v & 0x1u) {}
    constexpr HalfIndex(const HalfIndex &) noexcept = default;
    HalfIndex &operator=(const HalfIndex &) noexcept = default;

    [[nodiscard]] constexpr HalfIndex operator~() const noexcept {
        return {static_cast<u8>(value_ ^ 0x1u)};
    }

    [[nodiscard]] constexpr bool operator==(const HalfIndex &other) const noexcept {
        return value_ == other.value_;
    }
    [[nodiscard]] constexpr bool operator!=(const HalfIndex &other) const noexcept {
        return value_ != other.value_;
    }

    [[nodiscard]] constexpr operator u8() const noexcept {
        return value_;
    }

    [[nodiscard]] static constexpr HalfIndex lower() noexcept {
        return {LOWER};
    }
    [[nodiscard]] static constexpr HalfIndex upper() noexcept {
        return {UPPER};
    }

  private:
    u8 value_;
};

class NodeIndex {
  public:
    static constexpr u32 INDEX_BITS = 31;
    static constexpr u32 INDEX_MASK = (1u << INDEX_BITS) - 1u;
    static constexpr u32 HALF_MASK = 1u << INDEX_BITS;
    static constexpr u32 NONE_INDEX = INDEX_MASK;

    constexpr NodeIndex(u32 index = NONE_INDEX, HalfIndex half = 0) noexcept : packed_(pack(index, half)) {}

    static constexpr NodeIndex none() noexcept {
        return NodeIndex(NONE_INDEX, 0);
    }

    [[nodiscard]] constexpr bool is_none() const noexcept {
        return index() == NONE_INDEX;
    }
    [[nodiscard]] explicit constexpr operator u32() const noexcept {
        return index();
    }

    [[nodiscard]] constexpr u32 index() const noexcept {
        return packed_ & INDEX_MASK;
    }
    [[nodiscard]] constexpr HalfIndex half() const noexcept {
        return {static_cast<u8>((packed_ & HALF_MASK) >> INDEX_BITS)};
    }

    NodeIndex &operator=(const NodeIndex &) = default;

    NodeIndex &operator+=(i32 delta) noexcept {
        packed_ = pack(static_cast<u32>(static_cast<i32>(index()) + delta), half());
        return *this;
    }
    NodeIndex &operator-=(i32 delta) noexcept {
        return (*this += -delta);
    }

    [[nodiscard]] constexpr bool operator==(const NodeIndex &other) const noexcept {
        return packed_ == other.packed_;
    }
    [[nodiscard]] constexpr bool operator!=(const NodeIndex &other) const noexcept {
        return !(*this == other);
    }

    friend constexpr NodeIndex operator+(NodeIndex n, i32 d) noexcept {
        n += d;
        return n;
    }
    friend constexpr NodeIndex operator+(i32 d, NodeIndex n) noexcept {
        n += d;
        return n;
    }
    friend constexpr NodeIndex operator-(NodeIndex n, i32 d) noexcept {
        n -= d;
        return n;
    }

    friend constexpr NodeIndex operator+(const NodeIndex &a, const NodeIndex &b) {
        return NodeIndex(a.index() + b.index(), a.half());
    }
    friend constexpr NodeIndex operator-(const NodeIndex &a, const NodeIndex &b) {
        return NodeIndex(a.index() - b.index(), a.half());
    }

    [[nodiscard]] static constexpr NodeIndex with_index(const NodeIndex &n, u32 new_index) noexcept {
        return NodeIndex(new_index, n.half());
    }
    [[nodiscard]] static constexpr NodeIndex with_half(const NodeIndex &n, u8 new_half) noexcept {
        return NodeIndex(n.index(), new_half);
    }

  private:
    static constexpr u32 pack(u32 index, HalfIndex half) noexcept {
        return index | (static_cast<u32>(half) << INDEX_BITS);
    }

    u32 packed_{pack(NONE_INDEX, 0)};
};

} // namespace search

#endif // NODE_INDEX_HPP
