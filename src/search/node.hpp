#ifndef NODE_HPP
#define NODE_HPP

#include "../chess/move.hpp"

namespace search {

class TerminalState {
  public:
    enum class Flag : u8 {
        NONE,
        WIN,
        DRAW,
        LOSS,
    };
    constexpr TerminalState() {}

    constexpr explicit TerminalState(u16 value) : value_(value) {}

    [[nodiscard]] constexpr static TerminalState none() {
        return TerminalState(static_cast<u8>(Flag::NONE) << 8);
    }

    [[nodiscard]] constexpr static TerminalState draw() {
        return TerminalState(static_cast<u8>(Flag::DRAW) << 8);
    }

    [[nodiscard]] constexpr static TerminalState draw(u8 ply) {
        return TerminalState(static_cast<u8>(Flag::DRAW) << 8 | ply);
    }

    [[nodiscard]] constexpr static TerminalState win(u8 ply) {
        return TerminalState(static_cast<u8>(Flag::WIN) << 8 | ply);
    }

    [[nodiscard]] constexpr static TerminalState loss(u8 ply) {
        return TerminalState(static_cast<u8>(Flag::LOSS) << 8 | ply);
    }

    [[nodiscard]] constexpr bool is_none() const {
        return flag() == Flag::NONE;
    }

    [[nodiscard]] constexpr bool is_draw() const {
        return flag() == Flag::DRAW;
    }

    [[nodiscard]] constexpr bool is_win() const {
        return flag() == Flag::WIN;
    }

    [[nodiscard]] constexpr bool is_loss() const {
        return flag() == Flag::LOSS;
    }

    [[nodiscard]] constexpr f32 score() const {
        switch (flag()) {
        case Flag::LOSS:
            return 0.0f;
        case Flag::WIN:
            return 1.0f;
        default:
            return 0.5f;
        }
    }

    [[nodiscard]] constexpr TerminalState operator-() const {
        switch (flag()) {
        case Flag::LOSS:
            return TerminalState::win(distance_to_terminal() + 1);
        case Flag::WIN:
            return TerminalState::loss(distance_to_terminal() + 1);
        default:
            return *this;
        }
    }

    [[nodiscard]] constexpr bool operator==(const TerminalState &other) const = default;

    [[nodiscard]] Flag flag() const {
        return static_cast<Flag>(value_ >> 8);
    }

    [[nodiscard]] u8 distance_to_terminal() const {
        return static_cast<u8>(value_ & 255);
    }

  private:
    u16 value_;
};

#include <limits>
#include <cassert>

class NodeIndex {
  public:
    static constexpr u32 NONE_INDEX = (1u << 31) - 1;

    constexpr NodeIndex(u32 index = NONE_INDEX, u8 half = 0) noexcept
        : index_bits(index), half_bits(half) {}

    static constexpr NodeIndex none() noexcept { return NodeIndex(NONE_INDEX, 0); }

    [[nodiscard]] constexpr bool is_none() const noexcept { return index_bits == NONE_INDEX; }

    // Avoid implicit integer conversion — it causes ambiguity.
    // If you really want it, make it explicit:
    explicit constexpr operator u32() const noexcept { return index_bits; }

    // Proper equality/inequality, const and noexcept, return bool by value.
    [[nodiscard]] constexpr bool operator==(const NodeIndex& other) const noexcept {
        return index_bits == other.index_bits && half_bits == other.half_bits;
    }
    [[nodiscard]] constexpr bool operator!=(const NodeIndex& other) const noexcept {
        return !(*this == other);
    }

    // No operator==(u32). Compare to NodeIndex::none() or use is_none().

    // Assign/modify
    NodeIndex& operator=(const NodeIndex&) = default;

    // Remove this; it leaves half_bits stale and creates confusion.
    // NodeIndex& operator=(i32) = delete;

    [[nodiscard]] constexpr u32 index() const noexcept { return index_bits; }
    [[nodiscard]] constexpr u8  half()  const noexcept { return half_bits; }

    NodeIndex& operator+=(i32 delta) noexcept {
        index_bits = static_cast<u32>(static_cast<i32>(index_bits) + delta);
        return *this;
    }
    NodeIndex& operator-=(i32 delta) noexcept { return (*this += -delta); }

    friend constexpr NodeIndex operator+(NodeIndex n, i32 d) noexcept { return n += d; }
    friend constexpr NodeIndex operator+(i32 d, NodeIndex n) noexcept { return n += d; }
    friend constexpr NodeIndex operator-(NodeIndex n, i32 d) noexcept { return n -= d; }

    friend constexpr NodeIndex operator+(const NodeIndex& a, const NodeIndex& b) {
        assert(a.half() == b.half());
        return NodeIndex(a.index() + b.index(), a.half());
    }
    friend constexpr NodeIndex operator-(const NodeIndex& a, const NodeIndex& b) {
        assert(a.half() == b.half());
        assert(a.index() >= b.index());
        return NodeIndex(a.index() - b.index(), a.half());
    }

  private:
    union {
        struct {
            u32 index_bits : 31;
            u32 half_bits  : 1;
        };
        u32 bits_;
    };
};


struct Node {
    // Sum of all scores that have been propagated back to this node
    f64 sum_of_scores = 0.0;
    // Policy given to us by our parent node
    f32 policy_score = 0.0;
    // Index of the parent node
    NodeIndex parent_idx = NodeIndex::none();
    // Index of the first child in the node table
    NodeIndex first_child_idx = NodeIndex::none();
    // Number of times this node has been visited
    u32 num_visits = 0;
    // Move that led i32o this node
    Move move = Move::null();
    // Number of legal moves this node has
    u16 num_children = 0;
    // What kind of state this (terminal) node is
    TerminalState terminal_state = TerminalState::none();

    // Average of all scores this node has received
    [[nodiscard]] f64 q() const;

    [[nodiscard]] bool visited() const;
    [[nodiscard]] bool terminal() const;
    [[nodiscard]] bool expanded() const;
};

} // namespace search

#endif // NODE_HPP
