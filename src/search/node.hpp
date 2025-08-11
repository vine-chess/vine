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
    NodeIndex(u32 index, u8 half = 0) {
        index_bits = index;
        half_bits = half;
    }

    static NodeIndex none() {
        return NodeIndex((1u << 31) - 1);
    }

    [[nodiscard]] bool is_none() const {
        return index_bits == none();
    }

    [[nodiscard]] operator u32() const {
        return index_bits;
    }

    NodeIndex &operator=(const NodeIndex &other) {
        index_bits = other.index_bits;
        half_bits = other.half_bits;
        return *this;
    }

    NodeIndex &operator=(i32 index) {
        index_bits = index;
        return *this;
    }

    [[nodiscard]] u32 index() const {
        return index_bits;
    }

    [[nodiscard]] u8 half() const {
        return half_bits;
    }

    [[nodiscard]] NodeIndex &operator+=(i32 delta) {
        index_bits = static_cast<u32>(static_cast<i32>(index_bits) + delta);
        return *this;
    }

    [[nodiscard]] NodeIndex &operator-=(i32 delta) {
        return (*this += -delta);
    }

    [[nodiscard]] friend NodeIndex operator+(NodeIndex n, i32 delta) {
        return n += delta;
    }

    [[nodiscard]] friend NodeIndex operator+(i32 delta, NodeIndex n) {
        return n += delta;
    }

    [[nodiscard]] friend NodeIndex operator-(NodeIndex n, i32 delta) {
        return n -= delta;
    }

    [[nodiscard]] friend NodeIndex operator+(const NodeIndex &a, const NodeIndex &b) {
        assert(a.half() == b.half());
        u32 sum = a.index() + b.index();
        return NodeIndex(sum, a.half());
    }

    [[nodiscard]] friend NodeIndex operator-(const NodeIndex &a, const NodeIndex &b) {
        assert(a.half() == b.half());
        assert(a.index() >= b.index());
        return NodeIndex(a.index() - b.index(), a.half());
    }

  private:
    union {
        struct {
            u32 index_bits : 31;
            u32 half_bits : 1;
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
