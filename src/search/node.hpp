#ifndef NODE_HPP
#define NODE_HPP

#include "../chess/move.hpp"
#include "tree_half.hpp"

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

class NodeIndex {
  public:
    static constexpr u32 INDEX_BITS = 31;
    static constexpr u32 INDEX_MASK = (1u << INDEX_BITS) - 1u;
    static constexpr u32 HALF_MASK = 1u << INDEX_BITS;
    static constexpr u32 NONE_INDEX = INDEX_MASK;

    constexpr NodeIndex(u32 index = NONE_INDEX, TreeHalf::Index half = 0) noexcept : packed_(pack(index, half)) {}

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
    [[nodiscard]] constexpr TreeHalf::Index half() const noexcept {
        return {static_cast<u8>((packed_ & HALF_MASK) >> INDEX_BITS)};
    }

    NodeIndex &operator=(const NodeIndex &) = default;

    NodeIndex &operator+=(i32 delta) noexcept {
        u32 i = (packed_ & INDEX_MASK);
        i = static_cast<u32>(static_cast<i32>(i) + delta);
        packed_ = (packed_ & HALF_MASK) | (i & INDEX_MASK);
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

    friend constexpr NodeIndex operator+(NodeIndex n, uint32_t rhs) noexcept {
        uint32_t i = (n.packed_ & INDEX_MASK) + rhs;
        n.packed_ = (n.packed_ & HALF_MASK) | (i & INDEX_MASK);
        return n;
    }
    friend constexpr NodeIndex operator-(NodeIndex n, uint32_t rhs) noexcept {
        uint32_t i = (n.packed_ & INDEX_MASK) - rhs;
        n.packed_ = (n.packed_ & HALF_MASK) | (i & INDEX_MASK);
        return n;
    }

    [[nodiscard]] static constexpr NodeIndex with_index(const NodeIndex &n, u32 new_index) noexcept {
        return NodeIndex(new_index, n.half());
    }
    [[nodiscard]] static constexpr NodeIndex with_half(const NodeIndex &n, u8 new_half) noexcept {
        return NodeIndex(n.index(), new_half);
    }

  private:
    static constexpr u32 pack(u32 index, TreeHalf::Index half) noexcept {
        return index | (static_cast<u32>(half) << INDEX_BITS);
    }

    u32 packed_{pack(NONE_INDEX, 0)};
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
    // Move that led into this node
    Move move = Move::null();
    // Number of legal moves this node has
    u16 num_children = 0;
    // What kind of state this (terminal) node is
    TerminalState terminal_state = TerminalState::none();

    [[nodiscard]] bool visited() const;
    [[nodiscard]] bool terminal() const;
    [[nodiscard]] bool expanded() const;
    // Average of all scores this node has received
    [[nodiscard]] f64 q() const;
};

} // namespace search

#endif // NODE_HPP
