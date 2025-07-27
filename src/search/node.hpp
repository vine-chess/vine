#ifndef NODE_HPP
#define NODE_HPP

#include "../chess/move.hpp"
#include <bits/types/wint_t.h>

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

struct Node {
    // Sum of all scores that have been propagated back to this node
    f64 sum_of_scores = 0.0;
    // Policy given to us by our parent node
    f32 policy_score = 0.0;
    // Index of the parent node
    i32 parent_idx = -1;
    // Index of the first child in the node table
    i32 first_child_idx = -1;
    // Number of times this node has been visited
    u32 num_visits = 0;
    // Move that led into this node
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
