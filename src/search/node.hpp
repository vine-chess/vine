#ifndef NODE_HPP
#define NODE_HPP

#include "../chess/move.hpp"

namespace search {

enum class TerminalState : u8 {
    NONE,
    WIN,
    DRAW,
    LOSS
};

struct Node {
    // Move that led into this node
    Move move;
    // Index of the first child in the node table
    u32 first_child_idx = 0;
    // Number of legal moves this node has
    u16 num_children = 0;
    // Number of times this node has been visited
    u32 num_visits = 0;
    // Sum of all scores that have been propagated back to this node
    f64 sum_of_scores = 0.0;
    // What kind of state this (terminal) node is
    TerminalState terminal_state = TerminalState::NONE;

    [[nodiscard]] bool terminal() const;
    [[nodiscard]] bool expanded() const;
};

} // namespace search

#endif // NODE_HPP