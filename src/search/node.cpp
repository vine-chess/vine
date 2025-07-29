#include "node.hpp"

namespace search {

bool Node::terminal() const {
    return (terminal_state.is_loss() || terminal_state.is_draw() || terminal_state.is_win()) &&
           terminal_state.distance_to_terminal() == 0;
}

bool Node::visited() const {
    return num_visits > 0;
}

bool Node::expanded() const {
    return first_child_idx != -1;
}

f64 Node::q() const {
    return terminal_state.is_none() ? sum_of_scores / static_cast<f64>(num_visits) : terminal_state.score();
}

} // namespace search
