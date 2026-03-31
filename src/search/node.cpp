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
    return num_children != 0;
}

f64 Node::q() const {
    return sum_of_scores / static_cast<f64>(num_visits);
}

bool NodeReference::terminal() const {
    return (info.terminal_state.is_loss() || info.terminal_state.is_draw() || info.terminal_state.is_win()) &&
           info.terminal_state.distance_to_terminal() == 0;
}

bool NodeReference::visited() const {
    return num_visits > 0;
}

bool NodeReference::expanded() const {
    return info.num_children != 0;
}

f64 NodeReference::q() const {
    return sum_of_scores / static_cast<f64>(num_visits);
}
} // namespace search
