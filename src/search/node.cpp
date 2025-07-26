#include "node.hpp"

namespace search {

bool Node::terminal() const {
    return terminal_state != TerminalState::none();
}

bool Node::visited() const {
    return num_visits > 0;
}

bool Node::expanded() const {
    return first_child_idx != -1;
}

f64 Node::q() const {
    return sum_of_scores / static_cast<f64>(num_visits);
}

} // namespace search
