#include "node.hpp"

namespace search {

bool Node::terminal() const {
    return terminal_state != TerminalState::NONE;
}

bool Node::expanded() const {
    return first_child_idx != -1;
}

} // namespace search