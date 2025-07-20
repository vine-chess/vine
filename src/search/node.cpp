#include "node.hpp"

namespace search {

bool Node::terminal() const {
    return terminal_state != TerminalState::NONE;
}

bool Node::expanded() const {
    return num_visits != 0;
}

} // namespace search