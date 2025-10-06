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

f64 Node::q_variance() const {
    const auto avg = sum_of_scores / static_cast<f64>(num_visits);
    const auto avg_squares = sum_of_scores_squared / static_cast<f64>(num_visits);
    return avg_squares - avg * avg;
}

} // namespace search
