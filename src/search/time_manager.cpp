#include "time_manager.hpp"
#include <chrono>
#include <iostream>

namespace search {

void TimeManager::start_tracking(const TimeSettings &settings) {
    start_time_ = std::chrono::high_resolution_clock::now();
    settings_ = settings;
}

bool TimeManager::times_up(const GameTree &tree, u64 iterations, Color color, i32 depth) const {
    if (iterations % 512 == 0 && settings_.time_left_per_side[color] != std::numeric_limits<i64>::max()) {
        auto time_to_search = settings_.time_left_per_side[color] / 20 + settings_.increment_per_side[color] / 2;

        // Reduce the time we need to search based on what % of visits the best move contributes to
        if (iterations >= 1024) {
            const auto get_child_score = [&](NodeIndex child_idx) {
                const f64 MATE_SCORE = 1000.0;
                const Node &child = tree.node_at(child_idx);
                switch (child.terminal_state.flag()) {
                case TerminalState::Flag::WIN:
                    return MATE_SCORE - child.terminal_state.distance_to_terminal();
                case TerminalState::Flag::LOSS:
                    return -MATE_SCORE + child.terminal_state.distance_to_terminal();
                default:
                    return child.q();
                }
            };

            NodeIndex best_child_idx = tree.root().first_child_idx;
            for (u16 i = 0; i < tree.root().num_children; ++i) {
                if (get_child_score(tree.root().first_child_idx + i) < get_child_score(best_child_idx)) {
                    best_child_idx = tree.root().first_child_idx + i;
                }
            }

            time_to_search *= 1.0 - (static_cast<f64>(tree.node_at(best_child_idx).num_visits) /
                                     static_cast<f64>(tree.root().num_visits) * 0.25);
        }

        const auto now = std::chrono::high_resolution_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_);
        if (elapsed.count() > time_to_search) {
            return true;
        }
    }

    if (depth >= settings_.max_depth) {
        return true;
    }

    if (iterations >= settings_.max_iters) {
        return true;
    }

    return false;
}

u64 TimeManager::time_elapsed() const {
    const auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_).count();
}

} // namespace search
