#include "time_manager.hpp"
#include <chrono>
#include <iostream>
#include <optional>

namespace search {

std::optional<f64> kld_gain(const util::StaticVector<u32, MAX_MOVES> &new_visit_dist,
                            const util::StaticVector<u32, MAX_MOVES> &old_visit_dist) {
    u64 new_parent_visits = 0;
    for (const auto v : new_visit_dist) {
        new_parent_visits += v;
    }

    u64 old_parent_visits = 0;
    for (const auto v : old_visit_dist) {
        old_parent_visits += v;
    }

    if (old_parent_visits == 0) {
        return std::nullopt;
    }

    f64 kld_gain = 0.0;
    for (usize i = 0; i < new_visit_dist.size() && i < old_visit_dist.size(); ++i) {
        const auto new_visits = new_visit_dist[i];
        const auto old_visits = old_visit_dist[i];

        if (old_visits == 0) {
            return std::nullopt;
        }

        const f64 q = static_cast<f64>(new_visits) / static_cast<f64>(new_parent_visits);
        const f64 p = static_cast<f64>(old_visits) / static_cast<f64>(old_parent_visits);

        kld_gain += p * std::log(p / q);
    }

    if (new_parent_visits <= old_parent_visits) {
        return std::nullopt;
    }

    return kld_gain / static_cast<f64>(new_parent_visits - old_parent_visits);
}

void TimeManager::start_tracking(const TimeSettings &settings) {
    start_time_ = std::chrono::high_resolution_clock::now();
    settings_ = settings;
}

bool TimeManager::times_up(const GameTree &tree, u64 iterations, Color color, i32 depth,
                           const util::StaticVector<u32, MAX_MOVES> &old_visit_dist,
                           const util::StaticVector<u32, MAX_MOVES> &new_visit_dist) const {

    if (iterations % 512 == 0) {
        auto get_elapsed = [&] {
            const auto now = std::chrono::high_resolution_clock::now();
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_);
            return elapsed.count();
        };

        if (get_elapsed() >= settings_.movetime) {
            return true;
        }

        if (settings_.time_left_per_side[color] != std::numeric_limits<i64>::max()) {
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

            if (get_elapsed() > time_to_search) {
                return true;
            }
        }
    }

    if (settings_.min_kld_gain > 0) {
        const auto gain = kld_gain(new_visit_dist, old_visit_dist);
        if (gain && *gain < settings_.min_kld_gain) {
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
