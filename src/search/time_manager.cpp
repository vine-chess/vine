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
            const auto& root = tree.root();
            const u16 K = root.num_children;
            const u64 N = root.num_visits;
            if (K > 1 && N > 0) {
                double H = 0.0;
                for (u16 i = 0; i < K; ++i) {
                    const auto& child = tree.node_at(root.first_child_idx + i);
                    if (child.num_visits == 0) {
                        continue;
                    }
                    const double p = static_cast<double>(child.num_visits) / static_cast<double>(N);
                    H += -p * std::log(p);
                }

                const double Hmax = std::log(static_cast<double>(K));
                const double h = (Hmax > 0.0) ? (H / Hmax) : 1.0; // normalize

                // Reduction when very peaked (h ≈ 0), no reduction when flat (h ≈ 1).
                const double reduction = (1.0 - h) * 0.25;
                time_to_search *= (1.0 - reduction);
            }
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
