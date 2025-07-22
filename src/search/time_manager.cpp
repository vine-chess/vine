#include "time_manager.hpp"
#include <chrono>

namespace search {

void TimeManager::start_tracking(const TimeSettings &settings) {
    start_time_ = std::chrono::high_resolution_clock::now();
    settings_ = settings;
}

bool TimeManager::times_up(u64 iterations, Color color, i32 depth) const {
    if (iterations % 512 == 0) {
        const auto now = std::chrono::high_resolution_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_);
        if (elapsed.count() > settings_.time_left_per_side[color] / 20 + settings_.increment_per_side[color] / 2) {
            return true;
        }
    }

    if (depth >= settings_.max_depth) {
        return true;
    }

    return false;
}

u64 TimeManager::time_elapsed() const {
    const auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_).count();
}

} // namespace search
