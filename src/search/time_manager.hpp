#ifndef TIME_MANAGER_HPP
#define TIME_MANAGER_HPP

#include "../util/types.hpp"
#include <chrono>
#include <limits>
#include <optional>

namespace search {

using TimePoint = std::chrono::high_resolution_clock::time_point;

struct TimeSettings {
    std::array<i64, 2> time_left_per_side = {std::numeric_limits<i64>::max(), std::numeric_limits<i64>::max()};
    std::array<i64, 2> increment_per_side = {std::numeric_limits<i64>::max(), std::numeric_limits<i64>::max()};
    i32 max_depth = std::numeric_limits<i32>::max();
};

class TimeManager {
  public:
    TimeManager() = default;
    ~TimeManager() = default;

    void start_tracking(const TimeSettings &settings);

    [[nodiscard]] bool times_up(Color color, i32 depth) const;

    [[nodiscard]] u64 time_elapsed() const;

  private:
    TimeSettings settings_;
    TimePoint start_time_;
};

} // namespace search

#endif // TIME_MANAGER_HPP
