#ifndef TIME_MANAGER_HPP
#define TIME_MANAGER_HPP

#include "../chess/move_gen.hpp"
#include "../util/static_vector.hpp"
#include "../util/types.hpp"
#include "game_tree.hpp"
#include <chrono>
#include <limits>

namespace search {

using TimePoint = std::chrono::high_resolution_clock::time_point;

struct TimeSettings {
    std::array<i64, 2> time_left_per_side = {std::numeric_limits<i64>::max(), std::numeric_limits<i64>::max()};
    std::array<i64, 2> increment_per_side = {0, 0};
    i32 max_depth = std::numeric_limits<i32>::max();
    u64 max_iters = std::numeric_limits<u64>::max();
    f64 min_kld_gain = 0;
};

class TimeManager {
  public:
    TimeManager() = default;
    ~TimeManager() = default;

    void start_tracking(const TimeSettings &settings);

    [[nodiscard]] bool times_up(const GameTree &tree, u64 iterations, Color color, i32 depth,
                                const util::StaticVector<u32, MAX_MOVES> &old_visit_dist,
                                const util::StaticVector<u32, MAX_MOVES> &new_visit_dist) const;

    [[nodiscard]] u64 time_elapsed() const;

  private:
    TimeSettings settings_;
    TimePoint start_time_;
};

} // namespace search

#endif // TIME_MANAGER_HPP
