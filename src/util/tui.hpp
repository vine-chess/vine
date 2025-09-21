#ifndef TUI_HPP
#define TUI_HPP

#include "types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <ostream>

namespace util {

namespace tui {

using Color = std::array<i32, 3>;

inline void set_color(std::ostream &out, Color c) {
    const auto [r, g, b] = c;

    out << "\u001b[38;2;" << r << ";" << g << ";" << b << "m";
}

inline void reset_color(std::ostream &out) {
    out << "\u001b[0m";
}

constexpr Color get_score_color(f64 score) {
    const auto smooth_step = [](const f64 x) { return std::clamp(3.0 * x * x - 2.0 * x * x * x, 0.0, 1.0); };
    const auto t = smooth_step(score);

    const auto high_color = std::array{80, 220, 80};
    const auto mid_color = std::array{215, 215, 80};
    const auto low_color = std::array{230, 80, 80};

    if (t < 0.5) {
        const f64 t2 = t * 2.0;
        return std::array{
            std::clamp<i32>(std::round(std::lerp(low_color[0], mid_color[0], t2)), 0, 255),
            std::clamp<i32>(std::round(std::lerp(low_color[1], mid_color[1], t2)), 0, 255),
            std::clamp<i32>(std::round(std::lerp(low_color[2], mid_color[2], t2)), 0, 255),
        };
    } else {
        const f64 t2 = (t - 0.5) * 2.0;
        return std::array{
            std::clamp<i32>(std::round(std::lerp(mid_color[0], high_color[0], t2)), 0, 255),
            std::clamp<i32>(std::round(std::lerp(mid_color[1], high_color[1], t2)), 0, 255),
            std::clamp<i32>(std::round(std::lerp(mid_color[2], high_color[2], t2)), 0, 255),
        };
    }
}

} // namespace tui

} // namespace util

#endif // TUI_HPP
