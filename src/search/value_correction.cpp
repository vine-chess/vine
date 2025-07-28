#include "value_correction.hpp"
#include <algorithm>
#include <iostream>

namespace search {

f64 scale_bonus(f64 correction, f64 bonus) {
    return bonus - correction * std::abs(bonus);
}

void ValueCorrection::clear() {
    correction_.fill(0.0);
}

void ValueCorrection::save(const BoardState &state, f64 q, u32 num_visits) {
    f64 &correction = correction_[index(state)][state.side_to_move];
    const f64 bonus = std::clamp(q - correction, -0.3, 0.3);
    correction += scale_bonus(correction, bonus);
}

[[nodiscard]] f64 ValueCorrection::correct(const BoardState &state, f64 value) const {
    return std::clamp(value + correction_[index(state)][state.side_to_move], 0.0, 1.0);
}

[[nodiscard]] usize ValueCorrection::index(const BoardState &state) const {
    return state.pawn_key % correction_.size();
}

}