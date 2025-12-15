#include "../eval/value_network.hpp"
#include "../util/math.hpp"
#include "value_history.hpp"
#include <algorithm>
#include <iostream>

namespace search {

[[nodiscard]] inline i16 scale_bonus(i16 score, i16 bonus) {
    return bonus - score * std::abs(bonus) / 1024;
}

void ValueHistory::Entry::update(f64 q, f64 score, u16 num_visits) {
    score = std::clamp(score, 0.001, 0.999);
    q = std::clamp(q, 0.001, 0.999);

    const i32 q_cp = static_cast<i32>(network::value::EVAL_SCALE * util::math::inverse_sigmoid(q));
    const i32 score_cp = static_cast<i32>(network::value::EVAL_SCALE * util::math::inverse_sigmoid(score));

    constexpr i32 min_div = 8;
    constexpr i32 max_div = 16;
    constexpr u16 max_visits = 1000;

    const i32 divisor = min_div
                  + (max_div - min_div) * std::min(num_visits, max_visits) / max_visits;

    const i32 bonus = std::clamp((q_cp - score_cp) / divisor, -256, 256);
    value += scale_bonus(value, bonus);
}

void ValueHistory::clear() {
    table_ = {};
}

[[nodiscard]] i32 ValueHistory::correct_cp(const BoardState &state, i32 score_cp) const {
    const auto history_entry = entry(state);
    return score_cp + history_entry.value / 32;
}

ValueHistory::Entry &ValueHistory::entry(const BoardState &state) {
    return table_[state.pawn_hash_key & 16383][state.side_to_move];
}

const ValueHistory::Entry &ValueHistory::entry(const BoardState &state) const {
    return table_[state.pawn_hash_key & 16383][state.side_to_move];
}

} // namespace search