#include "../eval/value_network.hpp"
#include "../util/math.hpp"
#include "policy_history.hpp"
#include <algorithm>

namespace search {

[[nodiscard]] inline i16 scale_bonus(i16 score, i16 bonus) {
    return bonus - score * std::abs(bonus) / 8192;
}

void PolicyHistory::Entry::update(f64 score) {
    score = std::clamp(score, 0.001, 0.999);
    value += scale_bonus(value, static_cast<i32>(network::value::EVAL_SCALE * util::math::inverse_sigmoid(score)));
}

void PolicyHistory::clear() {
    table_ = {};
}

PolicyHistory::Entry &PolicyHistory::entry(const BoardState &state, Move move) {
    return table_[state.side_to_move][move.from()][move.to()];
}

} // namespace search