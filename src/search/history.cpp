#include "history.hpp"
#include "../eval/value_network.hpp"
#include "../util/math.hpp"
#include <algorithm>

namespace search {

[[nodiscard]] i16 scale_bonus(i16 score, i16 bonus) {
    return bonus - score * std::abs(bonus) / 8192;
}

void History::Entry::update(f64 score, f64 q) {
    score = std::clamp(score, 0.001, 0.999);
    const auto cp_score = util::math::inverse_sigmoid(score), cp_q = util::math::inverse_sigmoid(q);
    const auto bonus = static_cast<i32>(network::value::EVAL_SCALE * cp_score * (128 + std::abs(cp_q - cp_score)) / 128);
    value += scale_bonus(value, bonus);
}

void History::clear() {
    table_ = {};
}

History::Entry &History::entry(const BoardState &state, Move move) {
    return table_[state.side_to_move][move.from()][move.to()];
}

} // namespace search