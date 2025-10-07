#include "history.hpp"
#include "../util/math.hpp"
#include <algorithm>

namespace search {

[[nodiscard]] i16 scale_bonus(i16 score, i16 bonus) {
    return bonus - score * std::abs(bonus) / 8192;
}

void History::Entry::update(f64 score) {
    score = std::clamp(score, 0.001, 0.999);
    value += scale_bonus(value, static_cast<i32>(util::math::inverse_sigmoid(score)));
}

void History::clear() {
    table_ = {};
}

History::Entry &History::entry(const BoardState &state, Move move) {
    return table_[state.side_to_move][move.from()][move.to()];
}

} // namespace search