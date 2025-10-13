#include "history.hpp"
#include "../eval/value_network.hpp"
#include "../util/math.hpp"
#include <algorithm>

namespace search {

[[nodiscard]] i16 scale_bonus(i16 score, i16 bonus, u32 gravity) {
    return bonus - score * std::abs(bonus) / gravity;
}

void History::update(const BoardState &state, Move move, f64 score) {
    score = std::clamp(score, 0.001, 0.999);
    const auto bonus = static_cast<i32>(network::value::EVAL_SCALE * util::math::inverse_sigmoid(score));
    const auto threats = state.threats_by(~state.side_to_move);

    auto &table_entry = butterfly_table_[state.side_to_move][move.from()][move.to()];

    i16 &entry_score = table_entry.score;
    entry_score += scale_bonus(entry_score, bonus, 6144);

    i16 &bucket = table_entry.threat_buckets[threats.is_set(move.from())][threats.is_set(move.to())];
    bucket += scale_bonus(bucket, bonus, 2048);
}

void History::clear() {
    butterfly_table_ = {};
}

i16 History::entry(const BoardState &state, Move move) {
    const auto &table_entry = butterfly_table_[state.side_to_move][move.from()][move.to()];
    const auto threats = state.threats_by(~state.side_to_move);
    return table_entry.score + table_entry.threat_buckets[threats.is_set(move.from())][threats.is_set(move.to())];
}

} // namespace search