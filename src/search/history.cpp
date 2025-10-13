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

    auto &table_entry = butterfly_table_[state.side_to_move][move.from()][move.to()];

    i16 &entry_score = table_entry.score;
    entry_score += scale_bonus(entry_score, bonus, 7168);

    i16 &bucket = table_entry.piece_buckets[state.get_piece_type(move.from())];
    bucket += scale_bonus(bucket, bonus, 1024);
}

void History::clear() {
    butterfly_table_ = {};
}

i16 History::entry(const BoardState &state, Move move) {
    const auto &table_entry = butterfly_table_[state.side_to_move][move.from()][move.to()];
    return table_entry.score + table_entry.piece_buckets[state.get_piece_type(move.from())];
}

} // namespace search