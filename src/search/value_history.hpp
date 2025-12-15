#ifndef VALUE_HISTORY_H
#define VALUE_HISTORY_H

#include "../chess/board.hpp"
#include "../util/multi_array.hpp"
#include "../util/types.hpp"

namespace search {

class ValueHistory {
  public:
    struct Entry {
        i16 value;

        void update(f64 q, f64 score, u16 num_visits);
    };

    void clear();

    [[nodiscard]] i32 correct_cp(const BoardState &state, i32 score_cp) const;

    [[nodiscard]] Entry &entry(const BoardState &state);

    [[nodiscard]] const Entry &entry(const BoardState &state) const;

  private:
    util::MultiArray<Entry, 16384, 2> table_{};
};

} // namespace search

#endif // VALUE_HISTORY_H
