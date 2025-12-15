#ifndef POLICY_HISTORY_HPP
#define POLICY_HISTORY_HPP

#include "../chess/board.hpp"
#include "../util/multi_array.hpp"
#include "../util/types.hpp"

namespace search {

class PolicyHistory {
  public:
    struct Entry {
        i16 value;

        void update(f64 score);
    };

    void clear();

    [[nodiscard]] Entry &entry(const BoardState &state, Move move);

  private:
    util::MultiArray<Entry, 2, 64, 64> table_{};
};

} // namespace search

#endif // POLICY_HISTORY_HPP