#ifndef HISTORY_HPP
#define HISTORY_HPP

#include "../chess/board.hpp"
#include "../util/multi_array.hpp"
#include "../util/types.hpp"

namespace search {

class History {
  public:
    void clear();

    void update(const BoardState &state, Move move, f64 score);

    [[nodiscard]] i16 entry(const BoardState &state, Move move);

  private:
    struct Entry {
      i16 score = 0;
      std::array<i16, 6> piece_buckets{};
    };

    util::MultiArray<Entry, 2, 64, 64> butterfly_table_{};
};

} // namespace search

#endif // HISTORY_HPP