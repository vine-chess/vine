#ifndef VALUE_CORRECTION_HPP
#define VALUE_CORRECTION_HPP

#include "../chess/board.hpp"
#include "../util/multi_array.hpp"
#include "../util/types.hpp"

namespace search {

class ValueCorrection {
  public:
    ValueCorrection() = default;

    void clear();

    void save(const BoardState &state, f64 q, u32 num_visits);
    [[nodiscard]] f64 correct(const BoardState &state, f64 value) const;

  private:
    [[nodiscard]] usize index(const BoardState &state) const;

    MultiArray<f64, 16384, 2> correction_;
};

} // namespace search

#endif // VALUE_CORRECTION_HPP