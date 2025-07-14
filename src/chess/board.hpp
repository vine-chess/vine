#pragma once

#include "../util/static_vector.hpp"
#include "board_state.hpp"
#include <string_view>

class Board {
  public:
    Board(std::string_view fen);
    Board();

    [[nodiscard]] BoardState &state();
    [[nodiscard]] const BoardState &state() const;

    void make_move(Move move);
    void undo_move();

    friend std::ostream &operator<<(std::ostream &os, const Board &board);

  private:
    BoardState state_;
    util::StaticVector<BoardState, 2048> state_history_;
};