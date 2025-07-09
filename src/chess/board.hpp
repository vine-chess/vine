#pragma once

#include "../util/static_vector.hpp"
#include "../util/types.hpp"

#include "bitboard.hpp"
#include "board_state.hpp"
#include "castle_rights.hpp"

#include <string>
#include <string_view>

class Board {
  public:
    Board(std::string_view fen);
    Board();

    void print();

  private:
    BoardState state_;
    util::StaticVector<BoardState, 2048> state_history_;
};
