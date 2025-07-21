#pragma once

#include "../util/static_vector.hpp"
#include "board_state.hpp"
#include <string_view>

constexpr std::string_view STARTPOS_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

class Board {
  public:
    Board(std::string_view fen);
    Board() = default;

    [[nodiscard]] BoardState &state();
    [[nodiscard]] const BoardState &state() const;

    [[nodiscard]] Move create_move(std::string_view uci_move) const;

    void make_move(Move move);
    void undo_move();

    friend std::ostream &operator<<(std::ostream &os, const Board &board);

  private:
    BoardState state_;
};
