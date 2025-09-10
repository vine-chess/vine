#pragma once

#include "../util/static_vector.hpp"
#include "board_state.hpp"
#include <string_view>

constexpr std::string_view STARTPOS_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

class Board {
  public:
    using History = std::vector<BoardState>;

    Board(std::string_view fen);
    Board(const BoardState &board_state);
    Board() = default;

    [[nodiscard]] BoardState &state();
    [[nodiscard]] const BoardState &state() const;
    [[nodiscard]] const BoardState &prev_state() const;

    [[nodiscard]] bool has_threefold_repetition() const;
    [[nodiscard]] bool is_fifty_move_draw() const;

    [[nodiscard]] Move create_move(std::string_view uci_move) const;

    [[nodiscard]] HashKey predict_hash_key(Move move) const;

    void make_move(Move move);
    void undo_move();

    void undo_n_moves(usize n);

    friend std::ostream &operator<<(std::ostream &os, const Board &board);

    [[nodiscard]] History &history();
    [[nodiscard]] const History &history() const;

  private:
    History history_;
};
