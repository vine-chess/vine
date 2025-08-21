#include "openings.hpp"
#include "../chess/move_gen.hpp"
#include "../search/searcher.hpp"

namespace datagen {

BoardState generate_opening(usize random_moves) {
    Board board(STARTPOS_FEN);
    search::Searcher searcher;
    bool success;
    do {
        board = Board(STARTPOS_FEN);
        success = true;

        for (usize ply = 0; ply < random_moves; ++ply) {
            MoveList moves;
            generate_moves(board.state(), moves);

            if (moves.empty() || board.is_fifty_move_draw() || board.has_threefold_repetition()) {
                success = false;
                break;
            }

            board.make_move(moves[rng::next_u64(0, moves.size() - 1)]);
        }

        const bool is_opening_valid = [&]() {
            // Opening ends in a terminal state
            MoveList moves;
            generate_moves(board.state(), moves);
            if (moves.empty() || board.is_fifty_move_draw() || board.has_threefold_repetition()) {
                return false;
            }

            // Position is too imbalanced
            searcher.go(board, {.max_depth = 5, .max_iters = 1000});
            const auto cp_score =
                static_cast<i32>(std::round(-400.0 * std::log(1.0 / searcher.game_tree().root().q() - 1.0)));
            if (std::abs(cp_score) >= 300) {
                return false;
            }

            return true;
        };

        success = is_opening_valid();
    } while (!success);

    return board.state();
}

} // namespace datagen
