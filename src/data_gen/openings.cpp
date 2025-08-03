#include "openings.hpp"
#include "../chess/move_gen.hpp"
#include <algorithm>

namespace datagen {

BoardState generate_opening(usize seed, usize random_moves) {
    rng::seed_generator(seed);

    Board board(STARTPOS_FEN);
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
    } while (!success);

    return board.state();
}

} // namespace datagen
