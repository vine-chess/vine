#include "perft.hpp"
#include "../chess/move_gen.hpp"
#include <iostream>

namespace tests {

u64 perft(Board &board, i32 depth) {
    if (depth == 0) {
        return 1;
    }

    MoveList moves;
    generate_moves(board.state(), moves);

    if (depth == 1) {
        return moves.size();
    }

    u64 nodes = 0;
    for (const auto move : moves) {
        board.make_move(move);
        nodes += perft(board, depth - 1);
        board.undo_move();
    }

    return nodes;
}

u64 perft_print(Board &board, i32 depth, std::ostream &out) {
    MoveList moves;
    generate_moves(board.state(), moves);

    u64 nodes = 0;
    for (const auto move : moves) {
        board.make_move(move);
        const auto child_nodes = perft(board, depth - 1);
        board.undo_move();

        out << move << ": " << child_nodes << std::endl;
        nodes += child_nodes;
    }

    return nodes;
}

void run_perft_tests(std::ostream &out) {
    for (const auto &batch : PERFT_BATCHES) {
        Board board(batch.fen);
        out << "FEN: " << batch.fen << "\n";

        for (const auto &[depth, expected] : batch.depths) {
            u64 result = perft(board, depth);
            out << "Depth: " << depth << " → Got: " << result << ", Expected: " << expected;
            if (result == expected) {
                out << " ✅" << std::endl;
            } else {
                out << " ❌" << std::endl;
            }
        }
    }
}

} // namespace tests
