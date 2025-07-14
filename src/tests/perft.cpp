#include "perft.hpp"
#include "../chess/move_gen.hpp"
#include <iostream>

namespace tests {

u64 perft(Board &board, i32 depth) {
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

    if (depth == 1) {
        return moves.size();
    }

    u64 nodes = 0;
    for (const auto move : moves) {
        board.make_move(move);
        const auto child_nodes = perft(board, depth - 1);
        board.undo_move();
        out << move << ": " << child_nodes << '\n';

        nodes += child_nodes;
    }

    return nodes;
}

void run_perft_tests(const std::vector<PerftTestCase> &tests, std::ostream &out) {
    for (const auto &[fen, depth, expected] : tests) {
        Board board(fen);
        u64 result = perft(board, depth);

        out << "FEN: " << fen << "\n";
        out << "Depth: " << depth << " → Got: " << result << ", Expected: " << expected;

        if (result == expected) {
            out << " ✅\n";
        } else {
            out << " ❌\n";
        }
    }
}

} // namespace tests