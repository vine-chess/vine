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

    u64 nodes = 0;
    for (const auto move : moves) {
        board.make_move(move);
        nodes += perft(board, depth - 1);
        board.undo_move();
    }

    return nodes;
}

u64 perft_print(Board &board, i32 depth, std::ostream &out) {
    if (depth == 0) {
        return 1;
    }

    MoveList moves;
    generate_moves(board.state(), moves);

    u64 nodes = 0;
    for (const auto move : moves) {
        board.make_move(move);
        // if (move.to() == Square::D3) {
        //     const auto parsed = Board("8/8/6k1/8/8/3b4/2B5/K7 w - - 1 2");
        //     if (parsed.state().pawns() != board.state().pawns()) {
        //         out << "pawns mismatch\n";
        //     }
        //     if (parsed.state().knights() != board.state().knights()) {
        //         out << "knights mismatch\n";
        //     }
        //     if (parsed.state().bishops() != board.state().bishops()) {
        //         out << "bishops mismatch\n";
        //     }
        //     if (parsed.state().rooks() != board.state().rooks()) {
        //         out << "rooks mismatch\n";
        //     }
        //     if (parsed.state().queens() != board.state().queens()) {
        //         out << "queens mismatch\n";
        //     }
        //     if (parsed.state().kings() != board.state().kings()) {
        //         out << "kings mismatch\n";
        //     }
        //     if (parsed.state().checkers != board.state().checkers) {
        //         out << "checkers mismatch\n";
        //     }
        //     if (parsed.state().castle_rights != board.state().castle_rights) {
        //         out << "castle_rights mismatch\n";
        //     }
        //     if (parsed.state().side_to_move != board.state().side_to_move) {
        //         out << "side_to_move mismatch\n";
        //     }
        //     if (parsed.state().en_passant_sq != board.state().en_passant_sq) {
        //         out << "en_passant_sq mismatch\n";
        //     }
        //     if (parsed.state().fifty_moves_clock != board.state().fifty_moves_clock) {
        //         out << parsed.state().fifty_moves_clock << ' ' << board.state().fifty_moves_clock << '\n';
        //         out << "fifty_moves_clock mismatch\n";
        //     }
        //     if (parsed.state().hash_key != board.state().hash_key) {
        //         out << "hash_key mismatch\n";
        //     }
        //     if (parsed.state().ortho_pins != board.state().ortho_pins) {
        //         out << "ortho_pins mismatch\n";
        //     }
        //     if (parsed.state().diag_pins != board.state().diag_pins) {
        //         out << "diag_pins mismatch\n";
        //     }
        // }
        const auto child_nodes = perft(board, depth - 1);
        board.undo_move();

        out << move << ": " << child_nodes << '\n';

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
