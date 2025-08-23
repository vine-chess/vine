#include "openings.hpp"
#include "../chess/move_gen.hpp"
#include "../search/searcher.hpp"

namespace datagen {

// https://github.com/official-monty/Monty/blob/0ae41b52509a04519e3cc0d8837323efa56803e7/src/tree.rs#L400-L437
Move pick_move_temperature(search::GameTree const &tree, f64 temperature) {
    const auto root = tree.root();

    std::vector<f64> distr(root.num_children, 0.0);

    f64 total = 0;
    for (usize i = 0; i < root.num_children; ++i) {
        const auto child = tree.node_at(root.first_child_idx + i);
        distr[i] = std::pow<f64>(child.num_visits, temperature);
        total += distr[i];
    }

    f64 random_choice = rng::next_double();
    f64 sum = 0;
    for (usize i = 0; i < root.num_children; ++i) {
        const auto child = tree.node_at(root.first_child_idx + i);
        sum += distr[i];

        if (sum / total > random_choice) {
            return child.move;
        }
    }
    return tree.node_at(root.first_child_idx + root.num_children - 1).move;
}

BoardState generate_opening(usize random_moves) {
    search::Searcher searcher;
    searcher.set_hash_size(4);
    searcher.set_verbosity(search::Verbosity::NONE);

    Board board(STARTPOS_FEN);
    bool success;
    do {
        board = Board(STARTPOS_FEN);
        success = true;

        f64 temperature = 1.0;
        for (usize ply = 0; ply < random_moves; ++ply) {
            MoveList moves;
            generate_moves(board.state(), moves);

            if (moves.empty() || board.is_fifty_move_draw() || board.has_threefold_repetition()) {
                success = false;
                break;
            }

            searcher.go(board, {.max_depth = 5, .max_iters = 1000});

            // Position is too imbalanced
            const auto cp_score =
                static_cast<i32>(std::round(-400.0 * std::log(1.0 / searcher.game_tree().root().q() - 1.0)));
            if (std::abs(cp_score) >= 300) {
                success = false;
                break;
            }

            temperature *= 0.9;
            board.make_move(pick_move_temperature(searcher.game_tree(), temperature));
        }

        const auto is_opening_valid = [&]() {
            // Opening ends in a terminal state
            MoveList moves;
            generate_moves(board.state(), moves);
            if (moves.empty() || board.is_fifty_move_draw() || board.has_threefold_repetition()) {
                return false;
            }

            return true;
        };

        success = is_opening_valid();
    } while (!success);

    return board.state();
}

} // namespace datagen
