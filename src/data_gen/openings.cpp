#include "openings.hpp"
#include "../chess/move_gen.hpp"
#include "../search/searcher.hpp"
#include <iostream>
#include <ostream>
#include <string_view>

namespace datagen {

// https://github.com/official-monty/Monty/blob/0ae41b52509a04519e3cc0d8837323efa56803e7/src/tree.rs#L400-L437
Move pick_move_temperature(search::GameTree const &tree, f64 temperature) {
    const auto root = tree.root();

    std::vector<f64> distr(root.num_children, 0.0);

    f64 total = 0;
    for (usize i = 0; i < root.num_children; ++i) {
        const auto child = tree.node_at(root.first_child_idx + i);
        distr[i] = std::pow<f64>(child.num_visits, 1.0 / temperature);
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

BoardState generate_opening(std::string_view initial_fen, const usize random_moves, const f64 initial_temperature,
                            const f64 gamma) {
    thread_local search::Searcher searcher;
    searcher.set_hash_size(4);
    searcher.set_verbosity(search::Verbosity::NONE);

    Board board(initial_fen);
    bool success;
    do {
        board = Board(initial_fen);
        success = true;

        f64 temperature = initial_temperature;
        for (usize ply = 0; ply < random_moves; ++ply) {
            MoveList moves;
            generate_moves(board.state(), moves);

            if (moves.empty() || board.is_fifty_move_draw() || board.has_threefold_repetition()) {
                success = false;
                break;
            }

            searcher.go(board, {.max_depth = 5, .max_iters = 1000});

            temperature *= gamma;
            board.make_move(pick_move_temperature(searcher.game_tree(), temperature));
        }

        const auto is_opening_valid = [&]() {
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
