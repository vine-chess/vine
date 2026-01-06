#include "openings.hpp"
#include "../chess/move_gen.hpp"
#include "../eval/value_network.hpp"
#include "../search/searcher.hpp"
#include "../util/math.hpp"
#include <iostream>
#include <optional>
#include <string_view>

namespace datagen {

// https://github.com/official-monty/Monty/blob/0ae41b52509a04519e3cc0d8837323efa56803e7/src/tree.rs#L400-L437
Move pick_move_temperature(search::GameTree &tree, f64 temperature) {
    const auto root = tree.root();

    std::vector<f64> distr(root.info.num_children, 0.0);

    f64 total = 0;
    for (usize i = 0; i < root.info.num_children; ++i) {
        const auto child = tree.node_at(root.info.first_child_idx + i);
        distr[i] = std::pow<f64>(child.num_visits, 1.0 / temperature);
        total += distr[i];
    }

    f64 random_choice = rng::next_f64();
    f64 sum = 0;
    for (usize i = 0; i < root.info.num_children; ++i) {
        auto child = tree.node_at(root.info.first_child_idx + i);
        sum += distr[i];

        if (sum / total > random_choice) {
            return child.info.move;
        }
    }
    return tree.node_at(root.info.first_child_idx + root.info.num_children - 1).info.move;
}

BoardState generate_opening(const std::unique_ptr<OpeningBook>& book, const usize random_moves, const f64 initial_temperature,
                            const f64 gamma) {
    thread_local search::Searcher searcher;
    searcher.set_hash_size(4);
    searcher.set_verbosity(search::Verbosity::NONE);

    Board board;
    bool success;
    do {
        if (book) {
            board = Board(book->get(rng::next_u64(0, book->size() - 1)));
        } else {
            board = Board(STARTPOS_FEN);
        }
        success = true;

        f64 temperature = initial_temperature;
        for (usize ply = 0; ply < random_moves; ++ply) {
            MoveList moves;
            generate_moves(board.state(), moves);

            if (moves.empty() || board.is_draw()) {
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
            if (moves.empty() || board.is_draw()) {
                return false;
            }

            // Position is too imbalanced
            searcher.go(board, {.max_depth = 5, .max_iters = 1000});
            const auto cp_score = static_cast<i32>(
                std::round(network::value::EVAL_SCALE * util::math::inverse_sigmoid(searcher.game_tree().root().q())));
            if (std::abs(cp_score) >= 10000) {
                return false;
            }

            return true;
        };

        success = is_opening_valid();
    } while (!success);

    return board.state();
}

} // namespace datagen
