#include "openings.hpp"
#include "../chess/move_gen.hpp"
#include "../eval/value_network.hpp"
#include "../search/searcher.hpp"
#include "../util/math.hpp"
#include <iostream>
#include <limits>
#include <optional>
#include <string_view>

namespace datagen {

// https://github.com/official-monty/Monty/blob/0ae41b52509a04519e3cc0d8837323efa56803e7/src/tree.rs#L400-L437
search::NodeIndex pick_node_temperature(search::GameTree const &tree, f64 temperature, bool downsample_captures,
                           std::optional<i32> score_limit, std::optional<f64> visits_limit) {
    const auto root = tree.root();

    std::vector<f64> distr(root.num_children, 0.0);
    std::vector<i32> scores(root.num_children, 0);

    i32 highest_score = std::numeric_limits<i32>::min();
    u32 highest_visits = 0;
    for (usize i = 0; i < root.num_children; ++i) {
        const auto child = tree.node_at(root.first_child_idx + i);
        const auto score = -util::math::inverse_sigmoid(child.q()) * network::value::EVAL_SCALE;
        scores[i] = score;
        if (score > highest_score) {
            highest_score = score;
            highest_visits = child.num_visits;
        }
    }

    f64 total = 0;

    for (usize i = 0; i < root.num_children; ++i) {
        const auto child = tree.node_at(root.first_child_idx + i);
        if (score_limit.has_value() && scores[i] + *score_limit < highest_score) {
            continue;
        }
        if (visits_limit.has_value() && child.num_visits < highest_visits * *visits_limit) {
            continue;
        }
        // std::cout << child.move.to_string() << ' ';
        distr[i] = std::pow<f64>(child.num_visits, 1.0 / temperature);
        if (downsample_captures && child.move.is_capture()) {
            distr[i] /= 8;
        }
        total += distr[i];
    }

    f64 random_choice = rng::next_f64();
    f64 sum = 0;
    for (usize i = 0; i < root.num_children; ++i) {
        const auto child = tree.node_at(root.first_child_idx + i);
        sum += distr[i];

        if (sum / total > random_choice) {
            return root.first_child_idx + i;
        }
    }
    return root.first_child_idx + root.num_children - 1;
}

BoardState generate_opening(std::span<const std::string> opening_fens, const usize random_moves,
                            const f64 initial_temperature, const f64 gamma) {
    thread_local search::Searcher searcher;
    searcher.set_hash_size(4);
    searcher.set_verbosity(search::Verbosity::NONE);

    Board board;
    bool success;
    do {
        board = Board(opening_fens[rng::next_u64(0, opening_fens.size() - 1)]);
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
            const auto picked_node_idx = pick_node_temperature(searcher.game_tree(), temperature, true);
            board.make_move(searcher.game_tree().node_at(picked_node_idx).move);
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
