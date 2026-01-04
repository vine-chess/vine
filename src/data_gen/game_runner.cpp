#include "game_runner.hpp"
#include "../chess/move_gen.hpp"
#include "../eval/value_network.hpp"
#include "../util/math.hpp"
#include "format/monty_format.hpp"
#include "format/viri_format.hpp"

#include <atomic>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <type_traits>
#include <unordered_set>

namespace datagen {

std::atomic_bool stop_flag;
void signal_handler([[maybe_unused]] i32 signum) {
    stop_flag = true;
}

std::atomic_size_t games_played = 0;
std::atomic_size_t positions_written = 0;

template <bool value = true>
void thread_loop(const Settings &settings, std::ostream &out_file, std::span<const std::string> opening_fens) {
    using DataWriter = std::conditional_t<value, ViriformatWriter, MontyFormatWriter>;
    auto writer = std::make_unique<DataWriter>(out_file);

    search::Searcher searcher;
    searcher.set_hash_size(settings.hash_size);
    searcher.set_verbosity(search::Verbosity::NONE);

    rng::seed_generator(std::random_device{}(), std::hash<std::thread::id>{}(std::this_thread::get_id()));

    const usize games_per_thread = settings.num_games / settings.num_threads;
    for (usize i = 0; i < games_per_thread && !stop_flag.load(std::memory_order_relaxed); i++) {
        Board board(generate_opening(opening_fens, settings.random_moves, settings.temperature, settings.gamma));
        writer->push_board_state(board.state());

        f64 game_result;
        u16 white_win_plies = 0, white_loss_plies = 0, draw_plies = 0;
        while (true) {
            searcher.go(board, settings.time_settings);

            const auto &game_tree = searcher.game_tree();
            const auto &root_node = game_tree.root();
            if (root_node.terminal()) {
                game_result = board.state().checkers != 0 ? board.state().side_to_move == Color::BLACK : 0.5;
                break;
            }

            search::NodeIndex best_child_idx = root_node.first_child_idx;
            for (usize j = 0; j < root_node.num_children; j++) {
                const auto &child = game_tree.node_at(root_node.first_child_idx + j);
                if (child.q() < game_tree.node_at(best_child_idx).q()) {
                    best_child_idx = root_node.first_child_idx + j;
                }
            }

            const auto &best_child = game_tree.node_at(best_child_idx);
            vine_assert(!best_child.move.is_null());

            const f64 score = 1.0 - best_child.q();

            // Adjudicate immediately if our move has a mate score
            if (best_child.terminal_state.is_win()) {
                game_result = static_cast<f64>(board.state().side_to_move == Color::BLACK);
            } else if (best_child.terminal_state.is_loss()) {
                game_result = static_cast<f64>(board.state().side_to_move == Color::WHITE);
            }
            // Otherwise adjudicate based on score agreement to a certain ply
            else {
                const f64 white_relative_cp =
                    400 * util::math::inverse_sigmoid(board.state().side_to_move == Color::WHITE ? score : 1.0 - score);
                if (white_relative_cp >= 2000) {
                    ++white_win_plies, white_loss_plies = draw_plies = 0;
                } else if (white_relative_cp <= -2000) {
                    ++white_loss_plies, white_win_plies = draw_plies = 0;
                } else if (std::abs(white_relative_cp) <= 30) {
                    ++draw_plies, white_win_plies = white_loss_plies = 0;
                }

                if (white_win_plies >= 5) {
                    game_result = 1.0;
                    break;
                } else if (white_loss_plies >= 5) {
                    game_result = 0.0;
                    break;
                } else if (draw_plies >= 10) {
                    game_result = 0.5;
                    break;
                }
            }

            if constexpr (value) {
                writer->push_move(best_child.move, score, board.state());
            } else {
                VisitsDistribution visits_dist;
                for (usize j = 0; j < root_node.num_children; j++) {
                    const auto &child = game_tree.node_at(root_node.first_child_idx + j);
                    visits_dist.emplace_back(writer->to_monty_move(child.move, board.state()), child.num_visits);
                }
                writer->push_move(best_child.move, score, visits_dist, board.state());
            }
            board.make_move(best_child.move);

            positions_written.fetch_add(1, std::memory_order_relaxed);

            if (board.is_draw()) {
                game_result = 0.5;
                break;
            }
        }

        writer->write_with_result(game_result);
        games_played.fetch_add(1, std::memory_order_relaxed);
    }
}

void run_games(Settings settings, std::ostream &out) {
    out << "starting datagen..." << std::endl;

    std::signal(SIGINT, signal_handler);

    std::vector<std::thread> threads;

    // Progress monitoring thread
    std::thread monitor([&out, &settings]() {
        usize last_games = 0;
        usize last_positions = 0;
        auto last_time = std::chrono::steady_clock::now();
        bool printed = false;

        while (!stop_flag.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(5));

            auto current_time = std::chrono::steady_clock::now();
            f64 elapsed_sec = std::chrono::duration<f64>(current_time - last_time).count();

            usize current_games = games_played.load();
            usize current_positions = positions_written.load();

            usize total_games = settings.num_games;
            f64 games_per_sec = (current_games - last_games) / elapsed_sec;
            f64 positions_per_sec = (current_positions - last_positions) / elapsed_sec;

            f64 remaining_games = total_games > current_games ? total_games - current_games : 0;
            f64 eta_sec = games_per_sec > 0.0 ? remaining_games / games_per_sec : 0.0;

            usize eta_min = static_cast<usize>(eta_sec) / 60;
            usize eta_hour = eta_min / 60;
            usize eta_rem_min = eta_min % 60;
            usize eta_rem_sec = static_cast<usize>(eta_sec) % 60;

            if (printed) {
                // Clear previous lines (5 lines)
                out << "\033[F\033[K\033[F\033[K\033[F\033[K\033[F\033[K\033[F\033[K";
            }

            out << "progress update:\n";
            out << "  games played      : " << current_games << " / " << total_games << '\n';
            out << "  positions written : " << current_positions << '\n';
            out << "  throughput        : " << games_per_sec << " games/s, " << positions_per_sec << " pos/s\n";
            out << "  eta               : " << eta_hour << "h " << eta_rem_min << "m " << eta_rem_sec << "s\n";

            last_games = current_games;
            last_positions = current_positions;
            last_time = current_time;

            if (stop_flag.load()) {
                break;
            }

            printed = true;
        }
    });

    std::vector<std::string> opening_fens;
    if (settings.book_path.empty()) {
        opening_fens.push_back(std::string(STARTPOS_FEN));
    } else {
        std::ifstream book{std::string(settings.book_path)};
        for (std::string opening; std::getline(book, opening);) {
            opening_fens.push_back(opening);
        }
    };

    std::ofstream final_output(settings.output_file, std::ios::binary);
    std::vector<char> big_buf(1 << 20);
    final_output.rdbuf()->pubsetbuf(big_buf.data(), big_buf.size());
    for (usize thread_id = 0; thread_id < settings.num_threads; ++thread_id) {

        threads.emplace_back([settings, &final_output, &out, &opening_fens]() {
            if (settings.mode == DatagenMode::value)
                thread_loop<true>(settings, final_output, opening_fens);
            else
                thread_loop<false>(settings, final_output, opening_fens);
        });
    }

    for (auto &thread : threads) {
        thread.join();
    }

    stop_flag = true;
    monitor.join();

    out << "\ndatagen complete:\n";
    out << "  total games played      : " << games_played.load() << '\n';
    out << "  total positions written : " << positions_written.load() << '\n';
    out << "output written to: " << settings.output_file << "\n";
}

} // namespace datagen
