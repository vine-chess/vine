#include "game_runner.hpp"
#include "../chess/move_gen.hpp"
#include "../eval/evaluator.hpp"
#include "format/monty_format.hpp"
#include <atomic>
#include <csignal>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <string_view>

namespace datagen {

std::atomic_bool stop_flag;
void signal_handler([[maybe_unused]] i32 signum) {
    stop_flag = true;
}

std::atomic_size_t games_played = 0;
std::atomic_size_t positions_written = 0;

namespace {

[[nodiscard]] constexpr std::string_view evaluator_backend_name(const EvaluatorBackend backend) {
    switch (backend) {
    case EvaluatorBackend::CPU:
        return "cpu";
    case EvaluatorBackend::QUEUED_CPU:
        return "queued_cpu";
    case EvaluatorBackend::GPU:
        return "gpu";
    case EvaluatorBackend::QUEUED_GPU:
        return "queued_gpu";
    }

    return "unknown";
}

template <class Evaluator>
void thread_loop(const Settings &settings, const usize thread_id, std::ofstream &out_file,
                 const std::vector<std::string> &opening_fens) {
    MontyFormatWriter writer(out_file);

    search::Searcher searcher;
    Evaluator evaluator;
    searcher.set_hash_size(settings.hash_size);
    searcher.set_verbosity(search::Verbosity::NONE);

    rng::seed_generator(std::random_device{}(), std::hash<std::thread::id>{}(std::this_thread::get_id()));

    for (usize i = thread_id; i < settings.num_games && !stop_flag.load(std::memory_order_relaxed);
         i += settings.num_threads) {
        const auto base_opening_fen = opening_fens[rng::next_u64(0, opening_fens.size() - 1)];
        Board board(generate_opening(base_opening_fen, settings.random_moves, settings.temperature, settings.gamma));
        writer.push_board_state(board.state());

        f64 game_result;
        while (true) {
            searcher.go(board, evaluator, settings.time_settings);

            const auto &game_tree = searcher.game_tree();
            const auto &root_node = game_tree.root();
            if (root_node.terminal()) {
                game_result = board.state().checkers != 0 ? board.state().side_to_move == Color::BLACK : 0.5;
                break;
            }

            VisitsDistribution visits_dist;
            search::NodeIndex best_child_idx = root_node.first_child_idx;
            for (usize j = 0; j < root_node.num_children; j++) {
                const auto &child = game_tree.node_at(root_node.first_child_idx + j);
                visits_dist.emplace_back(writer.to_monty_move(child.move, board.state()), child.num_visits);
                if (child.q() < game_tree.node_at(best_child_idx).q()) {
                    best_child_idx = root_node.first_child_idx + j;
                }
            }

            const auto &best_child = game_tree.node_at(best_child_idx);
            vine_assert(!best_child.move.is_null());

            writer.push_move(best_child.move, 1.0 - best_child.q(), visits_dist, board.state());
            board.make_move(best_child.move);

            positions_written.fetch_add(1, std::memory_order_relaxed);

            if (board.is_draw()) {
                game_result = 0.5;
                break;
            }
        }

        writer.write_with_result(game_result);
        games_played.fetch_add(1, std::memory_order_relaxed);
    }
}

template <class Evaluator>
void launch_threads(const Settings &settings, std::ostream &out, const std::vector<std::string> &opening_fens,
                    std::vector<std::string> &thread_files, std::vector<std::thread> &threads) {
    for (usize thread_id = 0; thread_id < settings.num_threads; ++thread_id) {
        const auto thread_file_path = settings.output_file + "_temp" + std::to_string(thread_id);
        thread_files.push_back(thread_file_path);

        threads.emplace_back([settings, thread_id, thread_file_path, &out, &opening_fens]() {
            std::ofstream thread_output(thread_file_path, std::ios::binary | std::ios::app);
            if (!thread_output) {
                out << "failed to open thread output file " << thread_file_path << std::endl;
                return;
            }

            thread_loop<Evaluator>(settings, thread_id, thread_output, opening_fens);
        });
    }
}

} // namespace

void run_games(Settings settings, std::ostream &out) {
    stop_flag = false;
    games_played = 0;
    positions_written = 0;

    if (settings.num_threads == 0) {
        out << "error: datagen requires at least one thread\n";
        return;
    }

    out << "starting datagen..." << std::endl;
    out << "  evaluator         : " << evaluator_backend_name(settings.evaluator_backend) << '\n';

    std::signal(SIGINT, signal_handler);

    std::vector<std::string> thread_files;
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
            usize eta_rem_sec = static_cast<usize>(eta_sec) % 60;

            if (printed) {
                // Clear previous lines (5 lines)
                out << "\033[F\033[K\033[F\033[K\033[F\033[K\033[F\033[K\033[F\033[K";
            }

            out << "progress update:\n";
            out << "  games played      : " << current_games << " / " << total_games << '\n';
            out << "  positions written : " << current_positions << '\n';
            out << "  throughput        : " << games_per_sec << " games/s, " << positions_per_sec << " pos/s\n";
            out << "  eta               : " << eta_min << "m " << eta_rem_sec << "s\n";

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

    try {
        switch (settings.evaluator_backend) {
        case EvaluatorBackend::CPU:
            launch_threads<network::CpuEvaluator>(settings, out, opening_fens, thread_files, threads);
            break;
        case EvaluatorBackend::QUEUED_CPU:
            launch_threads<network::QueuedCpuEvaluator>(settings, out, opening_fens, thread_files, threads);
            break;
#ifdef DATAGEN_CUDA
        case EvaluatorBackend::GPU:
            launch_threads<network::GpuEvaluator>(settings, out, opening_fens, thread_files, threads);
            break;
        case EvaluatorBackend::QUEUED_GPU:
            network::QueuedGpuEvaluator::set_batch_size(std::max<u32>(1, static_cast<u32>(settings.num_threads / 2)));
            launch_threads<network::QueuedGpuEvaluator>(settings, out, opening_fens, thread_files, threads);
            break;
#else
        case EvaluatorBackend::GPU:
        case EvaluatorBackend::QUEUED_GPU:
            out << "error: this datagen build does not include CUDA evaluators\n";
            stop_flag = true;
            monitor.join();
            return;
#endif
        }
    } catch (const std::exception &e) {
        stop_flag = true;
        monitor.join();
        out << "error: failed to initialize datagen evaluator: " << e.what() << '\n';
        return;
    }

    for (auto &thread : threads) {
        thread.join();
    }

    stop_flag = true;
    monitor.join();

    out << "\ndatagen complete:\n";
    out << "  total games played      : " << games_played.load() << '\n';
    out << "  total positions written : " << positions_written.load() << '\n';

    out << "\ncombining output files...\n";

    std::ofstream final_output(settings.output_file, std::ios::binary);
    if (!final_output) {
        out << "error: failed to open final output file " << settings.output_file << '\n';
        return;
    }

    for (const auto &temp_file : thread_files) {
        std::ifstream in(temp_file, std::ios::binary);
        if (!in) {
            out << "warning: failed to open temp file " << temp_file << '\n';
            continue;
        }

        final_output << in.rdbuf();
        in.close();

        std::remove(temp_file.c_str());
    }

    out << "output written to: " << settings.output_file << "\n";
}

} // namespace datagen
