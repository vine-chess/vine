#include "game_runner.hpp"
#include "../chess/move_gen.hpp"
#include "../eval/evaluator.hpp"
#include "../util/math.hpp"
#include "format/monty_format.hpp"
#include "format/viri_format.hpp"

#include <atomic>
#include <csignal>
#include <cstdlib>
#include <deque>
#include <exception>
#include <fstream>
#include <iostream>
#include <optional>
#include <string_view>
#include <syncstream>
#include <type_traits>

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
    case EvaluatorBackend::GPU:
        return "gpu";
    }

    return "unknown";
}

[[nodiscard]] constexpr usize workers_per_thread(const Settings &settings) {
    return settings.workers_per_thread == 0 ? 1 : settings.workers_per_thread;
}

[[nodiscard]] constexpr usize hash_per_worker(const Settings &settings) {
    const usize workers = workers_per_thread(settings);
    const usize hash = settings.hash_size / workers;
    return hash == 0 ? 1 : hash;
}

template <class DataWriter>
struct SearchContext {
    explicit SearchContext(std::ostream &out) : writer(out) {}

    search::Searcher searcher;
    Board board;
    DataWriter writer;
    bool running = false;
    u16 white_win_plies = 0;
    u16 white_loss_plies = 0;
    u16 draw_plies = 0;
};

[[nodiscard]] f64 terminal_game_result(const Board &board) {
    return board.state().checkers != 0 ? board.state().side_to_move == Color::BLACK : 0.5;
}

template <class DataWriter>
void reset_adjudication(SearchContext<DataWriter> &worker) {
    worker.white_win_plies = 0;
    worker.white_loss_plies = 0;
    worker.draw_plies = 0;
}

template <class DataWriter>
[[nodiscard]] std::optional<f64> adjudicated_result(SearchContext<DataWriter> &worker, const Board &board,
                                                    const search::NodeReference best_child, const f64 score) {
    if (best_child.info.terminal_state.is_win()) {
        return static_cast<f64>(board.state().side_to_move == Color::BLACK);
    }
    if (best_child.info.terminal_state.is_loss()) {
        return static_cast<f64>(board.state().side_to_move == Color::WHITE);
    }

    const f64 white_relative_cp =
        400 * util::math::inverse_sigmoid(board.state().side_to_move == Color::WHITE ? score : 1.0 - score);
    if (white_relative_cp >= 2000) {
        ++worker.white_win_plies;
        worker.white_loss_plies = 0;
        worker.draw_plies = 0;
    } else if (white_relative_cp <= -2000) {
        ++worker.white_loss_plies;
        worker.white_win_plies = 0;
        worker.draw_plies = 0;
    } else if (std::abs(white_relative_cp) <= 30) {
        ++worker.draw_plies;
        worker.white_win_plies = 0;
        worker.white_loss_plies = 0;
    } else {
        worker.white_win_plies = 0;
        worker.white_loss_plies = 0;
        worker.draw_plies = 0;
    }

    if (worker.white_win_plies >= 5) {
        return 1.0;
    }
    if (worker.white_loss_plies >= 5) {
        return 0.0;
    }
    if (worker.draw_plies >= 10 && board.state().fifty_moves_clock >= 20) {
        return 0.5;
    }

    return std::nullopt;
}

template <class DataWriter>
void push_move_data(DataWriter &writer, search::GameTree &game_tree, const search::NodeReference root_node,
                    const search::NodeReference best_child, const BoardState &state) {
    const f64 score = 1.0 - best_child.q();
    writer.push_move(best_child.info.move, score, state);

    if constexpr (std::is_same_v<DataWriter, MontyFormatWriter>) {
        for (usize j = 0; j < root_node.info.num_children; ++j) {
            const auto child = game_tree.node_at(root_node.info.first_child_idx + j);
            writer.push_visit(child.info.move, child.num_visits, state);
        }
    }
}

template <class Evaluator, class DataWriter>
void thread_loop(const Settings &settings, const usize thread_id, std::ostream &out_file,
                 const std::vector<std::string> &opening_fens) {
    Evaluator evaluator;

    rng::seed_generator(std::random_device{}(), std::hash<std::thread::id>{}(std::this_thread::get_id()));

    std::deque<SearchContext<DataWriter>> workers;
    for (usize i = 0; i < workers_per_thread(settings); ++i) {
        workers.emplace_back(out_file);
    }
    for (auto &worker : workers) {
        worker.searcher.set_hash_size(hash_per_worker(settings));
        worker.searcher.set_verbosity(search::Verbosity::NONE);
    }

    usize next_game_idx = thread_id;
    usize next_worker_idx = 0;

    while (!stop_flag.load(std::memory_order_relaxed)) {
        bool has_running_workers = false;
        for (const auto &worker : workers) {
            if (worker.running) {
                has_running_workers = true;
                break;
            }
        }

        if (next_game_idx >= settings.num_games && !has_running_workers) {
            break;
        }

        auto &worker = workers[next_worker_idx];
        next_worker_idx = (next_worker_idx + 1) % workers.size();

        if (!worker.running) {
            if (next_game_idx >= settings.num_games) {
                continue;
            }

            worker.board =
                Board(generate_opening(opening_fens, settings.random_moves, settings.temperature, settings.gamma));
            worker.searcher.clear();
            worker.writer.push_board_state(worker.board.state());
            reset_adjudication(worker);
            worker.running = true;
            next_game_idx += settings.num_threads;
        }

        worker.searcher.go(worker.board, evaluator, settings.time_settings);

        auto &game_tree = worker.searcher.game_tree();
        const auto root_node = game_tree.root();
        if (root_node.terminal()) {
            worker.writer.write_with_result(terminal_game_result(worker.board));
            worker.searcher.clear();
            worker.running = false;
            games_played.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        search::NodeIndex best_child_idx = root_node.info.first_child_idx;
        for (usize j = 0; j < root_node.info.num_children; ++j) {
            const auto child = game_tree.node_at(root_node.info.first_child_idx + j);
            if (child.q() < game_tree.node_at(best_child_idx).q()) {
                best_child_idx = root_node.info.first_child_idx + j;
            }
        }

        const auto best_child = game_tree.node_at(best_child_idx);
        vine_assert(!best_child.info.move.is_null());

        const f64 score = 1.0 - best_child.q();
        if (const auto result = adjudicated_result(worker, worker.board, best_child, score)) {
            worker.writer.write_with_result(*result);
            worker.searcher.clear();
            worker.running = false;
            games_played.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        push_move_data(worker.writer, game_tree, root_node, best_child, worker.board.state());
        worker.board.make_move(best_child.info.move);

        positions_written.fetch_add(1, std::memory_order_relaxed);

        if (worker.board.is_draw()) {
            worker.writer.write_with_result(0.5);
            worker.searcher.clear();
            worker.running = false;
            games_played.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

template <class Evaluator, class DataWriter>
void launch_threads(const Settings &settings, std::ostream &final_output, const std::vector<std::string> &opening_fens,
                    std::vector<std::thread> &threads) {
    for (usize thread_id = 0; thread_id < settings.num_threads; ++thread_id) {
        threads.emplace_back([settings, thread_id, &final_output, &opening_fens]() {
            std::osyncstream thread_output(final_output);
            thread_loop<Evaluator, DataWriter>(settings, thread_id, thread_output, opening_fens);
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

    std::vector<std::thread> threads;

    std::thread monitor([&out, &settings]() {
        usize last_games = 0;
        usize last_positions = 0;
        auto last_time = std::chrono::steady_clock::now();
        bool printed = false;

        while (!stop_flag.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(5));

            auto current_time = std::chrono::steady_clock::now();
            const f64 elapsed_sec = std::chrono::duration<f64>(current_time - last_time).count();

            const usize current_games = games_played.load();
            const usize current_positions = positions_written.load();
            const usize total_games = settings.num_games;
            const f64 games_per_sec = (current_games - last_games) / elapsed_sec;
            const f64 positions_per_sec = (current_positions - last_positions) / elapsed_sec;
            const f64 remaining_games = total_games > current_games ? total_games - current_games : 0;
            const f64 eta_sec = games_per_sec > 0.0 ? remaining_games / games_per_sec : 0.0;
            const usize eta_min = static_cast<usize>(eta_sec) / 60;
            const usize eta_hour = eta_min / 60;
            const usize eta_rem_min = eta_min % 60;
            const usize eta_rem_sec = static_cast<usize>(eta_sec) % 60;

            if (printed) {
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
    }

    std::ofstream final_output(settings.output_file, std::ios::binary);
    if (!final_output) {
        stop_flag = true;
        monitor.join();
        out << "error: failed to open output file: " << settings.output_file << '\n';
        return;
    }

    std::vector<char> big_buf(1 << 20);
    final_output.rdbuf()->pubsetbuf(big_buf.data(), big_buf.size());

    try {
        switch (settings.evaluator_backend) {
        case EvaluatorBackend::CPU:
            if (settings.mode == DatagenMode::value) {
                launch_threads<network::CpuEvaluator, ViriformatWriter>(settings, final_output, opening_fens, threads);
            } else {
                launch_threads<network::CpuEvaluator, MontyFormatWriter>(settings, final_output, opening_fens, threads);
            }
            break;
#ifdef DATAGEN_CUDA
        case EvaluatorBackend::GPU:
            network::QueuedGpuEvaluator::set_batch_size(std::max<u32>(1, static_cast<u32>(settings.num_threads / 2)));
            if (settings.mode == DatagenMode::value) {
                launch_threads<network::QueuedGpuEvaluator, ViriformatWriter>(settings, final_output, opening_fens,
                                                                              threads);
            } else {
                launch_threads<network::QueuedGpuEvaluator, MontyFormatWriter>(settings, final_output, opening_fens,
                                                                               threads);
            }
            break;
#else
        case EvaluatorBackend::GPU:
            stop_flag = true;
            monitor.join();
            out << "error: this datagen build does not include CUDA evaluators\n";
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
    out << "output written to: " << settings.output_file << "\n";
}

} // namespace datagen
