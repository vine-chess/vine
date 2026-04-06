#include "game_runner.hpp"
#include "../chess/move_gen.hpp"
#include "../eval/evaluator.hpp"
#include "../util/math.hpp"
#include "../util/ring_queue.hpp"
#include "../util/sharded_queue.hpp"
#include "format/monty_format.hpp"
#include "format/viri_format.hpp"

#include <atomic>
#include <csignal>
#include <cstdlib>
#include <deque>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
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

[[nodiscard]] constexpr usize searcher_count(const Settings &settings) {
    return std::max(settings.num_threads, settings.total_memory / std::max<usize>(1, settings.hash_size));
}

[[nodiscard]] constexpr usize gpu_batch_size(const Settings &settings) {
    return std::max<usize>(1, searcher_count(settings) / (2 * std::max<usize>(1, settings.gpu_workers_per_queue)));
}

template <class DataWriter>
struct SearchContext {
    explicit SearchContext(std::ostream &out) : writer(out) {}

    search::Searcher searcher;
    search::TimeManager time_manager;
    util::StaticVector<u32, MAX_MOVES> old_visit_dist;
    Board board;
    DataWriter writer;
    bool resume = false;
    u64 iterations = 0;
    u64 previous_depth = 0;
    u16 white_win_plies = 0;
    u16 white_loss_plies = 0;
    u16 draw_plies = 0;
};

template <class DataWriter>
struct GamePool {
    using Game = SearchContext<DataWriter>;
    using GamePtr = Game *;
    using ReadyQueue = util::RingQueue<GamePtr>;
    using WorkQueue = util::ShardedQueue<GamePtr>;

    std::deque<SearchContext<DataWriter>> games;
    ReadyQueue free;
    ReadyQueue search;
    WorkQueue value;
    WorkQueue policy;

    void reset(usize count, usize n, std::ostream &out) {
        games.clear();
        for (usize i = 0; i < count; ++i) {
            games.emplace_back(out);
        }

        free.reset(count);
        search.reset(count);
        value.reset(n, count);
        policy.reset(n, count);
        for (auto &game : games) {
            vine_assert(free.try_push(&game));
        }
    }
};

template <class Queue, class T>
void push_until(Queue &q, T value) {
    while (!q.try_push(value)) {
        std::this_thread::yield();
    }
}

[[nodiscard]] bool all_done(const Settings &settings, const std::atomic_size_t &next_game_idx,
                            const std::atomic_size_t &active_games) {
    return next_game_idx.load(std::memory_order_acquire) >= settings.num_games &&
           active_games.load(std::memory_order_acquire) == 0;
}

[[nodiscard]] f64 terminal_game_result(const Board &board) {
    return board.state().checkers != 0 ? board.state().side_to_move == Color::BLACK : 0.5;
}

template <class DataWriter>
void reset_adjudication(SearchContext<DataWriter> &worker) {
    worker.old_visit_dist.clear();
    worker.resume = false;
    worker.iterations = 0;
    worker.previous_depth = 0;
    worker.white_win_plies = 0;
    worker.white_loss_plies = 0;
    worker.draw_plies = 0;
}

template <class DataWriter>
[[nodiscard]] std::optional<f64> adjudicated_result(SearchContext<DataWriter> &worker, const Board &board,
                                                    const search::NodeReference best_child, const f64 score) {
    if (best_child.info.terminal_state.is_win()) {
        return board.state().side_to_move == Color::BLACK ? 1.0 : 0.0;
    }
    if (best_child.info.terminal_state.is_loss()) {
        return board.state().side_to_move == Color::WHITE ? 1.0 : 0.0;
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

template <class Evaluator>
void finish_policy(search::Searcher &searcher, Evaluator &evaluator, const search::Request request) {
    const auto &state = searcher.pending_state();
    auto &tree = searcher.game_tree();
    const auto node = tree.node_at(request.node);

    auto ctx = evaluator.policy_context(state);
    for (u16 i = 0; i < node.info.num_children; ++i) {
        const auto child = tree.node_at(node.info.first_child_idx + i);
        ctx.enqueue(child.info.move, state.get_piece_type(child.info.move.from()));
    }
    ctx.ready();
    searcher.finish_policy([&] { return ctx.logit(); });
}

template <class DataWriter>
void finish_game(SearchContext<DataWriter> *worker, std::atomic_size_t &active_games, GamePool<DataWriter> &pool,
                 std::mutex &out_mutex, const f64 result) {
    {
        const std::lock_guard lock(out_mutex);
        worker->writer.write_with_result(result);
    }
    worker->searcher.clear();
    active_games.fetch_sub(1, std::memory_order_relaxed);
    games_played.fetch_add(1, std::memory_order_relaxed);
    push_until(pool.free, worker);
}

template <class DataWriter>
void finish_move(SearchContext<DataWriter> *worker, const Settings &settings, std::atomic_size_t &active_games,
                 GamePool<DataWriter> &pool, std::mutex &out_mutex) {
    auto &game_tree = worker->searcher.game_tree();
    const auto root_node = game_tree.root();

    if (!root_node.info.terminal_state.is_none()) {
        finish_game(worker, active_games, pool, out_mutex, terminal_game_result(worker->board));
        return;
    }

    search::NodeIndex best_child_idx = root_node.info.first_child_idx;
    f32 best_q = game_tree.node_at(best_child_idx).q();
    for (usize j = 1; j < root_node.info.num_children; ++j) {
        const auto child = game_tree.node_at(root_node.info.first_child_idx + j);
        if (child.q() < best_q) {
            best_child_idx = root_node.info.first_child_idx + j;
            best_q = child.q();
        }
    }

    const auto best_child = game_tree.node_at(best_child_idx);
    vine_assert(!best_child.info.move.is_null());

    const f64 score = 1.0 - best_child.q();
    if (const auto result = adjudicated_result(*worker, worker->board, best_child, score)) {
        finish_game(worker, active_games, pool, out_mutex, *result);
        return;
    }

    push_move_data(worker->writer, game_tree, root_node, best_child, worker->board.state());
    worker->board.make_move(best_child.info.move);

    positions_written.fetch_add(1, std::memory_order_relaxed);

    if (worker->board.is_draw()) {
        finish_game(worker, active_games, pool, out_mutex, 0.5);
        return;
    }

    worker->time_manager.start_tracking(settings.time_settings);
    reset_adjudication(*worker);
    push_until(pool.search, worker);
}

template <class Evaluator, class DataWriter>
void value_thread_loop(const Settings &settings, std::atomic_size_t &next_game_idx, std::atomic_size_t &active_games,
                       GamePool<DataWriter> &pool, typename GamePool<DataWriter>::WorkQueue::Receiver rx) {
    if constexpr (std::is_same_v<Evaluator, network::QueuedGpuEvaluator>) {
        auto &queue = network::GlobalGpuValueQueue::get().queue();
        queue.start();
        std::vector<SearchContext<DataWriter> *> workers(queue.batch_size());
        std::vector<network::GpuValueQueue::ValueSlot> slots(queue.batch_size());

        while (!stop_flag.load(std::memory_order_relaxed)) {
            const usize count = rx.try_pop_some(std::span(workers.data(), workers.size()));
            if (count == 0) {
                if (all_done(settings, next_game_idx, active_games)) {
                    break;
                }
                std::this_thread::yield();
                continue;
            }

            for (usize i = 0; i < count; ++i) {
                slots[i] = queue.reserve_value_slot(workers[i]->searcher.pending_state());
                queue.mark_ready(slots[i]);
            }

            for (usize i = 0; i < count; ++i) {
                workers[i]->searcher.finish_value(static_cast<f32>(queue.wait_for_result(slots[i])));
                queue.mark_consumed(slots[i]);
                workers[i]->resume = true;
                push_until(pool.search, workers[i]);
            }
        }
    } else {
        Evaluator evaluator;

        while (!stop_flag.load(std::memory_order_relaxed)) {
            SearchContext<DataWriter> *worker = nullptr;
            if (!rx.try_pop(worker)) {
                if (all_done(settings, next_game_idx, active_games)) {
                    break;
                }
                std::this_thread::yield();
                continue;
            }

            worker->searcher.finish_value(static_cast<f32>(evaluator.value(worker->searcher.pending_state())));
            worker->resume = true;
            push_until(pool.search, worker);
        }
    }
}

template <class Evaluator, class DataWriter>
void policy_thread_loop(const Settings &settings, std::atomic_size_t &next_game_idx, std::atomic_size_t &active_games,
                        GamePool<DataWriter> &pool, typename GamePool<DataWriter>::WorkQueue::Receiver rx) {
    if constexpr (std::is_same_v<Evaluator, network::QueuedGpuEvaluator>) {
        auto &queue = network::GlobalGpuPolicyQueue::get().queue();
        queue.start();
        std::vector<SearchContext<DataWriter> *> workers(queue.batch_size());
        std::vector<search::Request> requests(queue.batch_size());
        std::vector<network::GpuPolicyQueue::PolicySlot> slots(queue.batch_size());

        while (!stop_flag.load(std::memory_order_relaxed)) {
            const usize count = rx.try_pop_some(std::span(workers.data(), workers.size()));
            if (count == 0) {
                if (all_done(settings, next_game_idx, active_games)) {
                    break;
                }
                std::this_thread::yield();
                continue;
            }

            for (usize i = 0; i < count; ++i) {
                const auto req = workers[i]->searcher.poll();
                vine_assert(req);
                requests[i] = *req;

                const auto &state = workers[i]->searcher.pending_state();
                auto &tree = workers[i]->searcher.game_tree();
                const auto node = tree.node_at(requests[i].node);

                slots[i] = queue.reserve_policy_slot(state, node.info.num_children);
                for (u16 j = 0; j < node.info.num_children; ++j) {
                    const auto child = tree.node_at(node.info.first_child_idx + j);
                    slots[i].move_indices[j] =
                        network::policy::move_output_idx(state, child.info.move,
                                                         state.get_piece_type(child.info.move.from()));
                }
                queue.mark_ready(slots[i]);
            }

            for (usize i = 0; i < count; ++i) {
                const auto result = queue.wait_for_result(slots[i]);
                usize next_idx = 0;
                workers[i]->searcher.finish_policy([&] { return result.logits[next_idx++]; });
                queue.mark_consumed(slots[i]);
                workers[i]->resume = true;
                push_until(pool.search, workers[i]);
            }
        }
    } else {
        Evaluator evaluator;

        while (!stop_flag.load(std::memory_order_relaxed)) {
            SearchContext<DataWriter> *worker = nullptr;
            if (!rx.try_pop(worker)) {
                if (all_done(settings, next_game_idx, active_games)) {
                    break;
                }
                std::this_thread::yield();
                continue;
            }

            const auto req = worker->searcher.poll();
            vine_assert(req);
            finish_policy(worker->searcher, evaluator, *req);
            worker->resume = true;
            push_until(pool.search, worker);
        }
    }
}

template <class DataWriter>
bool times_up(SearchContext<DataWriter> &worker) {
    auto &tree = worker.searcher.game_tree();
    ++worker.iterations;

    const u64 depth = tree.sum_depths() / worker.iterations;
    worker.previous_depth = std::max(worker.previous_depth, depth);

    util::StaticVector<u32, MAX_MOVES> new_visit_dist;
    for (u16 i = 0; i < tree.root().info.num_children; ++i) {
        new_visit_dist.push_back(tree.node_at(tree.root().info.first_child_idx + i).num_visits);
    }

    const bool stop = worker.time_manager.times_up(tree, worker.iterations, worker.board.state().side_to_move,
                                                   worker.previous_depth, worker.old_visit_dist, new_visit_dist);
    worker.old_visit_dist = new_visit_dist;
    return stop;
}

template <class Evaluator, class DataWriter>
void thread_loop(const Settings &settings, std::atomic_size_t &next_game_idx, std::atomic_size_t &active_games,
                 GamePool<DataWriter> &pool, std::mutex &out_mutex, const std::vector<std::string> &opening_fens,
                 typename GamePool<DataWriter>::WorkQueue::Sender value_tx,
                 typename GamePool<DataWriter>::WorkQueue::Sender policy_tx) {
    rng::seed_generator(std::random_device{}(), std::hash<std::thread::id>{}(std::this_thread::get_id()));

    while (!stop_flag.load(std::memory_order_relaxed)) {
        SearchContext<DataWriter> *worker = nullptr;
        if (!pool.search.try_pop(worker)) {
            if (!pool.free.try_pop(worker)) {
                if (all_done(settings, next_game_idx, active_games)) {
                    break;
                }
                std::this_thread::yield();
                continue;
            }

            const auto game_idx = next_game_idx.fetch_add(1, std::memory_order_relaxed);
            if (game_idx >= settings.num_games) {
                push_until(pool.free, worker);
                if (active_games.load(std::memory_order_relaxed) == 0) {
                    break;
                }
                std::this_thread::yield();
                continue;
            }
            active_games.fetch_add(1, std::memory_order_release);

            worker->board =
                Board(generate_opening(opening_fens, settings.random_moves, settings.temperature, settings.gamma));
            worker->searcher.clear();
            worker->searcher.set_hash_size(std::max<usize>(1, settings.hash_size));
            worker->searcher.set_verbosity(search::Verbosity::NONE);
            worker->writer.push_board_state(worker->board.state());
            worker->time_manager.start_tracking(settings.time_settings);
            reset_adjudication(*worker);
        }

        if (worker->resume) {
            worker->searcher.ready();
            worker->resume = false;
            if (times_up(*worker)) {
                finish_move(worker, settings, active_games, pool, out_mutex);
                continue;
            }
        }

        const auto request = worker->searcher.poll(worker->board);

        switch (request.kind) {
        case search::RequestKind::Value:
            if (!worker->searcher.game_tree().node_at(request.node).info.terminal_state.is_none()) {
                worker->searcher.finish_value(
                    static_cast<f32>(worker->searcher.game_tree().node_at(request.node).info.terminal_state.score()));
                worker->resume = true;
                push_until(pool.search, worker);
            } else {
                push_until(value_tx, worker);
            }
            break;
        case search::RequestKind::Policy:
            push_until(policy_tx, worker);
            break;
        }
    }
}

template <class Evaluator, class DataWriter>
void launch_threads(const Settings &settings, std::ostream &final_output, const std::vector<std::string> &opening_fens,
                    std::vector<std::thread> &threads) {
    auto pool = std::make_shared<GamePool<DataWriter>>();
    pool->reset(searcher_count(settings), settings.num_threads, final_output);
    auto next_game_idx = std::make_shared<std::atomic_size_t>(0);
    auto active_games = std::make_shared<std::atomic_size_t>(0);
    auto out_mutex = std::make_shared<std::mutex>();
    const usize gpu_workers = settings.evaluator_backend == EvaluatorBackend::GPU
                                  ? std::max<usize>(1, settings.gpu_workers_per_queue)
                                  : 1;
    std::vector<typename GamePool<DataWriter>::WorkQueue::Receiver> value_rxs;
    std::vector<typename GamePool<DataWriter>::WorkQueue::Receiver> policy_rxs;
    std::vector<typename GamePool<DataWriter>::WorkQueue::Sender> value_txs;
    std::vector<typename GamePool<DataWriter>::WorkQueue::Sender> policy_txs;

    value_rxs.reserve(gpu_workers);
    policy_rxs.reserve(gpu_workers);
    value_txs.reserve(settings.num_threads);
    policy_txs.reserve(settings.num_threads);

    for (usize i = 0; i < gpu_workers; ++i) {
        value_rxs.emplace_back(pool->value.receiver());
        policy_rxs.emplace_back(pool->policy.receiver());
    }
    for (usize i = 0; i < settings.num_threads; ++i) {
        value_txs.emplace_back(pool->value.sender());
        policy_txs.emplace_back(pool->policy.sender());
    }

    for (usize i = 0; i < gpu_workers; ++i) {
        auto rx = std::move(value_rxs[i]);
        threads.emplace_back([settings, pool, next_game_idx, active_games, rx = std::move(rx)]() mutable {
            value_thread_loop<Evaluator, DataWriter>(settings, *next_game_idx, *active_games, *pool, std::move(rx));
        });
    }
    for (usize i = 0; i < gpu_workers; ++i) {
        auto rx = std::move(policy_rxs[i]);
        threads.emplace_back([settings, pool, next_game_idx, active_games, rx = std::move(rx)]() mutable {
            policy_thread_loop<Evaluator, DataWriter>(settings, *next_game_idx, *active_games, *pool, std::move(rx));
        });
    }

    for (usize i = 0; i < settings.num_threads; ++i) {
        auto value_tx = std::move(value_txs[i]);
        auto policy_tx = std::move(policy_txs[i]);
        threads.emplace_back([settings, pool, next_game_idx, active_games, out_mutex, &opening_fens,
                              value_tx = std::move(value_tx), policy_tx = std::move(policy_tx)]() mutable {
            thread_loop<Evaluator, DataWriter>(settings, *next_game_idx, *active_games, *pool, *out_mutex,
                                               opening_fens, std::move(value_tx), std::move(policy_tx));
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
            const usize eta = eta_sec;
            const usize eta_min = eta / 60;
            const usize eta_hour = eta_min / 60;
            const usize eta_rem_min = eta_min % 60;
            const usize eta_rem_sec = eta % 60;

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
            network::QueuedGpuEvaluator::set_batch_size(static_cast<u32>(gpu_batch_size(settings)));
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
