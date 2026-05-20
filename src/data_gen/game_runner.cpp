#include "game_runner.hpp"
#include "../chess/move_gen.hpp"
#include "../eval/evaluator.hpp"
#include "../eval/gpu_queue.hpp"
#include "../util/math.hpp"
#include "../util/sharded_queue.hpp"
#include "../util/tui.hpp"
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

std::atomic_uint64_t games_played = 0;
std::atomic_uint64_t active_games_live = 0;
std::atomic_uint64_t positions_played = 0;
std::atomic_uint64_t positions_written = 0;
std::atomic_uint64_t value_batches = 0;
std::atomic_uint64_t value_batch_items = 0;
std::atomic_uint64_t policy_batches = 0;
std::atomic_uint64_t policy_batch_items = 0;
std::atomic_uint64_t policy_moves = 0;

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

[[nodiscard]] f64 child_selection_score(const search::NodeReference child) {
    constexpr f64 mate_score = 1000.0;
    switch (child.info.terminal_state.flag()) {
    case search::TerminalState::Flag::WIN:
        return mate_score - child.info.terminal_state.distance_to_terminal();
    case search::TerminalState::Flag::LOSS:
        return -mate_score + child.info.terminal_state.distance_to_terminal();
    default:
        vine_assert(child.visited());
        return 1.0 - child.q();
    }
}

[[nodiscard]] constexpr usize searcher_count(const Settings &settings) {
    return std::max(settings.num_threads, settings.total_memory / std::max<usize>(1, settings.hash_size));
}

[[nodiscard]] constexpr usize gpu_batch_size(const Settings &settings) {
    return std::max<usize>(1, searcher_count(settings) / (2 * std::max<usize>(1, settings.gpu_workers_per_queue)));
}

template <class DataWriter>
struct DatagenGame {
    explicit DatagenGame(std::ostream &out) : writer(out) {}

    search::Searcher searcher;
    search::TimeManager time_manager;
    util::StaticVector<u32, MAX_MOVES> old_visit_dist;
    Board board;
    DataWriter writer;
    bool resume = false;
    u64 iterations = 0;
    u64 previous_depth = 0;
    u16 plies = 0;
    u16 white_win_plies = 0;
    u16 white_loss_plies = 0;
    u16 draw_plies = 0;
    std::string opening_fen;
};

template <class DataWriter>
struct GamePool {
    using Game = DatagenGame<DataWriter>;
    using GamePtr = Game *;
    using ReadyQueue = util::RingQueue<GamePtr>;
    using WorkQueue = util::ShardedQueue<GamePtr>;

    std::deque<DatagenGame<DataWriter>> games;
    ReadyQueue free;
    ReadyQueue search;
    WorkQueue value;
    WorkQueue policy;

    void reset(usize count, usize shards, std::ostream &out) {
        games.clear();
        for (usize i = 0; i < count; ++i) {
            games.emplace_back(out);
        }

        free.reset(count);
        search.reset(count);
        value.reset(shards, count);
        policy.reset(shards, count);
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

template <class Receiver, class T>
[[nodiscard]] usize fill_batch(Receiver &rx, std::span<T> batch, const Settings &settings,
                               const std::atomic_size_t &next_game_idx, const std::atomic_size_t &active_games) {
    usize count = rx.try_pop_some(batch);
    while (count != 0 && count < batch.size() && !stop_flag.load(std::memory_order_relaxed) &&
           !all_done(settings, next_game_idx, active_games)) {
        const usize added = rx.try_pop_some(batch.subspan(count));
        if (added == 0) {
            std::this_thread::yield();
            continue;
        }
        count += added;
    }

    return count;
}

template <class DataWriter>
void start_game(DatagenGame<DataWriter> &game, const Settings &settings, const std::vector<std::string> &opening_fens,
                const usize game_idx) {
    rng::add_entropy(game_idx);
    game.board = Board(generate_opening(opening_fens, settings.random_moves, settings.temperature, settings.gamma));
    game.opening_fen = game.board.state().to_fen();
    game.searcher.clear();
    game.searcher.set_hash_size(std::max<usize>(1, settings.hash_size));
    game.searcher.set_verbosity(search::Verbosity::NONE);
    game.writer.push_board_state(game.board.state());
    game.time_manager.start_tracking(settings.time_settings);
    game.old_visit_dist.clear();
    game.resume = false;
    game.iterations = 0;
    game.previous_depth = 0;
    game.plies = 0;
    game.white_win_plies = 0;
    game.white_loss_plies = 0;
    game.draw_plies = 0;
}

template <class DataWriter>
[[nodiscard]] DatagenGame<DataWriter> *next_game(const Settings &settings, std::atomic_size_t &next_game_idx,
                                                 std::atomic_size_t &active_games, GamePool<DataWriter> &pool,
                                                 const std::vector<std::string> &opening_fens) {
    DatagenGame<DataWriter> *game = nullptr;
    if (!pool.free.try_pop(game)) {
        return nullptr;
    }

    const auto game_idx = next_game_idx.fetch_add(1, std::memory_order_relaxed);
    if (game_idx >= settings.num_games) {
        push_until(pool.free, game);
        return nullptr;
    }

    active_games.fetch_add(1, std::memory_order_release);
    active_games_live.fetch_add(1, std::memory_order_relaxed);
    start_game(*game, settings, opening_fens, game_idx);
    return game;
}

[[nodiscard]] f64 terminal_game_result(const Board &board) {
    return board.state().checkers != 0 ? board.state().side_to_move == Color::BLACK : 0.5;
}

template <class DataWriter>
[[nodiscard]] std::optional<f64> adjudicated_result(DatagenGame<DataWriter> &game, const Board &board,
                                                    const search::NodeReference best_child, const f64 score) {
    constexpr f64 win_loss_cp_threshold = 1000.0;
    constexpr f64 draw_cp_threshold = 25.0;
    constexpr u16 win_loss_plies_required = 5;
    constexpr u16 draw_plies_required = 10;
    constexpr u16 draw_fifty_clock_required = 20;
    constexpr u16 min_game_plies = 20;

    if (game.plies < min_game_plies) {
        return std::nullopt;
    }

    const f64 white_relative_cp =
        400 * util::math::inverse_sigmoid(board.state().side_to_move == Color::WHITE ? score : 1.0 - score);
    if (white_relative_cp >= win_loss_cp_threshold) {
        ++game.white_win_plies;
        game.white_loss_plies = 0;
        game.draw_plies = 0;
    } else if (white_relative_cp <= -win_loss_cp_threshold) {
        ++game.white_loss_plies;
        game.white_win_plies = 0;
        game.draw_plies = 0;
    } else if (std::abs(white_relative_cp) <= draw_cp_threshold) {
        ++game.draw_plies;
        game.white_win_plies = 0;
        game.white_loss_plies = 0;
    } else {
        game.white_win_plies = 0;
        game.white_loss_plies = 0;
        game.draw_plies = 0;
    }

    if (game.white_win_plies >= win_loss_plies_required) {
        return 1.0;
    }
    if (game.white_loss_plies >= win_loss_plies_required) {
        return 0.0;
    }
    if (game.draw_plies >= draw_plies_required && board.state().fifty_moves_clock >= draw_fifty_clock_required) {
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

template <class DataWriter>
void finish_game(DatagenGame<DataWriter> *game, std::atomic_size_t &active_games, GamePool<DataWriter> &pool,
                 std::mutex &out_mutex, const f64 result) {
    {
        const std::lock_guard lock(out_mutex);
        game->writer.write_with_result(result);
    }
    positions_written.fetch_add(game->plies, std::memory_order_relaxed);
    active_games.fetch_sub(1, std::memory_order_relaxed);
    active_games_live.fetch_sub(1, std::memory_order_relaxed);
    games_played.fetch_add(1, std::memory_order_relaxed);

    push_until(pool.free, game);
}

template <class DataWriter>
[[nodiscard]] bool finish_move(DatagenGame<DataWriter> *game, const Settings &settings,
                               std::atomic_size_t &active_games, GamePool<DataWriter> &pool, std::mutex &out_mutex,
                               bool requeue = true) {
    auto &game_tree = game->searcher.game_tree();
    const auto root_node = game_tree.root();

    if (root_node.terminal()) {
        finish_game(game, active_games, pool, out_mutex, terminal_game_result(game->board));
        return false;
    }

    std::optional<search::NodeIndex> best_child_idx;
    f64 best_score = -std::numeric_limits<f64>::max();
    for (usize j = 0; j < root_node.info.num_children; ++j) {
        const auto child = game_tree.node_at(root_node.info.first_child_idx + j);
        if (!child.visited()) {
            continue;
        }
        const auto score = child_selection_score(child);
        if (score > best_score) {
            best_child_idx = root_node.info.first_child_idx + j;
            best_score = score;
        }
    }

    if (!best_child_idx) {
        std::lock_guard lock(out_mutex);
        std::cerr << "no visited root children at move selection, fen: " << game->board.state().to_fen() << '\n';
    }
    vine_assert(best_child_idx);

    const auto best_child = game_tree.node_at(*best_child_idx);
    vine_assert(!best_child.info.move.is_null());

    const f64 score = 1.0 - best_child.q();
    if (const auto result = adjudicated_result(*game, game->board, best_child, score)) {
        finish_game(game, active_games, pool, out_mutex, *result);
        return false;
    }

    push_move_data(game->writer, game_tree, root_node, best_child, game->board.state());
    game->board.make_move(best_child.info.move);
    ++game->plies;
    positions_played.fetch_add(1, std::memory_order_relaxed);

    if (game->board.is_draw()) {
        finish_game(game, active_games, pool, out_mutex, 0.5);
        return false;
    }

    game->searcher.clear();
    game->time_manager.start_tracking(settings.time_settings);
    game->old_visit_dist.clear();
    game->resume = false;
    game->iterations = 0;
    game->previous_depth = 0;
    if (requeue) {
        push_until(pool.search, game);
    }
    return true;
}

template <class DataWriter>
void value_thread_loop_gpu(const Settings &settings, std::atomic_size_t &next_game_idx,
                           std::atomic_size_t &active_games, GamePool<DataWriter> &pool,
                           typename GamePool<DataWriter>::WorkQueue::Receiver rx, std::mutex &fill_mutex) {
    network::GpuValueQueue queue(static_cast<u32>(gpu_batch_size(settings)));
    queue.start();
    std::vector<DatagenGame<DataWriter> *> games(queue.batch_size());
    std::vector<network::GpuValueQueue::ValueSlot> slots(queue.batch_size());

    while (!stop_flag.load(std::memory_order_relaxed)) {
        usize count = 0;
        {
            std::unique_lock lock(fill_mutex, std::try_to_lock);
            if (!lock.owns_lock()) {
                if (all_done(settings, next_game_idx, active_games)) {
                    break;
                }
                std::this_thread::yield();
                continue;
            }

            count = fill_batch(rx, std::span(games.data(), games.size()), settings, next_game_idx, active_games);
        }
        if (count == 0) {
            if (all_done(settings, next_game_idx, active_games)) {
                break;
            }
            std::this_thread::yield();
            continue;
        }

        value_batches.fetch_add(1, std::memory_order_relaxed);
        value_batch_items.fetch_add(count, std::memory_order_relaxed);

        for (usize i = 0; i < count; ++i) {
            slots[i] = queue.reserve_value_slot(games[i]->searcher.pending_state());
            queue.mark_ready(slots[i]);
        }

        for (usize i = 0; i < count; ++i) {
            games[i]->searcher.finish_value(static_cast<f32>(queue.wait_for_result(slots[i])));
            queue.mark_consumed(slots[i]);
            games[i]->resume = true;
            push_until(pool.search, games[i]);
        }
    }
}

template <class DataWriter>
void value_thread_loop_cpu(const Settings &settings, std::atomic_size_t &next_game_idx,
                           std::atomic_size_t &active_games, GamePool<DataWriter> &pool,
                           typename GamePool<DataWriter>::WorkQueue::Receiver rx) {
    network::CpuEvaluator evaluator;

    while (!stop_flag.load(std::memory_order_relaxed)) {
        DatagenGame<DataWriter> *game = nullptr;
        if (!rx.try_pop(game)) {
            if (all_done(settings, next_game_idx, active_games)) {
                break;
            }
            std::this_thread::yield();
            continue;
        }

        game->searcher.finish_value(static_cast<f32>(evaluator.value(game->searcher.pending_state())));
        value_batch_items.fetch_add(1, std::memory_order_relaxed);
        game->resume = true;
        push_until(pool.search, game);
    }
}

template <class DataWriter>
void policy_thread_loop_gpu(const Settings &settings, std::atomic_size_t &next_game_idx,
                            std::atomic_size_t &active_games, GamePool<DataWriter> &pool,
                            typename GamePool<DataWriter>::WorkQueue::Receiver rx, std::mutex &fill_mutex) {
    network::GpuPolicyQueue queue(static_cast<u32>(gpu_batch_size(settings)));
    queue.start();
    std::vector<DatagenGame<DataWriter> *> games(queue.batch_size());
    std::vector<search::Request> requests(queue.batch_size());
    std::vector<network::GpuPolicyQueue::PolicySlot> slots(queue.batch_size());

    while (!stop_flag.load(std::memory_order_relaxed)) {
        usize count = 0;
        {
            std::unique_lock lock(fill_mutex, std::try_to_lock);
            if (!lock.owns_lock()) {
                if (all_done(settings, next_game_idx, active_games)) {
                    break;
                }
                std::this_thread::yield();
                continue;
            }

            count = fill_batch(rx, std::span(games.data(), games.size()), settings, next_game_idx, active_games);
        }
        if (count == 0) {
            if (all_done(settings, next_game_idx, active_games)) {
                break;
            }
            std::this_thread::yield();
            continue;
        }

        policy_batches.fetch_add(1, std::memory_order_relaxed);
        policy_batch_items.fetch_add(count, std::memory_order_relaxed);

        for (usize i = 0; i < count; ++i) {
            const auto req = games[i]->searcher.poll();
            vine_assert(req);
            requests[i] = *req;

            const auto &state = games[i]->searcher.pending_state();
            auto &tree = games[i]->searcher.game_tree();
            const auto node = tree.node_at(requests[i].node);
            policy_moves.fetch_add(node.info.num_children, std::memory_order_relaxed);

            slots[i] = queue.reserve_policy_slot(state, node.info.num_children);
            for (u16 j = 0; j < node.info.num_children; ++j) {
                const auto child = tree.node_at(node.info.first_child_idx + j);
                slots[i].move_indices[j] = network::policy::move_output_idx(
                    state, child.info.move, state.get_piece_type(child.info.move.from()));
            }
            queue.mark_ready(slots[i]);
        }

        for (usize i = 0; i < count; ++i) {
            const auto result = queue.wait_for_result(slots[i]);
            usize next_idx = 0;
            games[i]->searcher.finish_policy([&] { return result.logits[next_idx++]; });
            queue.mark_consumed(slots[i]);
            games[i]->resume = true;
            push_until(pool.search, games[i]);
        }
    }
}

template <class DataWriter>
void policy_thread_loop_cpu(const Settings &settings, std::atomic_size_t &next_game_idx,
                            std::atomic_size_t &active_games, GamePool<DataWriter> &pool,
                            typename GamePool<DataWriter>::WorkQueue::Receiver rx) {
    network::CpuEvaluator evaluator;

    while (!stop_flag.load(std::memory_order_relaxed)) {
        DatagenGame<DataWriter> *game = nullptr;
        if (!rx.try_pop(game)) {
            if (all_done(settings, next_game_idx, active_games)) {
                break;
            }
            std::this_thread::yield();
            continue;
        }

        const auto req = game->searcher.poll();
        vine_assert(req);
        const auto &state = game->searcher.pending_state();
        auto &tree = game->searcher.game_tree();
        const auto node = tree.node_at(req->node);
        policy_batch_items.fetch_add(1, std::memory_order_relaxed);
        policy_moves.fetch_add(node.info.num_children, std::memory_order_relaxed);

        auto ctx = evaluator.policy_context(state);
        for (u16 i = 0; i < node.info.num_children; ++i) {
            const auto child = tree.node_at(node.info.first_child_idx + i);
            ctx.enqueue(child.info.move, state.get_piece_type(child.info.move.from()));
        }
        ctx.ready();
        game->searcher.finish_policy([&] { return ctx.logit(); });
        game->resume = true;
        push_until(pool.search, game);
    }
}

template <class DataWriter>
bool times_up(DatagenGame<DataWriter> &game) {
    auto &tree = game.searcher.game_tree();
    ++game.iterations;

    const u64 depth = tree.sum_depths() / game.iterations;
    game.previous_depth = std::max(game.previous_depth, depth);

    util::StaticVector<u32, MAX_MOVES> new_visit_dist;
    for (u16 i = 0; i < tree.root().info.num_children; ++i) {
        new_visit_dist.push_back(tree.node_at(tree.root().info.first_child_idx + i).num_visits);
    }

    const bool stop = game.time_manager.times_up(tree, game.iterations, game.board.state().side_to_move,
                                                 game.previous_depth, game.old_visit_dist, new_visit_dist);
    game.old_visit_dist = new_visit_dist;
    return stop;
}

template <class DataWriter>
void advance_search(DatagenGame<DataWriter> *game, const Settings &settings, std::atomic_size_t &active_games,
                    GamePool<DataWriter> &pool, std::mutex &out_mutex,
                    typename GamePool<DataWriter>::WorkQueue::Sender &value_tx,
                    typename GamePool<DataWriter>::WorkQueue::Sender &policy_tx) {
    if (game->resume) {
        game->searcher.ready();
        game->resume = false;
        if (times_up(*game)) {
            (void)finish_move(game, settings, active_games, pool, out_mutex, true);
            return;
        }
    }

    const auto request = game->searcher.poll(game->board);

    switch (request.kind) {
    case search::RequestKind::Value:
        if (game->searcher.game_tree().node_at(request.node).terminal()) {
            game->searcher.finish_value(
                static_cast<f32>(game->searcher.game_tree().node_at(request.node).info.terminal_state.score()));
            game->resume = true;
            push_until(pool.search, game);
        } else {
            push_until(value_tx, game);
        }
        break;
    case search::RequestKind::Policy:
        push_until(policy_tx, game);
        break;
    }
}

template <class DataWriter>
void search_thread_loop(const Settings &settings, std::atomic_size_t &next_game_idx, std::atomic_size_t &active_games,
                        GamePool<DataWriter> &pool, std::mutex &out_mutex, const std::vector<std::string> &opening_fens,
                        typename GamePool<DataWriter>::WorkQueue::Sender value_tx,
                        typename GamePool<DataWriter>::WorkQueue::Sender policy_tx) {
    rng::seed(std::random_device{}());

    while (!stop_flag.load(std::memory_order_relaxed)) {
        DatagenGame<DataWriter> *game = nullptr;
        if (pool.search.try_pop(game)) {
            advance_search(game, settings, active_games, pool, out_mutex, value_tx, policy_tx);
            continue;
        }

        game = next_game(settings, next_game_idx, active_games, pool, opening_fens);
        if (!game) {
            if (all_done(settings, next_game_idx, active_games)) {
                break;
            }
            std::this_thread::yield();
            continue;
        }

        advance_search(game, settings, active_games, pool, out_mutex, value_tx, policy_tx);
    }
}

template <class DataWriter>
void cpu_thread_loop(const Settings &settings, std::atomic_size_t &next_game_idx, std::atomic_size_t &active_games,
                     GamePool<DataWriter> &pool, std::mutex &out_mutex, const std::vector<std::string> &opening_fens) {
    rng::seed(std::random_device{}());

    while (!stop_flag.load(std::memory_order_relaxed)) {
        auto *game = next_game(settings, next_game_idx, active_games, pool, opening_fens);
        if (!game) {
            if (all_done(settings, next_game_idx, active_games)) {
                break;
            }
            std::this_thread::yield();
            continue;
        }

        while (!stop_flag.load(std::memory_order_relaxed)) {
            game->searcher.go(game->board, settings.time_settings);
            if (!finish_move(game, settings, active_games, pool, out_mutex, false)) {
                break;
            }
        }
    }
}

template <class DataWriter>
void launch_gpu_threads(const Settings &settings, std::ostream &final_output,
                        const std::vector<std::string> &opening_fens, std::vector<std::thread> &threads) {
    auto pool = std::make_shared<GamePool<DataWriter>>();
    pool->reset(searcher_count(settings), settings.num_threads, final_output);
    auto next_game_idx = std::make_shared<std::atomic_size_t>(0);
    auto active_games = std::make_shared<std::atomic_size_t>(0);
    auto out_mutex = std::make_shared<std::mutex>();
    auto value_fill_mutex = std::make_shared<std::mutex>();
    auto policy_fill_mutex = std::make_shared<std::mutex>();

    const usize gpu_workers = std::max<usize>(1, settings.gpu_workers_per_queue);

    for (usize i = 0; i < gpu_workers; ++i) {
        threads.emplace_back(
            [settings, pool, next_game_idx, active_games, value_fill_mutex, rx = pool->value.receiver()]() mutable {
                value_thread_loop_gpu<DataWriter>(settings, *next_game_idx, *active_games, *pool, std::move(rx),
                                                  *value_fill_mutex);
            });
    }
    for (usize i = 0; i < gpu_workers; ++i) {
        threads.emplace_back(
            [settings, pool, next_game_idx, active_games, policy_fill_mutex, rx = pool->policy.receiver()]() mutable {
                policy_thread_loop_gpu<DataWriter>(settings, *next_game_idx, *active_games, *pool, std::move(rx),
                                                   *policy_fill_mutex);
            });
    }

    for (usize i = 0; i < settings.num_threads; ++i) {
        threads.emplace_back([settings, pool, next_game_idx, active_games, out_mutex, &opening_fens,
                              value_tx = pool->value.sender(), policy_tx = pool->policy.sender()]() mutable {
            search_thread_loop<DataWriter>(settings, *next_game_idx, *active_games, *pool, *out_mutex, opening_fens,
                                           std::move(value_tx), std::move(policy_tx));
        });
    }
}

template <class DataWriter>
void launch_cpu_threads(const Settings &settings, std::ostream &final_output,
                        const std::vector<std::string> &opening_fens, std::vector<std::thread> &threads) {
    auto pool = std::make_shared<GamePool<DataWriter>>();
    pool->reset(searcher_count(settings), settings.num_threads, final_output);
    auto next_game_idx = std::make_shared<std::atomic_size_t>(0);
    auto active_games = std::make_shared<std::atomic_size_t>(0);
    auto out_mutex = std::make_shared<std::mutex>();
    for (usize i = 0; i < settings.num_threads; ++i) {
        threads.emplace_back([settings, pool, next_game_idx, active_games, out_mutex, &opening_fens]() mutable {
            cpu_thread_loop<DataWriter>(settings, *next_game_idx, *active_games, *pool, *out_mutex, opening_fens);
        });
    }
}

} // namespace

void run_games(Settings settings, std::ostream &out) {
    stop_flag = false;
    games_played = 0;
    active_games_live = 0;
    positions_played = 0;
    positions_written = 0;
    value_batches = 0;
    value_batch_items = 0;
    policy_batches = 0;
    policy_batch_items = 0;
    policy_moves = 0;

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
        usize last_positions_played = 0;
        usize last_value_evals = 0;
        usize last_policy_evals = 0;
        auto last_time = std::chrono::steady_clock::now();
        bool printed = false;

        while (!stop_flag.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(5));

            auto current_time = std::chrono::steady_clock::now();
            const f64 elapsed_sec = std::chrono::duration<f64>(current_time - last_time).count();

            const usize current_games = games_played.load();
            const usize current_active_games = active_games_live.load(std::memory_order_relaxed);
            const usize current_positions_played = positions_played.load();
            const usize current_positions_written = positions_written.load();
            const usize current_active_positions = current_positions_played - current_positions_written;
            const usize current_value_evals = value_batch_items.load(std::memory_order_relaxed);
            const usize current_policy_evals = policy_batch_items.load(std::memory_order_relaxed);
            const usize total_games = settings.num_games;
            const f64 games_per_sec = (current_games - last_games) / elapsed_sec;
            const f64 positions_per_sec = (current_positions_played - last_positions_played) / elapsed_sec;
            const f64 value_evals_per_sec = (current_value_evals - last_value_evals) / elapsed_sec;
            const f64 policy_evals_per_sec = (current_policy_evals - last_policy_evals) / elapsed_sec;
            const f64 remaining_games = total_games > current_games ? total_games - current_games : 0;
            const f64 eta_sec = games_per_sec > 0.0 ? remaining_games / games_per_sec : 0.0;
            const usize eta = eta_sec;
            const usize eta_min = eta / 60;
            const usize eta_hour = eta_min / 60;
            const usize eta_rem_min = eta_min % 60;
            const usize eta_rem_sec = eta % 60;
            const auto current_value_batches = value_batches.load(std::memory_order_relaxed);
            const auto current_value_batch_items = value_batch_items.load(std::memory_order_relaxed);
            const auto current_policy_batches = policy_batches.load(std::memory_order_relaxed);
            const auto current_policy_batch_items = policy_batch_items.load(std::memory_order_relaxed);
            const auto current_policy_moves = policy_moves.load(std::memory_order_relaxed);
            const f64 average_value_batch_size =
                current_value_batches == 0 ? 0.0 : static_cast<f64>(current_value_batch_items) / current_value_batches;
            const f64 average_policy_batch_size =
                current_policy_batches == 0 ? 0.0
                                            : static_cast<f64>(current_policy_batch_items) / current_policy_batches;
            const f64 average_policy_moves = current_policy_batch_items == 0
                                                 ? 0.0
                                                 : static_cast<f64>(current_policy_moves) / current_policy_batch_items;
            const f64 average_active_game_length =
                current_active_games == 0 ? 0.0 : static_cast<f64>(current_active_positions) / current_active_games;

            if (printed) {
                util::tui::clear_lines(out, 11);
            }

            out << "current speed:\n";
            out << "  positions played  : " << current_positions_played << " (" << positions_per_sec << " pos/s)\n";
            out << "  evals             : value " << value_evals_per_sec << "/s, policy " << policy_evals_per_sec
                << "/s\n";
            out << "  avg batch size    : value " << average_value_batch_size << ", policy "
                << average_policy_batch_size << '\n';
            out << "  avg policy moves  : " << average_policy_moves << '\n';
            out << "  active games      : " << current_active_games << ", active positions " << current_active_positions
                << " (" << average_active_game_length << " avg/game)\n";
            out << "progress update:\n";
            out << "  games played      : " << current_games << " / " << total_games << '\n';
            out << "  positions written : " << current_positions_written << '\n';
            out << "  throughput        : " << games_per_sec << " games/s\n";
            out << "  eta               : " << eta_hour << "h " << eta_rem_min << "m " << eta_rem_sec << "s\n";
            out.flush();

            last_games = current_games;
            last_positions_played = current_positions_played;
            last_value_evals = current_value_evals;
            last_policy_evals = current_policy_evals;
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
                launch_cpu_threads<ViriformatWriter>(settings, final_output, opening_fens, threads);
            } else {
                launch_cpu_threads<MontyFormatWriter>(settings, final_output, opening_fens, threads);
            }
            break;
#ifdef DATAGEN_CUDA
        case EvaluatorBackend::GPU:
            if (settings.mode == DatagenMode::value) {
                launch_gpu_threads<ViriformatWriter>(settings, final_output, opening_fens, threads);
            } else {
                launch_gpu_threads<MontyFormatWriter>(settings, final_output, opening_fens, threads);
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
