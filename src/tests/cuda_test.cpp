#include "../chess/board.hpp"
#include "../chess/move_gen.hpp"
#include "../eval/evaluator.hpp"
#include "../util/random.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

struct TestOptions {
    bool is_value = true;
    usize count = 4096;
    bool benchmark_only = false;
};

struct RunStats {
    usize total_policy_moves = 0;
    std::chrono::nanoseconds cpu_eval_time{0};
    std::chrono::nanoseconds gpu_eval_time{0};
};

constexpr f32 EPSILON = 1e-3f;
constexpr usize POSITION_PLIES = 20;

[[nodiscard]] u64 parse_u64(const char *arg, u64 fallback) {
    if (arg == nullptr) {
        return fallback;
    }

    char *end = nullptr;
    const auto value = std::strtoull(arg, &end, 10);
    return end != arg ? value : fallback;
}

[[nodiscard]] usize cpu_thread_count(usize item_count) {
    return std::min(std::max<usize>(1, std::thread::hardware_concurrency()), item_count);
}

template <class Fn>
void parallel_for_chunks(usize item_count, Fn fn) {
    if (item_count == 0) {
        return;
    }

    const usize worker_count = cpu_thread_count(item_count);
    if (worker_count == 1) {
        fn(0, item_count);
        return;
    }

    const usize chunk_size = (item_count + worker_count - 1) / worker_count;
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (usize worker = 0; worker < worker_count; ++worker) {
        const usize begin = worker * chunk_size;
        const usize end = std::min(item_count, begin + chunk_size);
        if (begin >= end) {
            break;
        }
        workers.emplace_back([=, &fn] { fn(begin, end); });
    }

    for (auto &worker : workers) {
        worker.join();
    }
}

[[nodiscard]] Move sample_random_move(const MoveList &moves) {
    return moves[rng::next_u64(0, moves.size() - 1)];
}

[[nodiscard]] BoardState generate_position(usize plies) {
    Board board(STARTPOS_FEN);

    while (true) {
        board = Board(STARTPOS_FEN);

        bool success = true;
        for (usize ply = 0; ply < plies; ++ply) {
            MoveList moves;
            generate_moves(board.state(), moves);
            if (moves.empty() || board.is_draw()) {
                success = false;
                break;
            }

            board.make_move(sample_random_move(moves));
        }

        if (!success) {
            continue;
        }

        MoveList moves;
        generate_moves(board.state(), moves);
        if (moves.empty() || board.is_draw()) {
            continue;
        }

        return board.state();
    }
}

void print_summary(const TestOptions &options, const RunStats &stats) {
    const f64 gpu_seconds = std::chrono::duration<f64>(stats.gpu_eval_time).count();
    const f64 gpu_positions_per_second = static_cast<f64>(options.count) / gpu_seconds;

    std::cout << "cuda " << (options.is_value ? "value" : "policy") << ' ';
    std::cout << (options.benchmark_only ? "benchmark completed for " : "test passed for ");
    std::cout << options.count << " positions";
    std::cout << " (benchmark_only=" << options.benchmark_only << ")\n";
    std::cout << "mode: direct batch kernel\n";
    std::cout << "gpu eval: " << gpu_seconds << " s, " << gpu_positions_per_second << " pos/s\n";

    if (!options.is_value) {
        const f64 gpu_moves_per_second = static_cast<f64>(stats.total_policy_moves) / gpu_seconds;
        std::cout << "gpu logits: " << stats.total_policy_moves << ", " << gpu_moves_per_second << " logits/s\n";
    }

    if (!options.benchmark_only) {
        const f64 cpu_seconds = std::chrono::duration<f64>(stats.cpu_eval_time).count();
        const f64 cpu_positions_per_second = static_cast<f64>(options.count) / cpu_seconds;
        std::cout << "cpu eval: " << cpu_seconds << " s, " << cpu_positions_per_second << " pos/s\n";

        if (!options.is_value) {
            const f64 cpu_moves_per_second = static_cast<f64>(stats.total_policy_moves) / cpu_seconds;
            std::cout << "cpu logits: " << stats.total_policy_moves << ", " << cpu_moves_per_second << " logits/s\n";
        }

        std::cout << "speedup: " << cpu_seconds / gpu_seconds << "x\n";
    }
}

[[nodiscard]] bool run_value_test(const TestOptions &options) {
    RunStats stats;
    std::vector<BoardState> states(options.count);
    std::vector<network::value::CudaBoardInput> inputs(options.count);
    std::vector<f32> actual(options.count, 0.0f);
    std::vector<f32> expected(options.benchmark_only ? 0 : options.count, 0.0f);

    std::cout << "generating " << options.count << " positions\n";
    for (usize i = 0; i < options.count; ++i) {
        states[i] = generate_position(POSITION_PLIES + (i & 1));
        inputs[i].pieces.compress(states[i].piece_type_on_sq);
        inputs[i].side_to_move = static_cast<u8>(states[i].side_to_move);
    }

    if (!options.benchmark_only) {
        const auto cpu_start = std::chrono::high_resolution_clock::now();
        const network::CpuEvaluator cpu_evaluator;
        parallel_for_chunks(options.count, [&](usize begin, usize end) {
            for (usize i = begin; i < end; ++i) {
                expected[i] = static_cast<f32>(cpu_evaluator.value(states[i]));
            }
        });
        stats.cpu_eval_time +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - cpu_start);
    }

    const auto gpu_start = std::chrono::high_resolution_clock::now();
    network::value::evaluate_many(inputs.data(), actual.data(), options.count);
    stats.gpu_eval_time +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - gpu_start);

    if (!options.benchmark_only) {
        f32 max_err = 0.0f;
        f64 sum_err = 0.0;
        f64 sum_signed_err = 0.0;
        usize fail_count = 0;
        for (usize i = 0; i < options.count; ++i) {
            const f32 diff = actual[i] - expected[i];
            const f32 err = std::abs(diff);
            if (err > max_err)
                max_err = err;
            sum_err += err;
            sum_signed_err += diff;
            if (err > EPSILON)
                ++fail_count;
        }
        const f64 n = static_cast<f64>(options.count);
        std::cerr << "max error: " << max_err << ", mean abs error: " << sum_err / n
                  << ", mean signed error: " << sum_signed_err / n << ", failures (>" << EPSILON << "): " << fail_count
                  << '/' << options.count << '\n';
        if (fail_count > 0) {
            return false;
        }
    }

    print_summary(options, stats);
    return true;
}

[[nodiscard]] bool run_policy_test(const TestOptions &options) {
    RunStats stats;
    std::vector<BoardState> states(options.count);
    std::vector<network::policy::CudaPolicyInput> inputs(options.count);
    std::vector<MoveList> move_lists(options.count);
    std::vector<u16> move_indices;

    std::cout << "generating " << options.count << " positions\n";
    for (usize i = 0; i < options.count; ++i) {
        states[i] = generate_position(POSITION_PLIES + (i & 1));
        inputs[i].pieces.compress(states[i].piece_type_on_sq);
        inputs[i].side_to_move = static_cast<u8>(states[i].side_to_move);
        inputs[i].move_offset = static_cast<u32>(move_indices.size());

        MoveList moves;
        generate_moves(states[i], moves);
        move_lists[i] = moves;
        inputs[i].move_count = static_cast<u8>(moves.size());

        for (Move move : moves) {
            const PieceType moving_piece = states[i].piece_type_on_sq[move.from()].piece_type();
            move_indices.push_back(network::policy::move_output_idx(states[i], move, moving_piece));
        }
    }

    stats.total_policy_moves = move_indices.size();
    std::vector<f32> actual(move_indices.size(), 0.0f);
    std::vector<f32> expected(options.benchmark_only ? 0 : move_indices.size(), 0.0f);

    if (!options.benchmark_only) {
        const auto cpu_start = std::chrono::high_resolution_clock::now();
        const network::CpuEvaluator cpu_evaluator;
        parallel_for_chunks(options.count, [&](usize begin, usize end) {
            for (usize i = begin; i < end; ++i) {
                auto ctx = cpu_evaluator.policy_context(states[i]);
                for (usize move_idx = 0; move_idx < move_lists[i].size(); ++move_idx) {
                    const Move move = move_lists[i][move_idx];
                    const PieceType moving_piece = states[i].piece_type_on_sq[move.from()].piece_type();
                    ctx.enqueue(move, moving_piece);
                }
                ctx.ready();
                for (usize move_idx = 0; move_idx < move_lists[i].size(); ++move_idx) {
                    expected[inputs[i].move_offset + move_idx] = ctx.logit();
                }
            }
        });
        stats.cpu_eval_time +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - cpu_start);
    }

    const auto gpu_start = std::chrono::high_resolution_clock::now();
    network::policy::evaluate_many(inputs.data(), move_indices.data(), actual.data(), options.count);
    stats.gpu_eval_time +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - gpu_start);

    if (!options.benchmark_only) {
        for (usize i = 0; i < options.count; ++i) {
            for (usize move_idx = 0; move_idx < move_lists[i].size(); ++move_idx) {
                const usize flat_idx = inputs[i].move_offset + move_idx;
                if (std::abs(actual[flat_idx] - expected[flat_idx]) > EPSILON) {
                    const Move move = move_lists[i][move_idx];
                    std::cerr << "mismatch at position " << i << ", move " << move.to_string() << ": got "
                              << actual[flat_idx] << ", expected " << expected[flat_idx] << '\n';
                    std::cerr << "fen: " << states[i].to_fen() << '\n';
                    return false;
                }
            }
        }
    }

    print_summary(options, stats);
    return true;
}

} // namespace

int main(int argc, char **argv) {
    TestOptions options;

    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg == "--bench") {
            options.benchmark_only = true;
            continue;
        }
        if (arg == "value") {
            options.is_value = true;
            continue;
        }
        if (arg == "policy") {
            options.is_value = false;
            continue;
        }
        options.count = std::stoull(argv[i]);
    }

    rng::seed(0);
    return (options.is_value ? run_value_test(options) : run_policy_test(options)) ? 0 : 1;
}
