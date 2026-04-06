#ifndef GAME_RUNNER_HPP
#define GAME_RUNNER_HPP

#include "../search/searcher.hpp"
#include "../util/types.hpp"
#include "openings.hpp"

namespace datagen {

enum class EvaluatorBackend : u8 {
    CPU,
    GPU,
};

enum class DatagenMode : u8 {
    policy,
    value,
};

struct Settings {
    // Number of random moves to play for the opening
    usize random_moves;
    // Number of games to play
    usize num_games;
    // Number of threads to run games simultaneously
    usize num_threads;
    // Total memory budget in MB for all live searchers
    usize total_memory = 0;
    // Number of value threads and number of policy threads for queued GPU eval
    usize gpu_workers_per_queue = 2;
    // Hash size per searcher
    usize hash_size;
    // Time management settings for move stop conditions
    search::TimeSettings time_settings;
    // The output file location
    std::string output_file;
    // File path to the book
    std::string book_path;
    // Opening settings
    f64 temperature, gamma;
    // Evaluator backend
    EvaluatorBackend evaluator_backend = EvaluatorBackend::CPU;
    // Whether to do value or policy datagen
    DatagenMode mode = DatagenMode::policy;
};

void run_games(Settings settings, std::ostream &out);

} // namespace datagen

#endif // GAME_RUNNER_HPP
