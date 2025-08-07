#ifndef GAME_RUNNER_HPP
#define GAME_RUNNER_HPP

#include "../search/searcher.hpp"
#include "openings.hpp"

namespace datagen {

struct Settings {
    // Number of random moves to play for the opening
    usize random_moves;
    // Number of games to play
    usize num_games;
    // Number of threads to run games simultaneously
    usize num_threads;
    // Hash size per thread
    usize hash_size;
    // Time management settings for move stop conditions
    search::TimeSettings time_settings;
    // The output file location
    std::string output_file;
};

void run_games(Settings settings, std::ostream &out);

} // namespace datagen

#endif // GAME_RUNNER_HPP
