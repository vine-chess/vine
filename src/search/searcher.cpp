#include "searcher.hpp"
#include "../uci/uci.hpp"
#include "hash.hpp"
#include <iostream>

namespace search {

Searcher::Searcher() : verbosity_(Verbosity::VERBOSE) {
    set_thread_count(1);
}

void Searcher::set_thread_count(u16 thread_count) {
    threads_.clear();
    threads_.resize(thread_count);
}

void Searcher::set_hash_size(u32 size_in_mb) {
    const u64 bytes_in_mb = 1024 * 1024;
    const u64 fraction_in_nodes = 15;
    game_tree_.set_node_capacity((bytes_in_mb * size_in_mb * fraction_in_nodes / 16) / sizeof(Node));
    game_tree_.set_hash_capacity((bytes_in_mb * size_in_mb * (16 - fraction_in_nodes) / 16) / sizeof(HashEntry));
}

void Searcher::set_verbosity(Verbosity verbosity) {
    verbosity_ = verbosity;
}

void Searcher::go(Board &board, const TimeSettings &time_settings) {
    for (auto &thread : threads_) {
        thread.go(game_tree_, board, time_settings, verbosity_);
    }
}

const GameTree &Searcher::game_tree() const {
    return game_tree_;
}

u64 Searcher::iterations() const {
    u64 result = 0;
    for (const auto &thread : threads_) {
        result += thread.iterations();
    }
    return result;
}

} // namespace search
