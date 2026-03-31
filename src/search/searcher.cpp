#include "searcher.hpp"
#include "../uci/uci.hpp"
#include "game_tree.hpp"
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
    const usize size_in_bytes = 1024 * 1024 * size_in_mb;
    const usize hash_table_capacity = size_in_bytes / 25;
    game_tree_.set_node_capacity((size_in_bytes - hash_table_capacity) / sizeof(Node));
    game_tree_.set_hash_table_capacity(hash_table_capacity / sizeof(HashEntry));
}

void Searcher::set_verbosity(Verbosity verbosity) {
    verbosity_ = verbosity;
}

void Searcher::go(Board &board, const TimeSettings &time_settings) {
    for (auto &thread : threads_) {
        thread.go(game_tree_, board, time_settings, verbosity_);
    }
}

GameTree &Searcher::game_tree() {
    return game_tree_;
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

void Searcher::clear() {
    game_tree_.clear();
}

} // namespace search
