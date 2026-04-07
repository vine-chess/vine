#include "searcher.hpp"
#include "../uci/uci.hpp"
#include <iostream>

namespace search {

bool use_gini_option_enabled() {
    return std::get<bool>(uci::options.get("UseGiniImpurity")->value_as_variant());
}

Searcher::Searcher() : verbosity_(Verbosity::VERBOSE) {
    set_thread_count(1);
    clear();
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
    game_tree_.set_use_gini(use_gini_option_enabled());
    go(board, cpu_evaluator_, time_settings);
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

const BoardState &Searcher::pending_state() const {
    return game_tree_.state();
}

const std::optional<Request> &Searcher::poll() const {
    return request_;
}

void Searcher::clear() {
    game_tree_.clear();
    request_.reset();
    value_result_.reset();
    initialized_ = false;
}

void Searcher::restart() {
    request_.reset();
    value_result_.reset();
    initialized_ = false;
}

void Searcher::ready() {
    if (value_result_) {
        game_tree_.backpropagate_score(*value_result_);
        value_result_.reset();
    }
}

void Searcher::finish_value(f32 score) {
    vine_assert(request_ && request_->kind == RequestKind::Value);
    value_result_ = score;
    request_.reset();
}

} // namespace search
