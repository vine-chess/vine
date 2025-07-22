#include "search.hpp"
#include "../uci/uci.hpp"
#include <iostream>

namespace search {

Searcher::Searcher() {
    set_thread_count(1);
}

void Searcher::set_thread_count(u16 thread_count) {
    threads_.clear();
    threads_.resize(thread_count);
}

void Searcher::set_hash_size(u32 size_in_mb) {
    tree_.reserve(1024 * 1024 * size_in_mb / sizeof(Node));
    std::cout << tree_.capacity() << std::endl;
}

void Searcher::go(Board &board, const TimeSettings &time_settings) {
    // TODO: use Hash option for this
    // tree_.reserve(1024 * 1024);

    for (auto &thread : threads_) {
        thread.go(tree_, board, time_settings);
    }
}

u64 Searcher::iterations() const {
    u64 result = 0;
    for (const auto &thread : threads_) {
        result += thread.iterations();
    }
    return result;
}
} // namespace search
