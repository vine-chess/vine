#include "search.hpp"
#include <iostream>

namespace search {

Searcher::Searcher() {
    set_thread_count(1);
}

void Searcher::set_thread_count(u16 thread_count) {
    threads_.clear();
    threads_.resize(thread_count);
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
