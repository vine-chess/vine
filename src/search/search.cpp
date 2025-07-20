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
    for (auto &thread : threads_) {
        thread.go(board, time_settings);
    }
}

} // namespace search