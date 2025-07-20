#include "thread.hpp"
#include <iostream>

namespace search {

Thread::Thread() {
    // raw_thread_ = std::thread(&Thread::thread_loop, this);
}

Thread::~Thread() {
    /* if (raw_thread_.joinable()) {
        raw_thread_.join();
    } */
}

void Thread::thread_loop() {
    // TODO: this shit
}

void Thread::go(Board &board, const TimeSettings &time_settings) {
    time_manager_.start_tracking(time_settings);

    while (!time_manager_.times_up(board.state().side_to_move)) {

    }
}

} // namespace search