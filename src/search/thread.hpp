#ifndef THREAD_HPP
#define THREAD_HPP

#include "../chess/board.hpp"
#include "time_manager.hpp"
#include <thread>

namespace search {

class Thread {
  public:
    Thread();
    ~Thread();

    Thread(const Thread &) = delete;
    Thread &operator=(const Thread &) = delete;

    Thread(Thread &&other) noexcept : raw_thread_(std::move(other.raw_thread_)) {}

    Thread &operator=(Thread &&other) noexcept {
        if (this != &other) {
            if (raw_thread_.joinable())
                raw_thread_.join();
            raw_thread_ = std::move(other.raw_thread_);
        }
        return *this;
    }

    void go(Board &board, const TimeSettings &time_settings);

    void thread_loop();

  private:
    TimeManager time_manager_;
    Board board_;
    std::thread raw_thread_;
};

} // namespace search

#endif // THREAD_HPP