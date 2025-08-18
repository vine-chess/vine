#ifndef THREAD_HPP
#define THREAD_HPP

#include "../chess/board.hpp"
#include "game_tree.hpp"
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

    void go(GameTree &tree, const Board &board, const TimeSettings &time_settings);

    [[nodiscard]] u64 iterations() const;

  private:
    void thread_loop();

    void write_info(GameTree &tree, u64 nodes, bool write_bestmove = false) const;

    std::thread raw_thread_;
    TimeManager time_manager_;
    u64 num_iterations_;
};

} // namespace search

#endif // THREAD_HPP
