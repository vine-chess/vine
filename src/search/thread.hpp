#ifndef THREAD_HPP
#define THREAD_HPP

#include "../chess/board.hpp"
#include "node.hpp"
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

    void go(std::vector<Node> &tree, Board &board, const TimeSettings &time_settings);

  private:
    [[nodiscard]] u32 select_node(std::vector<Node> &tree);

    void expand_node(u32 node_idx, std::vector<Node> &tree);

    [[nodiscard]] f64 simulate_node([[maybe_unused]] u32 node_idx, [[maybe_unused]] std::vector<Node> &tree);

    void backpropagate(f64 score, u32 node_idx, std::vector<Node> &tree);

    void thread_loop();

    void write_info(std::vector<Node> &tree, Board &board, u64 nodes, bool write_bestmove = false);

    std::thread raw_thread_;
    TimeManager time_manager_;
    Board board_;
    u64 num_iterations_;
    u64 sum_depth_;
};

} // namespace search

#endif // THREAD_HPP
