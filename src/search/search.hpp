#ifndef SEARCH_HPP
#define SEARCH_HPP

#include "../chess/board.hpp"
#include "node.hpp"
#include "thread.hpp"
#include "time_manager.hpp"

namespace search {

class Searcher {
  public:
    Searcher();
    ~Searcher() = default;

    void set_thread_count(u16 thread_count);
    void go(Board &board, const TimeSettings &time_settings = {});

    inline u64 get_iterations() const {
        u64 result = 0;
        for (const auto &thread : threads_) {
            result += thread.get_iterations();
        }
        return result;
    }

  private:
    std::vector<Thread> threads_;
    std::vector<Node> tree_;
};

} // namespace search

#endif // SEARCH_HPP
