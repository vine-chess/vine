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

    [[nodiscard]] u64 iterations() const;

  private:
    std::vector<Thread> threads_;
    std::vector<Node> tree_;
};

} // namespace search

#endif // SEARCH_HPP
