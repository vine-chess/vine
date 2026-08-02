#ifndef SEARCH_HPP
#define SEARCH_HPP

#include "../chess/board.hpp"
#include "game_tree.hpp"
#include "info.hpp"
#include "thread.hpp"
#include "time_manager.hpp"

namespace search {

class Searcher {
  public:
    Searcher();
    ~Searcher() = default;

    void set_thread_count(u16 thread_count);
    void set_hash_size(u32 size_in_mb);
    void set_verbosity(Verbosity verbosity);

    void go(Board &board, const TimeSettings &time_settings = {});

    [[nodiscard]] GameTree &game_tree();
    [[nodiscard]] const GameTree &game_tree() const;
    [[nodiscard]] u64 iterations() const;

    void clear();

  private:
    std::vector<Thread> threads_;
    GameTree game_tree_;
    Verbosity verbosity_;
};

} // namespace search

#endif // SEARCH_HPP
