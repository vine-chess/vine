#ifndef SEARCH_HPP
#define SEARCH_HPP

#include "../chess/board.hpp"
#include "game_tree.hpp"
#include "info.hpp"
#include "thread.hpp"
#include "time_manager.hpp"

namespace search {

enum class RequestKind : u8 {
    None,
    Value,
    Policy,
};

struct Request {
    RequestKind kind = RequestKind::None;
    NodeIndex node = NodeIndex::none();
};

[[nodiscard]] bool use_gini_option_enabled();

class Searcher {
  public:
    Searcher();
    ~Searcher() = default;

    void set_thread_count(u16 thread_count);
    void set_hash_size(u32 size_in_mb);
    void set_verbosity(Verbosity verbosity);

    void go(Board &board, const TimeSettings &time_settings = {});
    template <class Evaluator>
    void go(Board &board, Evaluator &evaluator, const TimeSettings &time_settings);

    [[nodiscard]] GameTree &game_tree();
    [[nodiscard]] const GameTree &game_tree() const;
    [[nodiscard]] u64 iterations() const;
    [[nodiscard]] Request poll() const;

    void clear();
    void finish_value(f64 score);
    void finish_policy();

  private:
    std::vector<Thread> threads_;
    GameTree game_tree_;
    network::CpuEvaluator cpu_evaluator_;
    Verbosity verbosity_;
    Request request_;
};

} // namespace search

template <class Evaluator>
void search::Searcher::go(Board &board, Evaluator &evaluator, const TimeSettings &time_settings) {
    game_tree_.set_use_gini(use_gini_option_enabled());
    for (auto &thread : threads_) {
        thread.go(game_tree_, evaluator, board, time_settings, verbosity_);
    }
}

#endif // SEARCH_HPP
