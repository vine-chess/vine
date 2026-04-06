#ifndef SEARCH_HPP
#define SEARCH_HPP

#include "../chess/board.hpp"
#include "game_tree.hpp"
#include "info.hpp"
#include "thread.hpp"
#include "time_manager.hpp"
#include <optional>

namespace search {

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
    Request poll(Board &board);

    [[nodiscard]] GameTree &game_tree();
    [[nodiscard]] const GameTree &game_tree() const;
    [[nodiscard]] const BoardState &pending_state() const;
    [[nodiscard]] u64 iterations() const;
    [[nodiscard]] const std::optional<Request> &poll() const;

    void clear();
    void ready();
    void finish_value(f32 score);
    template <class NextLogit>
    void finish_policy(NextLogit &&next_logit);

  private:
    std::vector<Thread> threads_;
    GameTree game_tree_;
    network::CpuEvaluator cpu_evaluator_;
    Verbosity verbosity_;
    std::optional<Request> request_;
    std::optional<f32> value_result_;
    bool initialized_ = false;
};

} // namespace search

template <class Evaluator>
void search::Searcher::go(Board &board, Evaluator &evaluator, const TimeSettings &time_settings) {
    game_tree_.set_use_gini(use_gini_option_enabled());
    for (auto &thread : threads_) {
        thread.go(game_tree_, evaluator, board, time_settings, verbosity_);
    }
}

inline search::Request search::Searcher::poll(Board &board) {
    if (!initialized_) {
        game_tree_.set_use_gini(use_gini_option_enabled());
        game_tree_.new_search(board);
        initialized_ = true;

        if (const auto root = game_tree_.root(); root.expanded() && root.info.terminal_state.is_none()) {
            request_ = {.kind = RequestKind::Policy, .node = game_tree_.root_idx()};
            return *request_;
        }
    }

    vine_assert(!request_);
    request_ = game_tree_.select_and_expand_node();
    return *request_;
}

template <class NextLogit>
void search::Searcher::finish_policy(NextLogit &&next_logit) {
    vine_assert(request_ && request_->kind == RequestKind::Policy);
    game_tree_.apply_policy(request_->node, std::forward<NextLogit>(next_logit));
    game_tree_.reset_to_root();
    request_.reset();
}

#endif // SEARCH_HPP
