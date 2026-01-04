#ifndef NODE_HPP
#define NODE_HPP

#include "../chess/move.hpp"
#include "node_index.hpp"
#include <strings.h>

namespace search {

class TerminalState {
  public:
    enum class Flag : u8 {
        NONE,
        WIN,
        DRAW,
        LOSS,
    };
    constexpr TerminalState() {}

    constexpr explicit TerminalState(u16 value) : value_(value) {}

    [[nodiscard]] constexpr static TerminalState none() {
        return TerminalState(static_cast<u8>(Flag::NONE) << 8);
    }

    [[nodiscard]] constexpr static TerminalState draw() {
        return TerminalState(static_cast<u8>(Flag::DRAW) << 8);
    }

    [[nodiscard]] constexpr static TerminalState draw(u8 ply) {
        return TerminalState(static_cast<u8>(Flag::DRAW) << 8 | ply);
    }

    [[nodiscard]] constexpr static TerminalState win(u8 ply) {
        return TerminalState(static_cast<u8>(Flag::WIN) << 8 | ply);
    }

    [[nodiscard]] constexpr static TerminalState loss(u8 ply) {
        return TerminalState(static_cast<u8>(Flag::LOSS) << 8 | ply);
    }

    [[nodiscard]] constexpr bool is_none() const {
        return flag() == Flag::NONE;
    }

    [[nodiscard]] constexpr bool is_draw() const {
        return flag() == Flag::DRAW;
    }

    [[nodiscard]] constexpr bool is_win() const {
        return flag() == Flag::WIN;
    }

    [[nodiscard]] constexpr bool is_loss() const {
        return flag() == Flag::LOSS;
    }

    [[nodiscard]] constexpr f32 score() const {
        switch (flag()) {
        case Flag::LOSS:
            return 0.0f;
        case Flag::WIN:
            return 1.0f;
        default:
            return 0.5f;
        }
    }

    [[nodiscard]] constexpr TerminalState operator-() const {
        switch (flag()) {
        case Flag::LOSS:
            return TerminalState::win(distance_to_terminal() + 1);
        case Flag::WIN:
            return TerminalState::loss(distance_to_terminal() + 1);
        default:
            return *this;
        }
    }

    [[nodiscard]] constexpr bool operator==(const TerminalState &other) const = default;

    [[nodiscard]] Flag flag() const {
        return static_cast<Flag>(value_ >> 8);
    }

    [[nodiscard]] u8 distance_to_terminal() const {
        return static_cast<u8>(value_ & 255);
    }

  private:
    u16 value_;
};

struct Node {
    // Sum of all scores that have been propagated back to this node
    f64 sum_of_scores = 0.0;
    // Policy given to us by our parent node
    f32 policy_score = 0.0;
    // Number of times this node has been visited
    u32 num_visits = 0;
    // Index of the first child in the node table
    NodeIndex first_child_idx = NodeIndex::none();
    // Move that led into this node
    Move move = Move::null();
    // Number of legal moves this node has
    u16 num_children = 0;
    // What kind of state this (terminal) node is
    TerminalState terminal_state = TerminalState::none();
    // A measure of the entropy of the policy distribution
    u8 gini_impurity = 0;
    [[nodiscard]] bool visited() const;
    [[nodiscard]] bool terminal() const;
    [[nodiscard]] bool expanded() const;
    // Average of all scores this node has received
    [[nodiscard]] f64 q() const;
};

struct NodeInfo {
    // Index of the first child in the node table
    NodeIndex first_child_idx = NodeIndex::none();
    // Move that led into this node
    Move move = Move::null();
    // Number of legal moves this node has
    u16 num_children = 0;
    // What kind of state this (terminal) node is
    TerminalState terminal_state = TerminalState::none();
    // A measure of the entropy of the policy distribution
    u8 gini_impurity = 0;

    NodeInfo() {}

    NodeInfo(Node x)
        : first_child_idx(x.first_child_idx), move(x.move), num_children(x.num_children),
          terminal_state(x.terminal_state), gini_impurity(x.gini_impurity) {}
};

struct NodeReference {
    NodeInfo &info;
    f64 &sum_of_scores;
    f32 &policy_score;
    u32 &num_visits;
    [[nodiscard]] bool visited() const;
    [[nodiscard]] bool terminal() const;
    [[nodiscard]] bool expanded() const;
    // Average of all scores this node has received
    [[nodiscard]] f64 q() const;

    NodeReference(NodeInfo &info, f64 &sum_of_scores, f32 &policy_score, u32 &num_visits)
        : info(info), sum_of_scores(sum_of_scores), policy_score(policy_score), num_visits(num_visits) {}

    operator Node() const {
        return Node{
            .sum_of_scores = sum_of_scores,
            .policy_score = policy_score,
            .num_visits = num_visits,
            .first_child_idx = info.first_child_idx,
            .move = info.move,
            .num_children = info.num_children,
            .terminal_state = info.terminal_state,
            .gini_impurity = info.gini_impurity,
        };
    }
};

struct NodeRangeSentinel {
    u32 end_idx;
};

struct NodeIterator {
    f64 *score_sums_;
    f32 *policy_scores_;
    u32 *visit_counts_;
    NodeInfo *nodes_;
    u32 index;

    NodeIterator operator++() {
        index++;
        return *this;
    }

    [[nodiscard]] NodeReference operator[](u32 i) {
        return NodeReference{
            nodes_[index + i],
            score_sums_[index + i],
            policy_scores_[index + i],
            visit_counts_[index + i],
        };
    }

    [[nodiscard]] NodeReference operator*() {
        return operator[](0);
    }

    bool operator==(NodeRangeSentinel sent) const {
        return index == sent.end_idx;
    }
};

struct NodeRange {
    NodeIterator begin_;
    NodeRangeSentinel end_;

    NodeIterator begin() const {
        return begin_;
    }

    NodeReference operator[](u32 i) {
        return begin_[i];
    }

    NodeRangeSentinel end() const {
        return end_;
    }
};

} // namespace search

#endif // NODE_HPP
