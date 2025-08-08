#include "game_tree.hpp"
#include "../chess/move_gen.hpp"
#include "../eval/policy_network.hpp"
#include "../eval/value_network.hpp"
#include "../util/assert.hpp"
#include "node.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <span>

namespace search {

constexpr f32 ROOT_SOFTMAX_TEMPERATURE = 3.5f;
constexpr f32 SOFTMAX_TEMPERATURE = 1.0f;
constexpr f32 ROOT_EXPLORATION_CONSTANT = 1.3f;
constexpr f32 EXPLORATION_CONSTANT = 1.0f;

GameTree::GameTree() {
    set_node_capacity(1);
}

void GameTree::set_node_capacity(usize node_capacity) {
    nodes_.reserve(node_capacity);
}

void GameTree::new_search(const Board &root_board) {
    nodes_.clear();
    nodes_.emplace_back(); // Root node always exists
    board_ = root_board;
    sum_depths_ = 0;
    vine_assert(expand_node(0));
}

const Node &GameTree::root() const {
    return node_at(0);
}

Node &GameTree::root() {
    return nodes_[0];
}

const Node &GameTree::node_at(u32 idx) const {
    return nodes_[idx];
}

[[nodiscard]] u32 GameTree::sum_depths() const {
    return sum_depths_;
}

std::pair<u32, bool> GameTree::select_and_expand_node() {
    // Lambda to compute the PUCT score for a given child node in MCTS
    // Arguments:
    // - parent: the parent node from which the child was reached
    // - child: the candidate child node being scored
    // - policy_score: the probability for this child being the best move
    // - exploration_constant: hyperparameter controlling exploration vs. exploitation
    const auto compute_puct = [&](Node &parent, Node &child, f32 policy_score, f32 exploration_constant) -> f64 {
        // Average value of the child from previous visits (Q value), flipped to match current node's perspective
        // If the node hasn't been visited, use the parent node's Q value
        const f64 q_value = child.num_visits > 0 ? 1.0 - child.q() : parent.q();
        // Uncertainty/exploration term (U value), scaled by the prior and parent visits
        const f64 u_value = exploration_constant * static_cast<f64>(policy_score) * std::sqrt(parent.num_visits) /
                            (1.0 + static_cast<f64>(child.num_visits));
        // Final PUCT score is exploitation (Q) + exploration (U)
        return q_value + u_value;
    };

    u32 node_idx = nodes_in_path_ = 0;
    while (true) {
        Node &node = nodes_[node_idx];

        // We limit expansion to the second visit for non-root nodes since the value of the node from the first visit
        // might have been bad enough that this node is likely to not get selected again
        if (node.num_visits == 1) {
            if (!expand_node(node_idx)) {
                return {0, false};
            }
        }

        // Return if we cannot go any further down the tree
        if (node.terminal() || !node.visited()) {
            sum_depths_ += nodes_in_path_ + 1;
            return {node_idx, true};
        }

        const bool q_declining = node_idx != 0 && (1.0 - nodes_[node.parent_idx].q()) - node.q() > 0.05;

        u32 best_child_idx = 0;
        f64 best_child_score = std::numeric_limits<f64>::min();
        for (u16 i = 0; i < node.num_children; ++i) {
            Node &child_node = nodes_[node.first_child_idx + i];

            // Track the child with the highest PUCT score
            const f64 child_score =
                compute_puct(node, child_node, child_node.policy_score,
                             node_idx == 0 ? ROOT_EXPLORATION_CONSTANT : EXPLORATION_CONSTANT - q_declining * 0.1);
            if (child_score > best_child_score) {
                best_child_idx = node.first_child_idx + i; // Store absolute index into nodes
                best_child_score = child_score;
            }
        }

        // Keep descending through the game nodes_ until we find a suitable node to expand
        node_idx = best_child_idx, ++nodes_in_path_;
        board_.make_move(nodes_[node_idx].move);
    }
}

void GameTree::compute_policy(u32 node_idx) {
    const Node &node = nodes_[node_idx];

    // We keep track of a policy context so that we only accumulate once per node
    const network::policy::PolicyContext ctx(board_.state());

    const bool root_node = node_idx == 0;
    const f32 temperature = root_node ? ROOT_SOFTMAX_TEMPERATURE : SOFTMAX_TEMPERATURE;

    f32 highest_policy = -std::numeric_limits<f32>::max();
    for (u16 i = 0; i < node.num_children; ++i) {
        Node &child = nodes_[node.first_child_idx + i];
        // Compute policy output for this move
        child.policy_score = ctx.logit(child.move) / temperature;
        // Keep track of highest policy so we can shift all the policy
        // values down to avoid precision loss from large exponents
        highest_policy = std::max(highest_policy, child.policy_score);
    }

    // Softmax the policy logits
    f32 sum_exponents = 0;
    for (u16 i = 0; i < node.num_children; ++i) {
        Node &child = nodes_[node.first_child_idx + i];
        const f32 exp_policy = std::exp(child.policy_score - highest_policy);
        sum_exponents += exp_policy;
        child.policy_score = exp_policy;
    }

    // Normalize into policy scores
    for (u16 i = 0; i < node.num_children; ++i) {
        Node &child = nodes_[node.first_child_idx + i];
        child.policy_score /= sum_exponents;
    }
}

bool GameTree::expand_node(u32 node_idx) {
    auto &node = nodes_[node_idx];
    if (node.expanded() || node.terminal()) {
        return true;
    }

    // We should only be expanding when the number of visits is one
    // This is due to the optimization of not expanding nodes whos children we don't know we'll need
    vine_assert(node_idx == 0 || node.num_visits == 1);

    if (board_.has_threefold_repetition() || board_.is_fifty_move_draw()) {
        node.terminal_state = TerminalState::draw();
        return true;
    }

    MoveList move_list;
    generate_moves(board_.state(), move_list);

    if (move_list.empty()) {
        node.terminal_state = board_.state().checkers != 0 ? TerminalState::loss(0) : TerminalState::draw();
        return true;
    }

    // Return early if we will run out of memory
    if (nodes_.size() + move_list.size() > nodes_.capacity()) {
        return false;
    }

    node.first_child_idx = nodes_.size();
    node.num_children = move_list.size();

    // Append all child nodes to the nodes with the move that leads to it
    for (const auto move : move_list) {
        nodes_.push_back(Node{
            .parent_idx = static_cast<i32>(node_idx),
            .move = move,
        });
    }

    // Compute and store policy values for all the child nodes
    compute_policy(node_idx);

    return true;
}

f64 GameTree::simulate_node(u32 node_idx) {
    const auto &node = nodes_[node_idx];
    if (node.terminal()) {
        return node.terminal_state.score();
    }

    return 1.0 / (1.0 + std::exp(-network::value::evaluate(board_.state())));
}

void GameTree::backpropagate_terminal_state(u32 node_idx, TerminalState child_terminal_state) {
    auto &node = nodes_[node_idx];
    switch (child_terminal_state.flag()) {
    case TerminalState::Flag::LOSS: { // If a child node is lost, then it's a win for us
        // Ensure that if we already had a shorter mate we preserve it
        const auto current_mate_distance =
            node.terminal_state.is_win() ? node.terminal_state.distance_to_terminal() : 255;
        node.terminal_state =
            TerminalState::win(std::min<u8>(current_mate_distance, child_terminal_state.distance_to_terminal() + 1));
        break;
    }
    case TerminalState::Flag::WIN: { // If a child node is won, it's a loss for us if all of its siblings are also won
        u8 longest_loss = 0;
        for (i32 i = 0; i < node.num_children; ++i) {
            const auto sibling_terminal_state = nodes_[node.first_child_idx + i].terminal_state;
            if (sibling_terminal_state.flag() != TerminalState::Flag::WIN) {
                return;
            }
            longest_loss = std::max(longest_loss, sibling_terminal_state.distance_to_terminal());
        }
        node.terminal_state = TerminalState::loss(longest_loss + 1);
        break;
    }
    default:
        break;
    }
}

void GameTree::backpropagate_score(f64 score, u32 node_idx) {
    auto child_terminal_state = TerminalState::none();
    while (node_idx != -1) {
        // A node's score is the average of all of its children's score
        auto &node = nodes_[node_idx];
        node.sum_of_scores += score;
        node.num_visits++;

        // If a terminal state from the child score exists, then we try to backpropagate it to this node
        if (!child_terminal_state.is_none()) {
            backpropagate_terminal_state(node_idx, child_terminal_state);
        }

        // If this node has a terminal state (either from backpropagation or it is terminal), we save it for the parent
        // node to try to use it
        if (!node.terminal_state.is_none()) {
            child_terminal_state = node.terminal_state;
        }

        // Travel up to the parent
        node_idx = node.parent_idx;
        // Negate the score to match the perspective of the node
        score = 1.0 - score;
    }
    // Undo all of the moves that were selected
    board_.undo_n_moves(nodes_in_path_);
}

} // namespace search
