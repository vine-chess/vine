#include "game_tree.hpp"
#include "../chess/move_gen.hpp"
#include "../eval/policy_network.hpp"
#include "../eval/value_network.hpp"
#include "../uci/uci.hpp"
#include "../util/assert.hpp"
#include "../util/math.hpp"

#include "node.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <span>

namespace search {

#ifdef DATAGEN
constexpr f32 ROOT_SOFTMAX_TEMPERATURE = 3.5f;
#else
constexpr f32 ROOT_SOFTMAX_TEMPERATURE = 2.0f;
#endif
constexpr f32 SOFTMAX_TEMPERATURE = 1.0f;
#ifdef DATAGEN
constexpr f32 ROOT_EXPLORATION_CONSTANT = 2.0f;
#else
constexpr f32 ROOT_EXPLORATION_CONSTANT = 1.3f;
#endif
constexpr f32 EXPLORATION_CONSTANT = 1.0f;
constexpr f32 CPUCT_VISIT_SCALE = 8192.0f;
constexpr f32 CPUCT_VISIT_SCALE_DIVISOR = 8192.0f; // Not for tuning
constexpr f32 GINI_BASE = 0.5;
constexpr f32 GINI_MULTIPLIER = 1.5;
constexpr f32 GINI_MAXIMUM = 2.25;

GameTree::GameTree()
    : halves_({TreeHalf(TreeHalf::Index::LOWER), TreeHalf(TreeHalf::Index::UPPER)}),
      active_half_(TreeHalf::Index::LOWER) {
    set_node_capacity(1);
}

void GameTree::set_node_capacity(usize node_capacity) {
    for (auto &half : halves_) {
        half.set_node_capacity(node_capacity / 2);
    }
}

void GameTree::set_hash_table_capacity(usize capacity) {
    hash_table_.set_entry_capacity(capacity);
}

void GameTree::new_search(const Board &root_board) {
    dirichlet_epsilon_ = std::get<i32>(uci::options.get("DirichletNoiseEpsilon")->value_as_variant()) / 100.0;
    dirichlet_alpha_ = std::get<i32>(uci::options.get("DirichletNoiseAlpha")->value_as_variant()) / 100.0;

    if (advance_root_node(board_, root_board, active_half().root_idx())) {
        // Re-compute root policy scores, since the node we advanced to was searched with non-root parameters
        compute_policy(root_board.state(), active_half().root_idx());
    } else {
        active_half().clear();
        active_half().push_node(Node{});
    }

    board_ = root_board;
    sum_depths_ = 0;
    tree_usage_ = 0;

    // Ensure the root node is expanded
    if (!expand_node(active_half().root_idx())) {
        flip_halves();
        vine_assert(expand_node(active_half().root_idx()));
    }

    // Add Dirichlet noise to the policy prior distributions of the root node during data generation
    if (dirichlet_alpha_ != 0 && dirichlet_epsilon_ != 0) {
        rng::seed_generator(root_board.state().hash_key);
        inject_dirichlet_noise(active_half().root_idx());
    }
}

const Node &GameTree::root() const {
    return active_half().root_node();
}

Node &GameTree::root() {
    return active_half().root_node();
}

Node &GameTree::node_at(NodeIndex idx) {
    return halves_[idx.half()][idx.index()];
}

const Node &GameTree::node_at(NodeIndex idx) const {
    return halves_[idx.half()][idx.index()];
}

[[nodiscard]] u32 GameTree::sum_depths() const {
    return sum_depths_;
}

u64 GameTree::tree_usage() const {
    return tree_usage_;
}

NodeIndex GameTree::select_and_expand_node() {
    // Lambda to compute the PUCT score for a given child node in MCTS
    // Arguments:
    // - parent: the parent node from which the child was reached
    // - child: the candidate child node being scored
    // - policy_score: the probability for this child being the best move
    // - exploration_constant: hyperparameter controlling exploration vs. exploitation
    const auto compute_puct = [&](Node &parent, Node &child, f32 exploration_constant) -> f64 {
        auto &history_entry = butterfly_table_[board_.state().side_to_move][child.move.from()][child.move.to()];
        // Average value of the child from previous visits (Q value), flipped to match current node's perspective
        // If the node hasn't been visited, use the parent node's Q value
        const f64 q_value = child.num_visits > 0 ? 1.0 - child.q() : parent.q();
        // Uncertainty/exploration term (U value), scaled by the prior and parent visits
        const f64 u_value = exploration_constant * static_cast<f64>(child.policy_score) * std::sqrt(parent.num_visits) /
                            (1.0 + static_cast<f64>(child.num_visits));
        // Final PUCT score is exploitation (Q) + exploration (U)
        return q_value + u_value;
    };

    NodeIndex node_idx = active_half().root_idx();
    nodes_in_path_.clear();
    nodes_in_path_.push_back(node_idx);

    const auto flip_and_restart = [&] {
        flip_halves();
        board_.undo_n_moves(nodes_in_path_.size() - 1);
        nodes_in_path_.clear();
        nodes_in_path_.push_back(node_idx = active_half().root_idx());
    };

    while (true) {
        Node &node = node_at(node_idx);

        // We don't expand on the first visit for non-root nodes since the value of the node from the first visit
        // might have been bad enough that this node is likely to not get selected again
        if (node.num_visits > 0) {
            if (!expand_node(node_idx)) {
                flip_and_restart();
                continue;
            }
        }

        // Return if we cannot go any further down the tree
        if (node.terminal() || !node.visited()) {
            sum_depths_ += nodes_in_path_.size();
            return node_idx;
        }

        if (!fetch_children(node_idx)) {
            flip_and_restart();
            continue;
        }

        const f64 cpuct = [&] {
            f64 base = node_idx == active_half().root_idx() ? ROOT_EXPLORATION_CONSTANT : EXPLORATION_CONSTANT;
            // Scale the exploration constant logarithmically with the number of visits this node has
            base *= 1.0 + std::log((node.num_visits + CPUCT_VISIT_SCALE) / CPUCT_VISIT_SCALE_DIVISOR);
            base *=
                std::min<f64>(GINI_MAXIMUM, GINI_BASE - GINI_MULTIPLIER * std::log(node.gini_impurity / 255.0 + 0.001));
            return base;
        }();

        NodeIndex best_child_idx = 0;
        f64 best_child_score = std::numeric_limits<f64>::min();
        for (u16 i = 0; i < node.num_children; ++i) {
            Node &child_node = node_at(node.first_child_idx + i);
            // Track the child with the highest PUCT score
            const f64 child_score = compute_puct(node, child_node, cpuct);
            if (child_score > best_child_score) {
                best_child_idx = node.first_child_idx + i; // Store absolute index into nodes
                best_child_score = child_score;
            }
        }

        // Keep descending through the game tree until we find a suitable node to expand
        node_idx = best_child_idx, nodes_in_path_.push_back(node_idx);
        board_.make_move(node_at(node_idx).move);
    }
}

void GameTree::compute_policy(const BoardState &state, NodeIndex node_idx) {
    Node &node = node_at(node_idx);

    // We keep track of a policy context so that we only accumulate once per node
    const network::policy::PolicyContext ctx(state);

    const bool root_node = node_idx == active_half().root_idx();
    const f32 temperature = root_node ? ROOT_SOFTMAX_TEMPERATURE : SOFTMAX_TEMPERATURE;

    f32 highest_policy = -std::numeric_limits<f32>::max();
    for (u16 i = 0; i < node.num_children; ++i) {
        Node &child = node_at(node.first_child_idx + i);
        // Compute policy output for this move
        const auto &history_entry = butterfly_table_[board_.state().side_to_move][child.move.from()][child.move.to()];
        child.policy_score = (ctx.logit(child.move) + history_entry / 16384.0) / temperature;
        // Keep track of highest policy so we can shift all the policy
        // values down to avoid precision loss from large exponents
        highest_policy = std::max(highest_policy, child.policy_score);
    }

    // Softmax the policy logits
    f32 sum_exponents = 0.0f;
    for (u16 i = 0; i < node.num_children; ++i) {
        Node &child = node_at(node.first_child_idx + i);
        const f32 exp_policy = std::exp(child.policy_score - highest_policy);
        sum_exponents += exp_policy;
        child.policy_score = exp_policy;
    }

    f32 sum_squares = 0.0f;
    // Normalize into policy scores
    for (u16 i = 0; i < node.num_children; ++i) {
        Node &child = node_at(node.first_child_idx + i);
        child.policy_score /= sum_exponents;
        sum_squares += child.policy_score * child.policy_score;
    }

    node.gini_impurity = static_cast<u8>(255.0f * std::clamp(1.0f - sum_squares, 0.0f, 1.0f));
}

bool GameTree::expand_node(NodeIndex node_idx) {
    auto &node = node_at(node_idx);
    if (node.expanded() || node.terminal()) {
        return true;
    }

    // We should only be expanding when the number of visits is one
    // This is due to the optimization of not expanding nodes whose children we don't know we'll need
    vine_assert(node_idx.index() == 0 || node.num_visits > 0);

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

    // Return early if we will run out of tree capacity
    if (!active_half().has_room_for(move_list.size())) {
        return false;
    }

    node.first_child_idx = active_half().construct_idx(active_half().filled_size());
    node.num_children = move_list.size();

    // Append all child nodes to the nodes with the move that leads to it
    for (const auto move : move_list) {
        active_half().push_node(Node{
            .move = move,
        });
    }

    tree_usage_ += node.num_children * sizeof(Node);

    // Compute and store policy values for all the child nodes
    compute_policy(board_.state(), node_idx);

    return true;
}

f64 GameTree::simulate_node(NodeIndex node_idx) {
    const auto &node = node_at(node_idx);
    if (node.terminal()) {
        return node.terminal_state.score();
    }

    // Return the cached Q of this node if it exists instead of calling out to the value network
    if (const auto hash_entry = hash_table_.probe(board_.state().hash_key)) {
        return hash_entry->q;
    }

    return util::math::sigmoid(network::value::evaluate(board_.state()));
}

void GameTree::backpropagate_terminal_state(NodeIndex node_idx, TerminalState child_terminal_state) {
    auto &node = node_at(node_idx);
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
            const auto sibling_terminal_state = node_at(node.first_child_idx + i).terminal_state;
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

i16 scale_bonus(i16 score, i16 bonus) {
    return bonus - score * std::abs(bonus) / 8192;
}

void GameTree::backpropagate_score(f64 score) {
    vine_assert(!nodes_in_path_.empty());

    auto child_terminal_state = TerminalState::none();
    while (!nodes_in_path_.empty()) {
        const auto node_idx = nodes_in_path_.pop_back();

        // A node's score is the average of all of its children's score
        auto &node = node_at(node_idx);
        node.sum_of_scores += score;
        node.num_visits++;
        hash_table_.update(board_.state().hash_key, node.q(), node.num_visits);

        // If a terminal state from the child score exists, then we try to backpropagate it to this node
        if (!child_terminal_state.is_none()) {
            backpropagate_terminal_state(node_idx, child_terminal_state);
        }

        // If this node has a terminal state (either from backpropagation or it is terminal), we save it for the parent
        // node to try to use it
        if (!node.terminal_state.is_none()) {
            child_terminal_state = node.terminal_state;
        }

        // Negate the score to match the perspective of the node
        score = 1.0 - score;

        if (!nodes_in_path_.empty()) {
            board_.undo_move();
            if (child_terminal_state.is_none()) {
                auto &history_entry = butterfly_table_[board_.state().side_to_move][node.move.from()][node.move.to()];
                score = std::clamp(score, 0.001, 0.999);
                history_entry +=
                    scale_bonus(history_entry, static_cast<i32>(std::round(-400.0 * std::log(1.0 / score - 1.0))));
            }
        }
    }
}

bool GameTree::fetch_children(NodeIndex node_idx) {
    auto &node = node_at(node_idx);
    // Don't do anything if the node's children already exist in our half
    if (node.first_child_idx.half() == active_half_) {
        return true;
    }

    // Check if we need to the active tree half
    vine_assert(node.num_children > 0);
    if (!active_half().has_room_for(node.num_children)) {
        return false;
    }

    // Copy over the children from the other tree half to this half
    for (u16 i = 0; i < node.num_children; ++i) {
        active_half().push_node(node_at(node.first_child_idx + i));
    }
    node.first_child_idx = active_half().construct_idx(active_half().filled_size() - node.num_children);

    return true;
}

void GameTree::flip_halves() {
    auto old_root_idx = active_half().root_idx();
    active_half().clear_dangling_references();
    active_half_ = ~active_half_;
    active_half().clear();
    active_half().push_node(node_at(old_root_idx));
}

[[nodiscard]] TreeHalf &GameTree::active_half() {
    return halves_[active_half_];
}

[[nodiscard]] const TreeHalf &GameTree::active_half() const {
    return halves_[active_half_];
}

bool GameTree::advance_root_node(Board old_board, const Board &new_board, NodeIndex start) {
    if (active_half().filled_size() == 0) {
        return false;
    }

    const auto &node = node_at(start);
    if (!node.expanded()) {
        return false;
    }

    for (u16 i = 0; i < node.num_children; ++i) {
        const auto child_node = node_at(node.first_child_idx + i);
        // Ensure this move leads to the same resulting position
        old_board.make_move(child_node.move);
        if (old_board.state() == new_board.state()) {
            // Don't advance to unexpanded nodes
            if (!child_node.expanded()) {
                return false;
            }
            // Copy over the new root node to the correct place
            root() = child_node;
            return true;
        }
        // Check two moves deep from the root position
        if (start == active_half().root_idx() && advance_root_node(old_board, new_board, node.first_child_idx + i)) {
            return true;
        }
        old_board.undo_move();
    }

    return old_board.state() == new_board.state();
}

void GameTree::inject_dirichlet_noise(NodeIndex node_idx) {
    auto &node = node_at(node_idx);
    vine_assert(node_idx == active_half().root_idx());

    std::vector<f64> noise;
    noise.reserve(node.num_children);

    // Generate a distribution of random numbers and normalize
    f64 sum = 0.0f;
    for (usize i = 0; i < node.num_children; i++) {
        noise.push_back(rng::next_f64_gamma(dirichlet_alpha_));
        sum += noise.back();
    }
    for (usize i = 0; i < node.num_children; i++) {
        noise[i] /= sum;
    }

    // Mix in the Dirichlet noise with the policy priors
    for (u16 i = 0; i < node.num_children; ++i) {
        Node &child = node_at(node.first_child_idx + i);
        child.policy_score =
            static_cast<f32>((1.0 - dirichlet_epsilon_) * child.policy_score + dirichlet_epsilon_ * noise[i]);
    }
}

void GameTree::clear() {
    for (auto &half : halves_) {
        half.clear();
    }
    hash_table_.clear();
    tree_usage_ = 0;
    active_half_ = {};
    board_ = {};
    butterfly_table_ = {};
    sum_depths_ = 0;
    nodes_in_path_.clear();
}

} // namespace search
