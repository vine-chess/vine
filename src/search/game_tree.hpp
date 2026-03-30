#ifndef GAME_TREE_HPP
#define GAME_TREE_HPP

#include "../chess/board.hpp"
#include "../eval/evaluator.hpp"
#include "../util/tunable.hpp"
#include "hash_table.hpp"
#include "history.hpp"
#include "node.hpp"
#include "tree_half.hpp"
#include "../chess/move_gen.hpp"
#include "../util/assert.hpp"
#include "../util/math.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <span>

namespace search {

#ifdef DATAGEN
TUNABLE_STEP(ROOT_SOFTMAX_TEMPERATURE, 3.5f, 0.5f, 5.0f, 0.1f);
#else
TUNABLE_STEP(ROOT_SOFTMAX_TEMPERATURE, 2.010033808133197, 0.5f, 3.0f, 0.1f);
#endif
TUNABLE_STEP(SOFTMAX_TEMPERATURE, 1.288386775295493, 1.0f, 3.0f, 0.08);
TUNABLE_STEP(ROOT_EXPLORATION_CONSTANT, 1.3847280475167136, 0.5f, 2.5f, 0.05f);
TUNABLE_STEP(EXPLORATION_CONSTANT, 0.8481266142929403, 0.5f, 2.5f, 0.05f);
TUNABLE_STEP(CPUCT_VISIT_SCALE, 7827, 2048, 16384, 256);
TUNABLE_STEP(CPUCT_VISIT_SCALE_DIVISOR, 8815, 2048, 16384, 256);
TUNABLE_STEP(GINI_BASE, 0.4426053054270583, 0.0f, 1.5f, 0.05f);
TUNABLE_STEP(GINI_MULTIPLIER, 1.4479240539808982, 0.5f, 3.0f, 0.1f);
TUNABLE_STEP(GINI_MAXIMUM, 2.1158971557968873, 1.25f, 3.25f, 0.1f);
TUNABLE_STEP(POLICY_HISTORY_DIVISOR, 16333, 8192, 32768, 1024);
TUNABLE_STEP(KNIGHT_MATERIAL, 298, 100, 600, 30);
TUNABLE_STEP(BISHOP_MATERIAL, 315, 100, 600, 30);
TUNABLE_STEP(ROOK_MATERIAL, 473, 300, 800, 40);
TUNABLE_STEP(QUEEN_MATERIAL, 863, 500, 1500, 50);

class GameTree {
  public:
    GameTree();
    ~GameTree() = default;

    void set_node_capacity(usize capacity);
    void set_hash_table_capacity(usize capacity);

    template <class Evaluator>
    void new_search(const Board &root_board, Evaluator &evaluator);

    [[nodiscard]] Node &node_at(NodeIndex idx);
    [[nodiscard]] const Node &node_at(NodeIndex idx) const;
    [[nodiscard]] const Node &root() const;
    [[nodiscard]] Node &root();

    [[nodiscard]] u32 sum_depths() const;
    [[nodiscard]] u64 tree_usage() const;

    // Stage 1/2: Selection & Expansion
    // Selection is the first stage of an iteration and finds a leaf node for us to expand and/or simulate.
    // Expansion is the second stage of an iteration. However, due to memory-usage optimization we perform expansion
    // whenever a node is selected twice, which is handled in the selection stage.
    template <class Evaluator>
    [[nodiscard]] NodeIndex select_and_expand_node(Evaluator &evaluator);
    // This function computes the policy scores for all children of a node that is already expanded. The policy score is
    // the main influence of the PUCT algorithm, which drives the selection stage toward a new leaf node to expand.
    template <class Evaluator>
    void compute_policy(const BoardState &state, NodeIndex node_idx, Evaluator &evaluator);

    // Stage 3: Simulation
    // Calls out to the value head to return a score for the node that is being simulated.
    template <class Evaluator>
    [[nodiscard]] f64 simulate_node(NodeIndex node_idx, Evaluator &evaluator);

    // Stage 4 (Final): Backpropagation
    // Propagates the scores of a node that was just simulated to itself and its ancestor nodes, increasing the number
    // of visits to each node that had a score propagated to it.
    void backpropagate_score(f64 score);

    void flip_halves();

    void clear();

  private:
    void backpropagate_terminal_state(NodeIndex node_idx, TerminalState child_terminal_state);

    [[nodiscard]] std::span<Node> get_children(Node node);

    template <class Evaluator>
    [[nodiscard]] bool expand_node(NodeIndex node_idx, Evaluator &evaluator);

    [[nodiscard]] bool fetch_children(NodeIndex node_idx);

    [[nodiscard]] TreeHalf &active_half();
    [[nodiscard]] const TreeHalf &active_half() const;

    [[nodiscard]] bool advance_root_node(Board old_board, const Board &new_board, NodeIndex start);

    std::array<TreeHalf, 2> halves_;
    HashTable hash_table_;
    u64 tree_usage_ = 0;
    TreeHalf::Index active_half_;
    Board board_;
    u32 sum_depths_ = 0;
    util::StaticVector<NodeIndex, 512> nodes_in_path_;
    History history_;
};

template <class Evaluator>
void GameTree::new_search(const Board &root_board, Evaluator &evaluator) {
    if (advance_root_node(board_, root_board, active_half().root_idx())) {
        compute_policy(root_board.state(), active_half().root_idx(), evaluator);
    } else {
        active_half().clear();
        active_half().push_node(Node{});
    }

    board_ = root_board;
    sum_depths_ = 0;
    tree_usage_ = 0;

    if (!expand_node(active_half().root_idx(), evaluator)) {
        flip_halves();
        vine_assert(expand_node(active_half().root_idx(), evaluator));
    }
}

template <class Evaluator>
NodeIndex GameTree::select_and_expand_node(Evaluator &evaluator) {
    const auto compute_puct = [&](const Node &parent, const Node &child, f64 exploration_constant) -> f64 {
        const f64 q_value = child.num_visits > 0 ? 1.0 - child.q() : parent.q();
        const f64 u_scale = exploration_constant * std::sqrt(static_cast<f64>(parent.num_visits));
        const f64 u_base = child.policy_score / (1.0 + static_cast<f64>(child.num_visits));
        return std::fma(u_base, u_scale, q_value);
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

        if (node.num_visits > 0) {
            if (!expand_node(node_idx, evaluator)) {
                flip_and_restart();
                continue;
            }
        }

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
            base *= 1.0 + std::log((node.num_visits + CPUCT_VISIT_SCALE) / static_cast<f64>(CPUCT_VISIT_SCALE_DIVISOR));
            base *=
                std::min<f64>(GINI_MAXIMUM, GINI_BASE - GINI_MULTIPLIER * std::log(node.gini_impurity / 255.0 + 0.001));
            return base;
        }();

        NodeIndex best_child_idx = 0;
        f64 best_child_score = std::numeric_limits<f64>::min();

        const Node parent = node;
        const auto children = get_children(node);
        for (u16 i = 0; i < node.num_children; ++i) {
            Node &child_node = children[i];
            const f64 child_score = compute_puct(parent, child_node, cpuct);
            if (child_score > best_child_score) {
                best_child_idx = node.first_child_idx + i;
                best_child_score = child_score;
            }
        }

        node_idx = best_child_idx, nodes_in_path_.push_back(node_idx);
        board_.make_move(node_at(node_idx).move);
    }
}

template <class Evaluator>
void GameTree::compute_policy(const BoardState &state, NodeIndex node_idx, Evaluator &evaluator) {
    Node &node = node_at(node_idx);

    auto ctx = evaluator.policy_context(state);
    for (Node &child : get_children(node)) {
        ctx.enqueue(child.move, state.get_piece_type(child.move.from()));
    }
    ctx.ready();

    const bool root_node = node_idx == active_half().root_idx();
    const f32 temperature = root_node ? ROOT_SOFTMAX_TEMPERATURE : SOFTMAX_TEMPERATURE;

    f32 highest_policy = -std::numeric_limits<f32>::max();
    for (Node &child : get_children(node)) {
        const auto history_score =
            history_.entry(board_.state(), child.move).value / static_cast<f64>(POLICY_HISTORY_DIVISOR);
        child.policy_score = (ctx.logit() + history_score) / temperature;
        highest_policy = std::max(highest_policy, child.policy_score);
    }

    f32 sum_exponents = 0.0f;
    for (Node &child : get_children(node)) {
        const f32 exp_policy = std::exp(child.policy_score - highest_policy);
        sum_exponents += exp_policy;
        child.policy_score = exp_policy;
    }

    f32 sum_squares = 0.0f;
    for (Node &child : get_children(node)) {
        child.policy_score /= sum_exponents;
        sum_squares += child.policy_score * child.policy_score;
    }

    node.gini_impurity = static_cast<u8>(255.0f * std::clamp(1.0f - sum_squares, 0.0f, 1.0f));
}

template <class Evaluator>
bool GameTree::expand_node(NodeIndex node_idx, Evaluator &evaluator) {
    auto &node = node_at(node_idx);
    if (node.expanded() || node.terminal()) {
        return true;
    }

    vine_assert(node_idx.index() == 0 || node.num_visits > 0);

    if (board_.is_draw() && node_idx != active_half().root_idx()) {
        node.terminal_state = TerminalState::draw();
        return true;
    }

    MoveList move_list;
    generate_moves(board_.state(), move_list);

    if (move_list.empty()) {
        node.terminal_state = board_.state().checkers != 0 ? TerminalState::loss(0) : TerminalState::draw();
        return true;
    }

    if (!active_half().has_room_for(move_list.size())) {
        return false;
    }

    node.first_child_idx = active_half().construct_idx(active_half().filled_size());
    node.num_children = move_list.size();

    for (const auto move : move_list) {
        active_half().push_node(Node{
            .move = move,
        });
    }

    tree_usage_ += node.num_children * sizeof(Node);
    compute_policy(board_.state(), node_idx, evaluator);
    return true;
}

template <class Evaluator>
f64 GameTree::simulate_node(NodeIndex node_idx, Evaluator &evaluator) {
    const auto &node = node_at(node_idx);
    if (node.terminal()) {
        return node.terminal_state.score();
    }

    if (const auto hash_entry = hash_table_.probe(board_.state().hash_key)) {
        return hash_entry->q;
    }

    const auto num_knights = board_.state().knights().pop_count();
    const auto num_bishops = board_.state().bishops().pop_count();
    const auto num_rooks = board_.state().rooks().pop_count();
    const auto num_queens = board_.state().queens().pop_count();
    const auto sum_material = KNIGHT_MATERIAL * num_knights + BISHOP_MATERIAL * num_bishops +
                              ROOK_MATERIAL * num_rooks + QUEEN_MATERIAL * num_queens;
    const auto raw_eval = evaluator.value(board_.state());
    const auto scaled = raw_eval * (sum_material + 8192) / 16384;

    return util::math::sigmoid(scaled);
}

} // namespace search

#endif // GAME_TREE_HPP
