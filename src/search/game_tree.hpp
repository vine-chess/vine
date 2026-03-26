#ifndef GAME_TREE_HPP
#define GAME_TREE_HPP

#include "../chess/board.hpp"
#include "../eval/evaluator.hpp"
#include "../util/tunable.hpp"
#include "hash_table.hpp"
#include "history.hpp"
#include "node.hpp"
#include "tree_half.hpp"
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

} // namespace search

#include "game_tree.tpp"

#endif // GAME_TREE_HPP
