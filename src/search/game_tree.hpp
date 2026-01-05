#ifndef GAME_TREE_HPP
#define GAME_TREE_HPP

#include "../chess/board.hpp"
#include "hash_table.hpp"
#include "history.hpp"
#include "node.hpp"
#include "node_index.hpp"
#include "tree_half.hpp"
#include <span>

namespace search {


class GameTree {
  public:
    GameTree();
    ~GameTree() = default;

    void set_node_capacity(usize capacity);
    void set_hash_table_capacity(usize capacity);

    void new_search(const Board &root_board);

    [[nodiscard]] NodeReference node_at(NodeIndex idx);
    [[nodiscard]] NodeReference root();

    [[nodiscard]] u32 sum_depths() const;
    [[nodiscard]] u64 tree_usage() const;

    // Stage 1/2: Selection & Expansion
    // Selection is the first stage of an iteration and finds a leaf node for us to expand and/or simulate.
    // Expansion is the second stage of an iteration. However, due to memory-usage optimization we perform expansion
    // whenever a node is selected twice, which is handled in the selection stage.
    [[nodiscard]] NodeIndex select_and_expand_node();
    // This function computes the policy scores for all children of a node that is already expanded. The policy score is
    // the main influence of the PUCT algorithm, which drives the selection stage toward a new leaf node to expand.
    void compute_policy(const BoardState &state, NodeIndex node_idx);

    // Stage 3: Simulation
    // Calls out to the value head to return a score for the node that is being simulated.
    [[nodiscard]] f64 simulate_node(NodeIndex node_idx);

    // Stage 4 (Final): Backpropagation
    // Propagates the scores of a node that was just simulated to itself and its ancestor nodes, increasing the number
    // of visits to each node that had a score propagated to it.
    void backpropagate_score(f64 score);

    void flip_halves();

    void clear();

  private:
    void backpropagate_terminal_state(NodeIndex node_idx, TerminalState child_terminal_state);

    [[nodiscard]] NodeRange get_children(NodeReference node);

    [[nodiscard]] bool expand_node(NodeIndex node_idx);

    [[nodiscard]] bool fetch_children(NodeIndex node_idx);

    [[nodiscard]] TreeHalf &active_half();
    [[nodiscard]] const TreeHalf &active_half() const;

    [[nodiscard]] bool advance_root_node(Board old_board, const Board &new_board, NodeIndex start);

    void inject_dirichlet_noise(NodeIndex node_idx);

    [[nodiscard]] NodeIndex pick_highest_puct(NodeReference parent, f64 exploration_constant);

    std::array<TreeHalf, 2> halves_;
    HashTable hash_table_;
    u64 tree_usage_ = 0;
    search::HalfIndex active_half_;
    Board board_;
    u32 sum_depths_ = 0;
    util::StaticVector<NodeIndex, 512> nodes_in_path_;
    f64 dirichlet_epsilon_ = 0.0;
    f64 dirichlet_alpha_ = 0.0;
    bool use_gini_ = true;
    History history_;
};

} // namespace search

#endif // GAME_TREE_HPP
