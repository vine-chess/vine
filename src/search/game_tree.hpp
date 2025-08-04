#ifndef GAME_TREE_HPP
#define GAME_TREE_HPP

#include "../chess/board.hpp"
#include "hash.hpp"
#include "node.hpp"

namespace search {

class GameTree {
  public:
    GameTree();
    ~GameTree() = default;

    void set_node_capacity(usize capacity);
    void set_hash_capacity(usize capacity);

    void new_search(const Board &root_board);

    [[nodiscard]] const Node &node_at(u32 idx) const;
    [[nodiscard]] const Node &root() const;
    [[nodiscard]] Node &root();

    [[nodiscard]] u32 sum_depths() const;

    // Stage 1/2: Selection & Expansion
    // Selection is the first stage of an iteration and finds a leaf node for us to expand and/or simulate.
    // Expansion is the second stage of an iteration. However, due to memory-usage optimization we perform expansion
    // whenever a node is selected twice, which is handled in the selection stage.
    [[nodiscard]] std::pair<u32, bool> select_and_expand_node();
    // This function computes the policy scores for all children of a node that is already expanded. The policy score is
    // the main influence of the PUCT algorithm, which drives the selection stage toward a new leaf node to expand.
    void compute_policy(u32 node_idx);

    // Stage 3: Simulation
    // Calls out to the value head to return a score for the node that is being simulated.
    [[nodiscard]] f64 simulate_node(u32 node_idx);

    // Stage 4 (Final): Backpropagation
    // Propagates the scores of a node that was just simulated to itself and its ancestor nodes, increasing the number
    // of visits to each node that had a score propagated to it.
    void backpropagate_score(f64 score, u32 node_idx);

  private:
    void backpropagate_terminal_state(u32 node_idx, TerminalState child_terminal_state);
    [[nodiscard]] bool expand_node(u32 node_idx);

    const HashEntry &probe_hash(u64 hash_key) const {
        return hash_table_[hash_key % hash_table_.size()];
    }

    void write_hash(u64 hash_key, f64 q) {
        hash_table_[hash_key % hash_table_.size()] = {hash_key, q};
    }

    std::vector<Node> nodes_;
    std::vector<HashEntry> hash_table_;
    Board board_;
    u32 sum_depths_ = 0;
    u32 nodes_in_path_ = 0;
};

} // namespace search

#endif // GAME_TREE_HPP
