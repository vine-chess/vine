#ifndef GAME_TREE_HPP
#define GAME_TREE_HPP

#include "../chess/board.hpp"
#include "node.hpp"

namespace search {

class TreeHalf {
  public:
    enum Index : u8 {
        LOWER = 0,
        UPPER
    };

    TreeHalf(Index our_half) : our_half_(our_half) {}

    void set_node_capacity(usize capacity) {
        nodes_.clear();
        nodes_.shrink_to_fit();
        nodes_.reserve(capacity);
    }

    [[nodiscard]] usize filled_size() const {
        return nodes_.size();
    }

    [[nodiscard]] bool has_room_for(usize n) const {
        return filled_size() + n <= nodes_.capacity();
    }

    void clear_dangling() {
        for (auto &node : nodes_) {
            if (node.first_child_idx.half() != our_half_) {
                node.first_child_idx = NodeIndex::none();
                node.num_children = 0;
            }
        }
    }

    void push_node(const Node &node) {
        vine_assert(has_room_for(1));
        nodes_.push_back(node);
    }

    Node& operator[](NodeIndex idx) {
        return nodes_[idx.index()];
    }

    const Node& operator[](NodeIndex idx) const {
        return nodes_[idx.index()];
    }

    void clear() {
        nodes_.clear();
    }

  private:
    std::vector<Node> nodes_;
    Index our_half_;
};

class GameTree {
  public:
    GameTree();
    ~GameTree() = default;

    void set_node_capacity(usize capacity);

    void new_search(const Board &root_board);

    [[nodiscard]] const Node &node_at(NodeIndex idx) const;
    [[nodiscard]] Node &node_at(NodeIndex idx);
    [[nodiscard]] const Node &root() const;
    [[nodiscard]] Node &root();

    [[nodiscard]] u32 sum_depths() const;

    // Stage 1/2: Selection & Expansion
    // Selection is the first stage of an iteration and finds a leaf node for us to expand and/or simulate.
    // Expansion is the second stage of an iteration. However, due to memory-usage optimization we perform expansion
    // whenever a node is selected twice, which is handled in the selection stage.
    [[nodiscard]] std::pair<NodeIndex, bool> select_and_expand_node();
    // This function computes the policy scores for all children of a node that is already expanded. The policy score is
    // the main influence of the PUCT algorithm, which drives the selection stage toward a new leaf node to expand.
    void compute_policy(NodeIndex idx);

    // Stage 3: Simulation
    // Calls out to the value head to return a score for the node that is being simulated.
    [[nodiscard]] f64 simulate_node(NodeIndex idx);

    // Stage 4 (Final): Backpropagation
    // Propagates the scores of a node that was just simulated to itself and its ancestor nodes, increasing the number
    // of visits to each node that had a score propagated to it.
    void backpropagate_score(f64 score, NodeIndex idx);

  private:
    void backpropagate_terminal_state(NodeIndex idx, TerminalState child_terminal_state);

    [[nodiscard]] bool expand_node(NodeIndex idx);

    [[nodiscard]] bool fetch_children(NodeIndex idx);

    void flip_halves();

    std::array<TreeHalf, 2> halves_;
    TreeHalf::Index active_half_ = TreeHalf::Index::LOWER;
    Board board_;
    u32 sum_depths_ = 0;
    u32 nodes_in_path_ = 0;
};

} // namespace search

#endif // GAME_TREE_HPP
