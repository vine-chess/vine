#include "thread.hpp"
#include <iostream>

namespace search {

constexpr f64 EXPLORATION_CONSTANT = 1.5;

Thread::Thread() {
    // raw_thread_ = std::thread(&Thread::thread_loop, this);
}

Thread::~Thread() {
    /* if (raw_thread_.joinable()) {
        raw_thread_.join();
    } */
}

void Thread::go(std::vector<Node> &tree, Board &board, const TimeSettings &time_settings) {
    board_ = board;
    time_manager_.start_tracking(time_settings);

    while (!time_manager_.times_up(board.state().side_to_move)) {
        auto node = select_node(tree);
    }
}

Node &Thread::select_node(std::vector<Node> &tree) {
    // Lambda to compute the PUCT score for a given child node in MCTS
    // Arguments:
    // - parent: the parent node from which the child was reached
    // - child: the candidate child node being scored
    // - policy_score: the probability for this child being the best move
    // - exploration_constant: hyperparameter controlling exploration vs. exploitation
    const auto compute_puct = [&](Node &parent, Node &child, f64 policy_score, f64 exploration_constant) -> f64 {
        // Average value of the child from previous visits (Q value)
        // If the node hasn't been visited, default to 0.0 (TODO: Implement FPU)
        const f64 q_value = child.num_visits > 0 ? child.sum_of_scores / static_cast<f64>(child.num_visits) : 0.0;
        // Total visit count of the parent node (+1 to prevent division by zero)
        const f64 parent_visits = parent.num_visits + 1;
        // Uncertainty/exploration term (U value), scaled by the prior and parent visits
        const f64 u_value =
            exploration_constant * policy_score * std::sqrt(parent_visits) / (1.0 + static_cast<f64>(child.num_visits));
        // Final PUCT score is exploitation (Q) + exploration (U)
        return q_value + u_value;
    };

    u32 node_idx = 0;
    while (true) {
        Node &node = tree[node_idx];
        // Return if we cannot go any further down the tree
        if (node.terminal()) {
            return node;
        }

        // Select this node to expand if it has an unvisited child
        for (u16 i = 0; i < node.num_children; ++i) {
            Node &child_node = tree[node.first_child_idx + i];
            if (!child_node.expanded()) {
                return node;
            }
        }

        u32 best_child_idx = 0;
        f64 best_child_score = std::numeric_limits<f64>::min();
        for (u16 i = 0; i < node.num_children; ++i) {
            Node &child_node = tree[node.first_child_idx + i];

            // Temporary placeholder for NN — currently using uniform policy
            // TODO: Replace with real neural net policy output when available
            const f64 policy_score = 1.0 / static_cast<f64>(node.num_children);

            // Track the child with the highest score so far
            const f64 child_score = compute_puct(node, child_node, policy_score, EXPLORATION_CONSTANT);
            if (child_score > best_child_score) {
                best_child_idx = node.first_child_idx + i; // Store absolute index into tree
                best_child_score = child_score;
            }
        }

        // Keep searching through the game tree until we find a suitable node to expand
        node_idx = best_child_idx;
    }
}

void Thread::thread_loop() {
    // TODO: this shit
}

} // namespace search