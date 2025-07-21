#include "thread.hpp"
#include "../chess/move_gen.hpp"
#include <cmath>
#include <iostream>

namespace search {

constexpr f64 EXPLORATION_CONSTANT = 1.0;

Thread::Thread() {
    // raw_thread_ = std::thread(&Thread::thread_loop, this);
}

Thread::~Thread() {
    /* if (raw_thread_.joinable()) {
        raw_thread_.join();
    } */
}

void Thread::go(std::vector<Node> &tree, Board &board, const TimeSettings &time_settings) {
    time_manager_.start_tracking(time_settings);

    tree.emplace_back();

    u64 iterations = 0;
    while (++iterations) {
        board_ = board;

        auto node = select_node(tree);
        expand_node(node, tree);
        auto score = simulate_node(node, tree);
        backpropagate(score, node, tree);

        if (iterations % 512 == 0 && time_manager_.times_up(board.state().side_to_move)) {
            break;
        }
    }

    board_ = board;

    const Node &root = tree[0];
    if (root.num_children == 0) {
        return;
    }

    u32 best_child_idx = root.first_child_idx;
    u32 most_visits = 0;

    for (u16 i = 0; i < root.num_children; ++i) {
        const Node &child = tree[root.first_child_idx + i];
        if (child.num_visits > most_visits) {
            most_visits = child.num_visits;
            best_child_idx = root.first_child_idx + i;
        }
    }

    const auto sigmoid_score = root.sum_of_scores / root.num_visits;
    const auto score = static_cast<int>(std::round(-std::log(1.0 / sigmoid_score - 1.0)));
    std::cout << "info nodes " << iterations << " time " << time_manager_.time_elapsed() << " nps "
              << iterations * 1000 / time_manager_.time_elapsed() << " score " << score << "cp" << std::endl
              << "bestmove " << tree[best_child_idx].move.to_string() << std::endl;
}

u32 Thread::select_node(std::vector<Node> &tree) {
    // Lambda to compute the PUCT score for a given child node in MCTS
    // Arguments:
    // - parent: the parent node from which the child was reached
    // - child: the candidate child node being scored
    // - policy_score: the probability for this child being the best move
    // - exploration_constant: hyperparameter controlling exploration vs. exploitation
    const auto compute_puct = [&](Node &parent, Node &child, f64 policy_score, f64 exploration_constant) -> f64 {
        // Average value of the child from previous visits (Q value), flipped to match current node's perspective
        // If the node hasn't been visited, default to 0.0 (TODO: Implement FPU)
        const f64 q_value =
            1.0 - (child.num_visits > 0 ? child.sum_of_scores / static_cast<f64>(child.num_visits) : 0.0);
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
        if (node.terminal() || !node.expanded()) {
            return node_idx;
        }

        u32 best_child_idx = 0;
        f64 best_child_score = std::numeric_limits<f64>::min();
        for (u16 i = 0; i < node.num_children; ++i) {
            Node &child_node = tree[node.first_child_idx + i];

            // Temporary placeholder for NN — currently using uniform policy
            // TODO: Replace with real neural net policy output when available
            const f64 policy_score = 1.0 / static_cast<f64>(node.num_children);

            // Track the child with the highest PUCT score
            const f64 child_score = compute_puct(node, child_node, policy_score, EXPLORATION_CONSTANT);
            if (child_score > best_child_score) {
                best_child_idx = node.first_child_idx + i; // Store absolute index into tree
                best_child_score = child_score;
            }
        }

        // Keep descending through the game tree until we find a suitable node to expand
        node_idx = best_child_idx;
        board_.make_move(tree[node_idx].move);
    }
}

void Thread::expand_node(u32 node_idx, std::vector<Node> &tree) {
    auto &node = tree[node_idx];
    if (node.expanded() || node.terminal()) {
        return;
    }

    MoveList move_list;
    generate_moves(board_.state(), move_list);

    // If we have no legal moves then the position is terminal
    if (move_list.empty()) {
        node.terminal_state = board_.state().checkers != 0 ? TerminalState::LOSS : TerminalState::DRAW;
        return;
    }

    node.first_child_idx = tree.size();
    node.num_children = move_list.size();

    // Append all child nodes to the tree with the move that leads to it
    for (auto move : move_list) {
        tree.push_back(Node{
            .parent_idx = static_cast<i32>(node_idx),
            .move = move,
        });
    }
}

f64 Thread::simulate_node(u32 node_idx, std::vector<Node> &tree) {
    const auto &node = tree[node_idx];
    if (node.terminal()) {
        switch (node.terminal_state) {
        case TerminalState::WIN:
            return 1.0;
        case TerminalState::DRAW:
            return 0.5;
        case TerminalState::LOSS:
            return 0.0;
        default:
            break;
        }
    }

    const auto &state = board_.state();
    const i32 stm_material =
        100 * state.pawns(state.side_to_move).pop_count() + 280 * state.knights(state.side_to_move).pop_count() +
        310 * state.bishops(state.side_to_move).pop_count() + 500 * state.rooks(state.side_to_move).pop_count() +
        1000 * state.queens(state.side_to_move).pop_count();
    const i32 nstm_material =
        100 * state.pawns(~state.side_to_move).pop_count() + 280 * state.knights(~state.side_to_move).pop_count() +
        310 * state.bishops(~state.side_to_move).pop_count() + 500 * state.rooks(~state.side_to_move).pop_count() +
        1000 * state.queens(~state.side_to_move).pop_count();
    const f64 eval = stm_material - nstm_material + 20;
    return 1.0 / (1.0 + std::exp(-eval / 400.0)); // Sigmoid approximation
}

void Thread::backpropagate(f64 score, u32 node_idx, std::vector<Node> &tree) {
    while (node_idx != -1) {
        // A node's score is the average of all of its children's score
        auto &node = tree[node_idx];
        node.sum_of_scores += score;
        node.num_visits++;

        // Travel up to the parent
        node_idx = node.parent_idx;
        // Negate the score to match the perspective of the node
        score = 1.0 - score;
    }
}

void Thread::thread_loop() {
    // TODO: this shit
}

} // namespace search
