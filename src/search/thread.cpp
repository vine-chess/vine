#include "thread.hpp"
#include "../chess/move_gen.hpp"
#include "../util/assert.hpp"

#include <algorithm>
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

u64 Thread::iterations() const {
    return num_iterations_;
}

void Thread::go(std::vector<Node> &tree, Board &board, const TimeSettings &time_settings) {
    time_manager_.start_tracking(time_settings);

    // Push the root node to the game tree
    tree.clear();
    tree.emplace_back();

    sum_depth_ = 0;
    u64 iterations = 0;
    u64 previous_depth = 0;

    while (++iterations) {
        board_ = board;

        const auto node = select_node(tree);
        if (!expand_node(node, tree)) {
            break;
        }
        const auto score = simulate_node(node, tree);
        backpropagate(score, node, tree);

        const u64 depth = sum_depth_ / iterations;
        if (depth > previous_depth) {
            previous_depth = depth;
            write_info(tree, board, iterations);
        }

        if (time_manager_.times_up(iterations, board.state().side_to_move, depth)) {
            break;
        }
    }

    const Node &root = tree[0];
    if (root.terminal()) {
        return;
    }

    write_info(tree, board, iterations, true);
    num_iterations_ = iterations;
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
        // If the node hasn't been visited, use the parent nodes Q value, but inverted as a proxy
        vine_assert(parent.num_visits > 0);
        const f64 q_value =
            1.0 - (child.num_visits > 0 ? child.sum_of_scores / static_cast<f64>(child.num_visits)
                                        : 1.0 - parent.sum_of_scores / static_cast<f64>(parent.num_visits));
        // Uncertainty/exploration term (U value), scaled by the prior and parent visits
        const f64 u_value = exploration_constant * policy_score * std::sqrt(parent.num_visits) /
                            (1.0 + static_cast<f64>(child.num_visits));
        // Final PUCT score is exploitation (Q) + exploration (U)
        return q_value + u_value;
    };

    u32 node_idx = 0, ply = 0;
    while (true) {
        Node &node = tree[node_idx];
        // Return if we cannot go any further down the tree
        if (node.terminal() || !node.expanded()) {
            sum_depth_ += ply;
            return node_idx;
        }

        u32 best_child_idx = 0;
        f64 best_child_score = std::numeric_limits<f64>::min();
        for (u16 i = 0; i < node.num_children; ++i) {
            Node &child_node = tree[node.first_child_idx + i];

            // Track the child with the highest PUCT score
            const f64 child_score = compute_puct(node, child_node, child_node.policy_score, EXPLORATION_CONSTANT);
            if (child_score > best_child_score) {
                best_child_idx = node.first_child_idx + i; // Store absolute index into tree
                best_child_score = child_score;
            }
        }

        // Keep descending through the game tree until we find a suitable node to expand
        node_idx = best_child_idx, ++ply;
        board_.make_move(tree[node_idx].move);
    }
}

void Thread::compute_policy(std::vector<Node> &tree, u32 node_idx) {
    const Node &node = tree[node_idx];
    f64 sum_exponents = 0;
    for (u16 i = 0; i < node.num_children; ++i) {
        Node &child = tree[node.first_child_idx + i];

        // Temporary placeholder for NN
        // TODO: Replace with real neural net policy output when available
        const f64 policy_score = [&]() {
            const Move move = child.move;
            if (!move.is_capture()) {
                return 0.0;
            }
            const PieceType victim = move.is_ep() ? PieceType::PAWN : board_.state().get_piece_type(move.to());
            const PieceType attacker = board_.state().get_piece_type(move.from());
            return (10.0 * victim - attacker) / 40.0;
        }();
        const f64 exp_policy = std::exp(policy_score);
        child.policy_score = static_cast<f32>(exp_policy);
        sum_exponents += exp_policy;
    }
    for (u16 i = 0; i < node.num_children; ++i) {
        Node &child = tree[node.first_child_idx + i];
        child.policy_score /= sum_exponents;
    }
}

bool Thread::expand_node(u32 node_idx, std::vector<Node> &tree) {
    auto &node = tree[node_idx];
    if (node.expanded() || node.terminal()) {
        return true;
    }

    MoveList move_list;
    generate_moves(board_.state(), move_list);

    // If we have no legal moves then the position is terminal
    if (move_list.empty()) {
        node.terminal_state = board_.state().checkers != 0 ? TerminalState::LOSS : TerminalState::DRAW;
        return true;
    }

    node.first_child_idx = tree.size();
    node.num_children = move_list.size();

    // Return early if we will run out of memory
    if (tree.size() + node.num_children > tree.capacity()) {
        return false;
    }

    // Append all child nodes to the tree with the move that leads to it
    for (auto move : move_list) {
        tree.push_back(Node{
            .parent_idx = static_cast<i32>(node_idx),
            .move = move,
        });
    }
    compute_policy(tree, node_idx);

    return true;
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

void extract_pv_internal(std::vector<Move> &pv, u32 node_idx, std::vector<Node> &tree) {
    const auto &node = tree[node_idx];
    if (node.terminal() || !node.expanded()) {
        return;
    }

    u32 best_child_idx = node.first_child_idx;
    u32 most_visits = 0;
    for (u16 i = 0; i < node.num_children; ++i) {
        const Node &child = tree[node.first_child_idx + i];
        if (child.num_visits > most_visits) {
            most_visits = child.num_visits;
            best_child_idx = node.first_child_idx + i;
        }
    }

    pv.push_back(tree[best_child_idx].move);
    extract_pv_internal(pv, best_child_idx, tree);
}

void extract_pv(std::vector<Move> &pv, std::vector<Node> &tree) {
    extract_pv_internal(pv, 0, tree);
}

void Thread::write_info(std::vector<Node> &tree, Board &board, u64 iterations, bool write_bestmove) const {
    const Node &root = tree[0];
    const auto cp = static_cast<int>(std::round(-400.0 * std::log(1.0 / (root.sum_of_scores / root.num_visits) - 1.0)));

    std::vector<Move> pv;
    extract_pv(pv, tree);

    std::ostringstream pv_stream;
    for (int i = 0; i < pv.size(); ++i) {
        pv_stream << pv[i].to_string();
        if (i != pv.size() - 1) {
            pv_stream << " ";
        }
    }

    const auto elapsed = std::max<u64>(1, time_manager_.time_elapsed());
    std::cout << "info depth " << sum_depth_ / iterations << " nodes " << iterations << " time " << elapsed << " nps "
              << iterations * 1000 / elapsed << " score cp " << cp << " pv " << pv_stream.str() << std::endl;
    if (write_bestmove) {
        std::cout << "bestmove " << pv[0].to_string() << std::endl;
    }
}

} // namespace search
