#include "thread.hpp"
#include "info.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace search {

Thread::Thread() {}

Thread::~Thread() {}

u64 Thread::iterations() const {
    return num_iterations_;
}

void Thread::go(GameTree &tree, const Board &root_board, const TimeSettings &time_settings, Verbosity verbosity) {
    time_manager_.start_tracking(time_settings);

    tree.new_search(root_board);

    u64 iterations = 0;
    u64 previous_depth = 0;

    while (++iterations) {
        const auto node = tree.select_and_expand_node();
        tree.backpropagate_score(tree.simulate_node(node), node);

        const u64 depth = tree.sum_depths() / iterations;
        if (depth > previous_depth) {
            previous_depth = depth;
            if (verbosity == Verbosity::VERBOSE) {
                write_info(tree, iterations);
            }
        }

        if (time_manager_.times_up(iterations, root_board.state().side_to_move, depth)) {
            break;
        }
    }

    const Node &root = tree.root();
    if (root.num_children == 0) {
        return;
    }

    if (verbosity != Verbosity::NONE) {
        write_info(tree, iterations, true);
    }
    num_iterations_ = iterations;
}

void Thread::thread_loop() {
    // TODO: this shit
}

void extract_pv_internal(std::vector<Move> &pv, const Node &node, GameTree &tree) {
    if (node.terminal() || !node.expanded()) {
        return;
    }

    const auto get_child_score = [&](NodeIndex child_idx) {
        const f64 MATE_SCORE = 1000.0;
        const Node &child = tree.node_at(child_idx);
        switch (child.terminal_state.flag()) {
        case TerminalState::Flag::WIN:
            return MATE_SCORE - child.terminal_state.distance_to_terminal();
        case TerminalState::Flag::LOSS:
            return -MATE_SCORE + child.terminal_state.distance_to_terminal();
        default:
            return child.q();
        }
    };

    NodeIndex best_child_idx = node.first_child_idx;
    for (u16 i = 0; i < node.num_children; ++i) {
        if (get_child_score(node.first_child_idx + i) < get_child_score(best_child_idx)) {
            best_child_idx = node.first_child_idx + i;
        }
    }

    pv.push_back(tree.node_at(best_child_idx).move);
    extract_pv_internal(pv, tree.node_at(best_child_idx), tree);
}

void extract_pv(std::vector<Move> &pv, GameTree &tree) {
    extract_pv_internal(pv, tree.root(), tree);
}

void Thread::write_info(GameTree &tree, u64 iterations, bool write_bestmove) const {
    const Node &root = tree.root();
    const auto is_mate = root.terminal_state.is_win() || root.terminal_state.is_loss();
    const auto score = is_mate ? (root.terminal_state.distance_to_terminal() + 1) / 2
                               : static_cast<int>(std::round(-400.0 * std::log(1.0 / root.q() - 1.0)));

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
    std::cout << "info depth " << tree.sum_depths() / iterations << " nodes " << iterations << " time " << elapsed
              << " nps " << iterations * 1000 / elapsed << " score " << (is_mate ? "mate " : "cp ")
              << (root.terminal_state.is_loss() ? "-" : "") << score << " pv " << pv_stream.str() << std::endl;
    if (write_bestmove) {
        std::cout << "bestmove " << pv[0].to_string() << std::endl;
    }
}

} // namespace search
