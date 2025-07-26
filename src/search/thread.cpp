#include "thread.hpp"
#include "../chess/move_gen.hpp"
#include "../util/assert.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace search {

Thread::Thread() {}

Thread::~Thread() {}

u64 Thread::iterations() const {
    return num_iterations_;
}

void Thread::go(GameTree &tree, const Board &root_board, const TimeSettings &time_settings) {
    time_manager_.start_tracking(time_settings);

    tree.new_search(root_board);
    tree.new_search(root_board);

    u64 iterations = 0;
    u64 previous_depth = 0;

    while (++iterations) {
        const auto [node, success] = tree.select_and_expand_node();
        // Stop searching if we can't add any more nodes to the tree
        if (!success) {
            break;
        }
        tree.backpropagate_score(tree.simulate_node(node), node);

        const u64 depth = tree.sum_depths() / iterations;
        if (depth > previous_depth) {
            previous_depth = depth;
            write_info(tree, iterations);
        }

        if (time_manager_.times_up(iterations, root_board.state().side_to_move, depth)) {
            break;
        }
    }

    const Node &root = tree.root();
    if (root.num_children == 0) {
        return;
    }

    write_info(tree, iterations, true);
    num_iterations_ = iterations;
}

void Thread::thread_loop() {
    // TODO: this shit
}

void extract_pv_internal(std::vector<Move> &pv, u32 node_idx, GameTree &tree) {
    const auto &node = tree.node_at(node_idx);
    if (node.terminal() || !node.expanded()) {
        return;
    }

    u32 best_child_idx = node.first_child_idx;
    u32 most_visits = 0;
    for (u16 i = 0; i < node.num_children; ++i) {
        const Node &child = tree.node_at(node.first_child_idx + i);
        if (child.num_visits > most_visits) {
            most_visits = child.num_visits;
            best_child_idx = node.first_child_idx + i;
        }
    }

    pv.push_back(tree.node_at(best_child_idx).move);
    extract_pv_internal(pv, best_child_idx, tree);
}

void extract_pv(std::vector<Move> &pv, GameTree &tree) {
    extract_pv_internal(pv, 0, tree);
}

void Thread::write_info(GameTree &tree, u64 iterations, bool write_bestmove) const {
    const Node &root = tree.root();
    const auto cp = static_cast<int>(std::round(-400.0 * std::log(1.0 / root.q() - 1.0)));

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
              << " nps " << iterations * 1000 / elapsed << " score cp " << cp << " pv " << pv_stream.str() << std::endl;
    if (write_bestmove) {
        std::cout << "bestmove " << pv[0].to_string() << std::endl;
    }
}

} // namespace search
