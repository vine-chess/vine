#ifndef THREAD_TPP
#define THREAD_TPP

namespace search {

template <class Evaluator>
void Thread::go(GameTree &tree, Evaluator &evaluator, const Board &root_board, const TimeSettings &time_settings,
                Verbosity verbosity) {
    time_manager_.start_tracking(time_settings);

    tree.new_search(root_board, evaluator);

    u64 iterations = 0, nodes = 0;
    u64 previous_depth = 0, previous_sum_depths = 0;

    while (++iterations) {
        const auto node = tree.select_and_expand_node(evaluator);
        tree.backpropagate_score(tree.simulate_node(node, evaluator));

        nodes += tree.sum_depths() - previous_sum_depths;
        previous_sum_depths = tree.sum_depths();

        const u64 depth = tree.sum_depths() / iterations;
        if (depth > previous_depth) {
            previous_depth = depth;
            if (verbosity == Verbosity::VERBOSE) {
                write_info(tree, iterations, nodes);
            }
        }

        if (time_manager_.times_up(tree, iterations, root_board.state().side_to_move, depth)) {
            break;
        }
    }

    const Node &root = tree.root();
    if (root.num_children == 0) {
        return;
    }

    if (verbosity != Verbosity::NONE) {
        write_info(tree, iterations, nodes, true);
    }
    num_iterations_ = iterations;
}

} // namespace search

#endif // THREAD_TPP
