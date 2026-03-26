#include "game_tree.hpp"
#include "../eval/value_network.hpp"
#include "../util/assert.hpp"
#include "../util/math.hpp"

namespace search {

GameTree::GameTree()
    : halves_({TreeHalf(TreeHalf::Index::LOWER), TreeHalf(TreeHalf::Index::UPPER)}),
      active_half_(TreeHalf::Index::LOWER) {
    set_node_capacity(1);
}

void GameTree::set_node_capacity(usize node_capacity) {
    for (auto &half : halves_) {
        half.set_node_capacity(node_capacity / 2);
    }
}

void GameTree::set_hash_table_capacity(usize capacity) {
    hash_table_.set_entry_capacity(capacity);
}

const Node &GameTree::root() const {
    return active_half().root_node();
}

Node &GameTree::root() {
    return active_half().root_node();
}

Node &GameTree::node_at(NodeIndex idx) {
    return halves_[idx.half()][idx.index()];
}

const Node &GameTree::node_at(NodeIndex idx) const {
    return halves_[idx.half()][idx.index()];
}

[[nodiscard]] u32 GameTree::sum_depths() const {
    return sum_depths_;
}

u64 GameTree::tree_usage() const {
    return tree_usage_;
}

void GameTree::backpropagate_terminal_state(NodeIndex node_idx, TerminalState child_terminal_state) {
    auto &node = node_at(node_idx);
    switch (child_terminal_state.flag()) {
    case TerminalState::Flag::LOSS: { // If a child node is lost, then it's a win for us
        // Ensure that if we already had a shorter mate we preserve it
        const auto current_mate_distance =
            node.terminal_state.is_win() ? node.terminal_state.distance_to_terminal() : 255;
        node.terminal_state =
            TerminalState::win(std::min<u8>(current_mate_distance, child_terminal_state.distance_to_terminal() + 1));
        break;
    }
    case TerminalState::Flag::WIN: { // If a child node is won, it's a loss for us if all of its siblings are also won
        u8 longest_loss = 0;
        for (const Node &sibling : get_children(node)) {
            if (sibling.terminal_state.flag() != TerminalState::Flag::WIN) {
                return;
            }
            longest_loss = std::max(longest_loss, sibling.terminal_state.distance_to_terminal());
        }
        node.terminal_state = TerminalState::loss(longest_loss + 1);
        break;
    }
    default:
        break;
    }
}

void GameTree::backpropagate_score(f64 score) {
    vine_assert(!nodes_in_path_.empty());

    auto cp_score = static_cast<i32>(network::value::EVAL_SCALE *
                                     util::math::inverse_sigmoid(std::clamp(score, 0.001, 0.999)));
    auto child_terminal_state = TerminalState::none();

    while (!nodes_in_path_.empty()) {
        const auto node_idx = nodes_in_path_.pop_back();

        // A node's score is the average of all of its children's score
        auto &node = node_at(node_idx);
        node.sum_of_scores += score;
        node.num_visits++;
        hash_table_.update(board_.state().hash_key, node.q(), node.num_visits);

        // If a terminal state from the child score exists, then we try to backpropagate it to this node
        if (!child_terminal_state.is_none()) {
            backpropagate_terminal_state(node_idx, child_terminal_state);
        }

        // If this node has a terminal state (either from backpropagation or it is terminal), we save it for the parent
        // node to try to use it
        if (!node.terminal_state.is_none()) {
            child_terminal_state = node.terminal_state;
        }

        // Negate the score to match the perspective of the node
        score = 1.0 - score;
        cp_score = -cp_score;

        // Undo all moves except the move that led to the root node
        if (!nodes_in_path_.empty()) {
            board_.undo_move();

            // Update the history for this move to influence new node policy scores
            if (child_terminal_state.is_none()) {
                history_.entry(board_.state(), node.move).update(cp_score);
            }
        }
    }
}

std::span<Node> GameTree::get_children(Node node) {
    return {&node_at(node.first_child_idx), node.num_children};
}

bool GameTree::fetch_children(NodeIndex node_idx) {
    Node &node = node_at(node_idx);
    // Don't do anything if the node's children already exist in our half
    if (node.first_child_idx.half() == active_half_) {
        return true;
    }

    // Check if we need to the active tree half
    vine_assert(node.num_children > 0);
    if (!active_half().has_room_for(node.num_children)) {
        return false;
    }

    // Copy over the children from the other tree half to this half
    for (const Node &child : get_children(node)) {
        active_half().push_node(child);
    }
    node.first_child_idx = active_half().construct_idx(active_half().filled_size() - node.num_children);

    return true;
}

void GameTree::flip_halves() {
    auto old_root_idx = active_half().root_idx();
    active_half().clear_dangling_references();
    active_half_ = ~active_half_;
    active_half().clear();
    active_half().push_node(node_at(old_root_idx));
}

[[nodiscard]] TreeHalf &GameTree::active_half() {
    return halves_[active_half_];
}

[[nodiscard]] const TreeHalf &GameTree::active_half() const {
    return halves_[active_half_];
}

bool GameTree::advance_root_node(Board old_board, const Board &new_board, NodeIndex start) {
    if (active_half().filled_size() == 0) {
        return false;
    }

    const auto &node = node_at(start);
    if (!node.expanded()) {
        return false;
    }

    const auto children = get_children(node);
    for (u16 i = 0; i < node.num_children; ++i) {
        const auto child_node = children[i];
        // Ensure this move leads to the same resulting position
        old_board.make_move(child_node.move);
        if (old_board.state() == new_board.state()) {
            // Don't advance to unexpanded nodes
            if (!child_node.expanded()) {
                return false;
            }
            // Copy over the new root node to the correct place
            root() = child_node;
            return true;
        }
        // Check two moves deep from the root position
        if (start == active_half().root_idx() && advance_root_node(old_board, new_board, node.first_child_idx + i)) {
            return true;
        }
        old_board.undo_move();
    }

    return old_board.state() == new_board.state();
}

void GameTree::clear() {
    for (auto &half : halves_) {
        half.clear();
    }
    hash_table_.clear();
    tree_usage_ = 0;
    active_half_ = {};
    board_ = {};
    sum_depths_ = 0;
    nodes_in_path_.clear();
    history_.clear();
}

} // namespace search
