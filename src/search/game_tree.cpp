#include "game_tree.hpp"
#include "../eval/value/cpu.hpp"
#include "../uci/uci.hpp"
#include "../util/assert.hpp"
#include "../util/math.hpp"

namespace search {

GameTree::GameTree()
    : halves_({TreeHalf(search::HalfIndex::LOWER), TreeHalf(search::HalfIndex::UPPER)}),
      active_half_(search::HalfIndex::LOWER) {
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

NodeReference GameTree::root() {
    return active_half().root_node();
}

NodeIndex GameTree::root_idx() {
    return active_half().root_idx();
}

NodeReference GameTree::node_at(NodeIndex idx) {
    return halves_[idx.half()][idx.index()];
}

[[nodiscard]] u32 GameTree::sum_depths() const {
    return sum_depths_;
}

u64 GameTree::tree_usage() const {
    return tree_usage_;
}

void GameTree::set_use_gini(bool use_gini) {
    use_gini_ = use_gini;
}

const BoardState &GameTree::state() const {
    return board_.state();
}

f64 GameTree::cpuct(NodeIndex node_idx, NodeReference node) {
    f64 base = node_idx == active_half().root_idx() ? ROOT_EXPLORATION_CONSTANT : EXPLORATION_CONSTANT;
    base *= 1.0 + std::log((node.num_visits + CPUCT_VISIT_SCALE) / static_cast<f64>(CPUCT_VISIT_SCALE_DIVISOR));
    if (use_gini_) {
        base *= std::min<f64>(GINI_MAXIMUM,
                              GINI_BASE - GINI_MULTIPLIER * std::log(node.info.gini_impurity / 255.0 + 0.001));
    }
    return base;
}

NodeIndex GameTree::pick_highest_puct(NodeReference parent, f64 exploration_constant) {
    const auto VECTOR_SIZE = 16;
    const auto first_child = parent.info.first_child_idx;
    const auto num_children = parent.info.num_children;
    const f64 u_scale = exploration_constant * std::sqrt(parent.num_visits);
    const auto u_scale_vector = util::set1<f32, VECTOR_SIZE>(u_scale);
    const f64 parent_q = parent.q();
    const auto parent_q_vector = util::set1<f32, VECTOR_SIZE>(parent_q);

    usize i = 0;
    auto children = get_children(parent);

    const auto LOWEST_POLICY = -std::numeric_limits<f32>::max();
    const auto LOWEST_POLICY_VECTOR = util::set1<f32, VECTOR_SIZE>(LOWEST_POLICY);
    util::SimdVector<f32, VECTOR_SIZE> best_puct = LOWEST_POLICY_VECTOR;
    util::SimdVector<u32, VECTOR_SIZE> best_indices = util::set1<u32, VECTOR_SIZE>(0);
    util::SimdVector<u32, VECTOR_SIZE> indices;
    for (usize i = 0; i < VECTOR_SIZE; ++i) {
        indices[i] = i;
    }

    auto iteration = [&](usize i) {
        auto child = children[i];

        const auto scores =
            util::convert_vector<f32, f64, VECTOR_SIZE>(util::loadu<f64, VECTOR_SIZE>(&child.sum_of_scores));
        const auto visits = util::loadu<u32, VECTOR_SIZE>(&child.num_visits);
        const auto policies = util::loadu<f32, VECTOR_SIZE>(&child.policy_score);

        const auto visitsf = util::convert_vector<f32, u32, VECTOR_SIZE>(visits);
        const auto u_base = policies / (visitsf + 1.0f);
        const auto child_q = scores / visitsf;
        const auto q = util::select_vector32<f32, VECTOR_SIZE>(parent_q_vector, 1.0 - child_q, visits != 0);
        return u_base * u_scale_vector + q;
    };

    for (; i + VECTOR_SIZE <= num_children; i += VECTOR_SIZE, indices += VECTOR_SIZE) {
        const auto puct = iteration(i);
        best_indices = util::select_vector32<f32, VECTOR_SIZE>(best_indices, indices, puct > best_puct);
        best_puct = util::max<f32, VECTOR_SIZE>(best_puct, puct);
    }
    if (i < num_children) {
        const auto puct = iteration(i);
        const auto mask = (indices < num_children) & (puct > best_puct);
        best_indices = util::select_vector32<f32, VECTOR_SIZE>(best_indices, indices, mask);
        best_puct = util::select_vector32<f32, VECTOR_SIZE>(best_puct, puct, mask);
    }

    f64 best_score = best_puct[0];
    usize best = 0;
    for (usize i = 1; i < VECTOR_SIZE; ++i) {
        if (best_puct[i] > best_score) {
            best = i;
            best_score = best_puct[i];
        }
    }

    return first_child + best_indices[best];
}

void GameTree::backpropagate_terminal_state(NodeIndex node_idx, TerminalState child_terminal_state) {
    auto node = node_at(node_idx);
    switch (child_terminal_state.flag()) {
    case TerminalState::Flag::LOSS: { // If a child node is lost, then it's a win for us
        // Ensure that if we already had a shorter mate we preserve it
        const auto current_mate_distance =
            node.info.terminal_state.is_win() ? node.info.terminal_state.distance_to_terminal() : 255;
        node.info.terminal_state =
            TerminalState::win(std::min<u8>(current_mate_distance, child_terminal_state.distance_to_terminal() + 1));
        break;
    }
    case TerminalState::Flag::WIN: { // If a child node is won, it's a loss for us if all of its siblings are also won
        u8 longest_loss = 0;
        for (auto sibling : get_children(node)) {
            const auto terminal_state = sibling.info.terminal_state;
            if (terminal_state.flag() != TerminalState::Flag::WIN) {
                return;
            }
            longest_loss = std::max(longest_loss, terminal_state.distance_to_terminal());
        }
        node.info.terminal_state = TerminalState::loss(longest_loss + 1);
        break;
    }
    default:
        break;
    }
}

void GameTree::backpropagate_score(f64 score) {
    vine_assert(!nodes_in_path_.empty());

    auto cp_score =
        static_cast<i32>(network::value::EVAL_SCALE * util::math::inverse_sigmoid(std::clamp(score, 0.001, 0.999)));
    auto child_terminal_state = TerminalState::none();

    while (!nodes_in_path_.empty()) {
        const auto node_idx = nodes_in_path_.pop_back();

        // A node's score is the average of all of its children's score
        auto node = node_at(node_idx);
        node.sum_of_scores += score;
        node.num_visits++;
        hash_table_.update(board_.state().hash_key, node.q(), node.num_visits);

        // If a terminal state from the child score exists, then we try to backpropagate it to this node
        if (!child_terminal_state.is_none()) {
            backpropagate_terminal_state(node_idx, child_terminal_state);
        }

        // If this node has a terminal state (either from backpropagation or it is terminal), we save it for the parent
        // node to try to use it
        if (!node.info.terminal_state.is_none()) {
            child_terminal_state = node.info.terminal_state;
        }

        // Negate the score to match the perspective of the node
        score = 1.0 - score;
        cp_score = -cp_score;

        if (!nodes_in_path_.empty()) {
            board_.undo_move();
            // Update the history for this move to influence new node policy scores
            if (child_terminal_state.is_none()) {
                history_.entry(board_.state(), node.info.move).update(cp_score);
            }
        }
    }
}

void GameTree::reset_to_root() {
    if (!nodes_in_path_.empty()) {
        board_.undo_n_moves(nodes_in_path_.size() - 1);
        nodes_in_path_.clear();
    }
}

NodeRange GameTree::get_children(NodeReference node) {
    return halves_[node.info.first_child_idx.half()].range(node);
}

bool GameTree::fetch_children(NodeIndex node_idx) {
    auto node = node_at(node_idx);
    // Don't do anything if the node's children already exist in our half
    if (node.info.first_child_idx.half() == active_half_) {
        return true;
    }

    // Check if we need to the active tree half
    vine_assert(node.info.num_children > 0);
    if (!active_half().has_room_for(node.info.num_children)) {
        return false;
    }

    // Copy over the children from the other tree half to this half
    for (auto child : get_children(node)) {
        active_half().push_node(child);
    }
    node.info.first_child_idx = active_half().construct_idx(active_half().filled_size() - node.info.num_children);

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

    auto node = node_at(start);
    if (!node.expanded()) {
        return false;
    }

    auto children = get_children(node);
    for (u16 i = 0; i < node.info.num_children; ++i) {
        auto child_node = children[i];
        // Ensure this move leads to the same resulting position
        old_board.make_move(child_node.info.move);
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
        if (start == active_half().root_idx() &&
            advance_root_node(old_board, new_board, node.info.first_child_idx + i)) {
            return true;
        }
        old_board.undo_move();
    }

    return old_board.state() == new_board.state();
}

void GameTree::inject_dirichlet_noise(NodeIndex node_idx) {
    auto node = node_at(node_idx);
    vine_assert(node_idx == active_half().root_idx());

    std::vector<f64> noise;
    noise.reserve(node.info.num_children);

    // Generate a distribution of random numbers and normalize
    f64 sum = 0.0f;
    for (usize i = 0; i < node.info.num_children; i++) {
        noise.push_back(rng::next_f64_gamma(dirichlet_alpha_));
        sum += noise.back();
    }
    for (usize i = 0; i < node.info.num_children; i++) {
        noise[i] /= sum;
    }

    // Mix in the Dirichlet noise with the policy priors
    for (u16 i = 0; i < node.info.num_children; ++i) {
        auto child = node_at(node.info.first_child_idx + i);
        child.policy_score =
            static_cast<f32>((1.0 - dirichlet_epsilon_) * child.policy_score + dirichlet_epsilon_ * noise[i]);
    }
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
