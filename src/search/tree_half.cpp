#include "tree_half.hpp"
#include "node.hpp"

namespace search {

TreeHalf::TreeHalf(Index our_half) : our_half_(our_half), filled_size_(0) {}

void TreeHalf::set_node_capacity(usize capacity) {
    clear();
    auto resize = [=](auto &vector) {
        vector.clear();
        vector.shrink_to_fit();
        vector.resize(capacity);
    };
    resize(score_sums_);
    resize(policy_scores_);
    resize(visit_counts_);
    resize(nodes_);
}

usize TreeHalf::filled_size() const {
    return filled_size_;
}

bool TreeHalf::has_room_for(usize n) const {
    return filled_size() + n <= nodes_.size();
}

void TreeHalf::clear_dangling_references() {
    for (auto &node : nodes_) {
        if (node.info.first_child_idx.half() != our_half_) {
            node.info.num_children = 0;
        }
    }
}

void TreeHalf::push_node(const Node &node) {
    nodes_[filled_size_++] = node.info;
}

NodeIndex TreeHalf::root_idx() const {
    return {0, our_half_};
}

Node &TreeHalf::root_node() {
    return nodes_[0];
}

const Node &TreeHalf::root_node() const {
    return nodes_[0];
}

Node &TreeHalf::operator[](NodeIndex idx) {
    return nodes_[idx.index()];
}

const Node &TreeHalf::operator[](NodeIndex idx) const {
    return nodes_[idx.index()];
}

NodeIndex TreeHalf::construct_idx(u32 idx) const noexcept {
    return {idx, our_half_};
}

void TreeHalf::clear() {
    filled_size_ = 0;
}

} // namespace search
