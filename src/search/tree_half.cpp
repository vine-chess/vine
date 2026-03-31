#include "tree_half.hpp"
#include "node.hpp"
#include "../util/simd.hpp"

namespace search {

TreeHalf::TreeHalf(HalfIndex our_half) : our_half_(our_half), filled_size_(0) {}

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
    // safety margin to allow SIMD code to read past end of children and mask off invalid entries
    return filled_size() + n + util::NATIVE_VECTOR_BYTES <= nodes_.size();
}

void TreeHalf::clear_dangling_references() {
    for (auto &node : nodes_) {
        if (node.first_child_idx.half() != our_half_) {
            node.num_children = 0;
        }
    }
}

void TreeHalf::push_node(const Node &node) {
    score_sums_[filled_size_] = node.sum_of_scores;
    policy_scores_[filled_size_] = node.policy_score;
    visit_counts_[filled_size_] = node.num_visits;
    nodes_[filled_size_] = node;
    filled_size_ += 1;
}

NodeIndex TreeHalf::root_idx() {
    return {0, our_half_};
}

NodeReference TreeHalf::root_node() {
    return (*this)[root_idx()];
}

NodeReference TreeHalf::operator[](NodeIndex idx) {
    return NodeReference(nodes_[idx.index()], score_sums_[idx.index()], policy_scores_[idx.index()],
                         visit_counts_[idx.index()]);
}

NodeRange TreeHalf::range(NodeReference node) {
    const auto first = node.info.first_child_idx.index();
    const auto num_children = node.info.num_children;
    return NodeRange{
        .begin_ =
            NodeIterator{
                .score_sums_ = score_sums_.data(),
                .policy_scores_ = policy_scores_.data(),
                .visit_counts_ = visit_counts_.data(),
                .nodes_ = nodes_.data(),
                .index = first,

            },
        .end_ = {.end_idx = first + num_children},
    };
}

NodeIndex TreeHalf::construct_idx(u32 idx) const noexcept {
    return {idx, our_half_};
}

void TreeHalf::clear() {
    filled_size_ = 0;
}

} // namespace search
