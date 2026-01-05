#ifndef TREE_HALF_H
#define TREE_HALF_H

#include "../util/types.hpp"
#include "node.hpp"
#include "node_index.hpp"
#include <vector>

namespace search {

class TreeHalf {
  public:

    explicit TreeHalf(HalfIndex our_half);

    void set_node_capacity(usize capacity);

    [[nodiscard]] usize filled_size() const;
    [[nodiscard]] bool has_room_for(usize n) const;

    void clear();
    void clear_dangling_references();
    void push_node(const Node &node);

    [[nodiscard]] NodeIndex root_idx();
    [[nodiscard]] NodeReference root_node();
    [[nodiscard]] NodeReference operator[](NodeIndex idx);
    [[nodiscard]] NodeIndex construct_idx(u32 idx) const noexcept;

    [[nodiscard]] NodeRange range(NodeReference node);

  private:
    // Sums of all scores that have been propagated back to each node
    std::vector<f64> score_sums_;
    // Policy given to each node by its parent node
    std::vector<f32> policy_scores_;
    // Number of times each node has been visited
    std::vector<u32> visit_counts_;

    std::vector<NodeInfo> nodes_;
    usize filled_size_;
    HalfIndex our_half_;
};

} // namespace search

#endif // TREE_HALF_H
