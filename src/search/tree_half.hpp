#ifndef TREE_HALF_H
#define TREE_HALF_H

#include "../util/types.hpp"
#include "../chess/board_state.hpp"
#include <vector>

namespace search {

class Node;
class NodeIndex;

class TreeHalf {
  public:
    class Index {
      public:
        static constexpr u8 LOWER = 0;
        static constexpr u8 UPPER = 1;

        constexpr Index(u8 v = LOWER) noexcept : value_(v & 0x1u) {}
        constexpr Index(const Index &) noexcept = default;
        Index &operator=(const Index &) noexcept = default;

        [[nodiscard]] constexpr Index operator~() const noexcept {
            return {static_cast<u8>(value_ ^ 0x1u)};
        }

        [[nodiscard]] constexpr bool operator==(const Index &other) const noexcept {
            return value_ == other.value_;
        }
        [[nodiscard]] constexpr bool operator!=(const Index &other) const noexcept {
            return value_ != other.value_;
        }

        [[nodiscard]] constexpr operator u8() const noexcept {
            return value_;
        }

        [[nodiscard]] static constexpr Index lower() noexcept {
            return {LOWER};
        }
        [[nodiscard]] static constexpr Index upper() noexcept {
            return {UPPER};
        }

      private:
        u8 value_;
    };

    explicit TreeHalf(Index our_half);

    void set_node_capacity(usize capacity);

    [[nodiscard]] usize filled_size() const;
    [[nodiscard]] bool has_room_for(usize n) const;

    void clear();
    void clear_dangling_references();
    void push_node(const Node &node);

    [[nodiscard]] NodeIndex root_idx() const;
    [[nodiscard]] Node &root_node();
    [[nodiscard]] const Node &root_node() const;
    [[nodiscard]] Node &operator[](NodeIndex idx);
    [[nodiscard]] const Node &operator[](NodeIndex idx) const;
    [[nodiscard]] NodeIndex construct_idx(u32 idx) const noexcept;

  private:
    std::vector<Node> nodes_;
    usize filled_size_;
    Index our_half_;
};

} // namespace search

#endif // TREE_HALF_H