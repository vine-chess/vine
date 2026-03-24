#ifndef VINE_EVAL_BATCHER_H
#define VINE_EVAL_BATCHER_H

#define BATCH_SIZE 64

#include "../chess/move_gen.hpp"
#include "../util/types.hpp"
#include <array>
#include <atomic>
#include <span>

namespace network {

class PolicyQueue {
  public:
    struct PolicySlot {
        u32 board_idx;
        std::span<u16> move_indices;
    };

    struct BatchResult {
        std::array<f32, MAX_MOVES> logits;
    };

    PolicyQueue() : policy_indices_(0), move_indices_({}) {}

    // Returns a structure with indexing information of this request in the current batch
    [[nodiscard]] PolicySlot get_policy_slot(u32 len) const {
        const auto old = policy_indices_.fetch_add(len | (1ull << 32));
        if (++num_enqueued_items_ > BATCH_SIZE) {
            throw std::runtime_error("double buffer time?");
        }
        return {old >> 32, std::span{&move_indices_[u32(old)], len}};
    }

    [[nodiscard]] bool is_full() const {
        return num_enqueued_items_ >= BATCH_SIZE;
    }

    void mark_as_completed() {
        policy_indices_ = 0;
    }

  private:
    std::array<BatchResult, BATCH_SIZE> batch_results_;
    std::array<u16, MAX_MOVES * BATCH_SIZE> move_indices_;
    std::atomic<u64> policy_indices_;
    std::atomic<u64> num_enqueued_items_;
};

class GlobalPolicyQueue {
  public:
    GlobalPolicyQueue() : queue_() {}
    ~GlobalPolicyQueue() {}

    static GlobalPolicyQueue get() const {
        static GlobalPolicyQueue instance;
        return instance;
    }

  private:
    PolicyQueue queue_;
};

} // namespace network

#endif // VINE_EVAL_BATCHER_H