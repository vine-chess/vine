#ifndef EVALUATOR_HPP
#define EVALUATOR_HPP

#include "policy_network.hpp"
#include "policy_queue.hpp"
#include "value_queue.hpp"

#include "../util/assert.hpp"
#include "../util/static_vector.hpp"

namespace network {

class CpuPolicyContext {
  public:
    explicit CpuPolicyContext(const BoardState &state) : state_(state) {}

    CpuPolicyContext(const CpuPolicyContext &) = delete;
    CpuPolicyContext &operator=(const CpuPolicyContext &) = delete;
    CpuPolicyContext(CpuPolicyContext &&) = delete;
    CpuPolicyContext &operator=(CpuPolicyContext &&) = delete;

    void enqueue(Move move, PieceType moving_piece) {
        vine_assert(!ready_);
        moves_.push_back(move);
        moving_pieces_.push_back(moving_piece);
    }

    void ready() {
        vine_assert(!ready_);

        ready_ = true;
        next_idx_ = 0;

        auto &queue = GlobalPolicyQueue::get().queue();

        slot_ = queue.reserve_policy_slot(state_, static_cast<u32>(moves_.size()));

        for (usize i = 0; i < moves_.size(); ++i) {
            slot_.move_indices[i] = policy::move_output_idx(state_, moves_[i], moving_pieces_[i]);
        }
        for (usize i = 0; i < moving_pieces_.size(); ++i) {
            slot_.piece_types[i] = moving_pieces_[i];
        }

        queue.mark_ready(slot_);
        result_ = queue.wait_for_result(slot_);
    }

    [[nodiscard]] f32 logit() {
        vine_assert(ready_);
        vine_assert(next_idx_ < moves_.size());

        const usize idx = next_idx_++;
        const f32 out = result_.logits[idx];

        if (next_idx_ == moves_.size()) {
            GlobalPolicyQueue::get().queue().mark_consumed(slot_);
        }

        return out;
    }

  private:
    const BoardState &state_;
    util::StaticVector<Move, 256> moves_;
    util::StaticVector<PieceType, 256> moving_pieces_;

    PolicyQueue::PolicySlot slot_{};
    PolicyQueue::PolicyResult result_{};

    usize next_idx_ = 0;
    bool ready_ = false;
};

class ValueEvaluator {
public:
    [[nodiscard]] f64 value(const BoardState &state) const {
        auto &queue = GlobalValueQueue::get().queue();

        const auto slot = queue.reserve_value_slot(state);
        queue.mark_ready(slot);

        const auto result = queue.wait_for_result(slot);
        queue.mark_consumed(slot);

        return result.value;
    }
};

class CpuEvaluator {
  public:
    [[nodiscard]] f64 value(const BoardState &state) const {
        return ValueEvaluator{}.value(state);
    }

    [[nodiscard]] CpuPolicyContext policy_context(const BoardState &state) const {
        return CpuPolicyContext(state);
    }
};

} // namespace network

#endif // EVALUATOR_HPP