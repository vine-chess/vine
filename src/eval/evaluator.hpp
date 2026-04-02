#ifndef EVALUATOR_HPP
#define EVALUATOR_HPP

#include "gpu_queue.hpp"
#include "policy_network.hpp"
#include "value_network.hpp"

#include "../util/static_vector.hpp"
#include <algorithm>
#include <array>
#include <stdexcept>

namespace network {

static_assert(MAX_MOVES < 256);

class CpuPolicyContext {
  public:
    explicit CpuPolicyContext(const BoardState &state) : state_(state) {}

    void enqueue(Move move, PieceType moving_piece) {
        moves_.push_back(move);
        moving_pieces_.push_back(moving_piece);
    }

    void ready() {
        const auto ctx = policy::PolicyContext(state_);
        for (usize i = 0; i < moves_.size(); ++i) {
            logits_[i] = ctx.logit(moves_[i], moving_pieces_[i]);
        }
    }

    [[nodiscard]] f32 logit() {
        return logits_[next_idx_++];
    }

  private:
    const BoardState &state_;
    util::StaticVector<Move, MAX_MOVES> moves_;
    util::StaticVector<PieceType, MAX_MOVES> moving_pieces_;
    std::array<f32, MAX_MOVES> logits_{};
    usize next_idx_ = 0;
};

class QueuedGpuPolicyContext {
  public:
    explicit QueuedGpuPolicyContext(const BoardState &state) : state_(state) {}

    void enqueue(Move move, PieceType moving_piece) {
        moves_.push_back(move);
        moving_pieces_.push_back(moving_piece);
        ++move_count_;
    }

    void ready() {
        auto &queue = GlobalGpuPolicyQueue::get().queue();
        slot_ = queue.reserve_policy_slot(state_, static_cast<u32>(move_count_));

        for (usize i = 0; i < move_count_; ++i) {
            slot_.move_indices[i] = policy::move_output_idx(state_, moves_[i], moving_pieces_[i]);
        }

        queue.mark_ready(slot_);
        result_ = queue.wait_for_result(slot_);
    }

    [[nodiscard]] f32 logit() {
        const f32 out = result_.logits[next_idx_++];
        if (next_idx_ == move_count_) {
            GlobalGpuPolicyQueue::get().queue().mark_consumed(slot_);
        }
        return out;
    }

  private:
    const BoardState &state_;
    util::StaticVector<Move, MAX_MOVES> moves_;
    util::StaticVector<PieceType, MAX_MOVES> moving_pieces_;
    usize move_count_ = 0;
    usize next_idx_ = 0;
    GpuPolicyQueue::PolicySlot slot_{};
    GpuPolicyQueue::PolicyResult result_{};
};

class CpuEvaluator {
  public:
    [[nodiscard]] f64 value(const BoardState &state) const {
        return value::evaluate(state);
    }

    [[nodiscard]] CpuPolicyContext policy_context(const BoardState &state) const {
        return CpuPolicyContext(state);
    }
};

class QueuedGpuEvaluator {
  public:
    QueuedGpuEvaluator() {
        if (!value::cuda_available() || !policy::cuda_available()) {
            throw std::runtime_error("CUDA backend unavailable");
        }
        GlobalGpuValueQueue::get().queue().start();
        GlobalGpuPolicyQueue::get().queue().start();
    }

    static void set_batch_size(u32 batch_size) {
        GlobalGpuValueQueue::get().queue().set_batch_size(batch_size);
        GlobalGpuPolicyQueue::get().queue().set_batch_size(batch_size);
    }

    [[nodiscard]] f64 value(const BoardState &state) const {
        auto &queue = GlobalGpuValueQueue::get().queue();
        const auto slot = queue.reserve_value_slot(state);
        queue.mark_ready(slot);
        const auto result = queue.wait_for_result(slot);
        queue.mark_consumed(slot);
        return result;
    }

    [[nodiscard]] QueuedGpuPolicyContext policy_context(const BoardState &state) const {
        return QueuedGpuPolicyContext(state);
    }
};

} // namespace network

#endif // EVALUATOR_HPP
