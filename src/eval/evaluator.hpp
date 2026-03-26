#ifndef EVALUATOR_HPP
#define EVALUATOR_HPP

#include "gpu_queue.hpp"
#include "policy_network.hpp"
#include "policy_queue.hpp"
#include "value_network.hpp"
#include "value_queue.hpp"

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

class QueuedCpuPolicyContext {
  public:
    explicit QueuedCpuPolicyContext(const BoardState &state) : state_(state) {}

    void enqueue(Move move, PieceType moving_piece) {
        moves_.push_back(move);
        moving_pieces_.push_back(moving_piece);
    }

    void ready() {
        auto &queue = GlobalPolicyQueue::get().queue();
        slot_ = queue.reserve_policy_slot(state_, static_cast<u32>(moves_.size()));

        for (usize i = 0; i < moves_.size(); ++i) {
            slot_.move_indices[i] = policy::move_output_idx(state_, moves_[i], moving_pieces_[i]);
            slot_.piece_types[i] = moving_pieces_[i];
        }

        queue.mark_ready(slot_);
        result_ = queue.wait_for_result(slot_);
    }

    [[nodiscard]] f32 logit() {
        const f32 out = result_.logits[next_idx_++];
        if (next_idx_ == moves_.size()) {
            GlobalPolicyQueue::get().queue().mark_consumed(slot_);
        }
        return out;
    }

  private:
    const BoardState &state_;
    util::StaticVector<Move, MAX_MOVES> moves_;
    util::StaticVector<PieceType, MAX_MOVES> moving_pieces_;
    usize next_idx_ = 0;
    PolicyQueue::PolicySlot slot_{};
    PolicyQueue::PolicyResult result_{};
};

class GpuPolicyContext {
  public:
    explicit GpuPolicyContext(const BoardState &state) : state_(state) {
        if (!policy::cuda_available()) {
            throw std::runtime_error("policy CUDA backend unavailable");
        }
    }

    void enqueue(Move move, PieceType moving_piece) {
        moves_[move_count_] = move;
        moving_pieces_[move_count_] = moving_piece;
        ++move_count_;
    }

    void ready() {
        if (move_count_ == 0) {
            return;
        }

        std::copy(state_.piece_type_on_sq.begin(), state_.piece_type_on_sq.end(), std::begin(input_.pieces));
        input_.move_offset = 0;
        input_.move_count = static_cast<u8>(move_count_);
        input_.side_to_move = static_cast<u8>(state_.side_to_move);

        for (usize i = 0; i < move_count_; ++i) {
            move_indices_[i] = policy::move_output_idx(state_, moves_[i], moving_pieces_[i]);
        }

        policy::evaluate_many(&input_, move_indices_.data(), logits_.data(), 1);
    }

    [[nodiscard]] f32 logit() {
        return logits_[next_idx_++];
    }

  private:
    const BoardState &state_;
    util::StaticVector<Move, MAX_MOVES> moves_;
    util::StaticVector<PieceType, MAX_MOVES> moving_pieces_;
    std::array<u16, MAX_MOVES> move_indices_{};
    std::array<f32, MAX_MOVES> logits_{};
    usize move_count_ = 0;
    usize next_idx_ = 0;
    policy::CudaPolicyInput input_{};
};

class QueuedGpuPolicyContext {
  public:
    explicit QueuedGpuPolicyContext(const BoardState &state) : state_(state) {}

    void enqueue(Move move, PieceType moving_piece) {
        moves_[move_count_] = move;
        moving_pieces_[move_count_] = moving_piece;
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

class QueuedCpuEvaluator {
  public:
    QueuedCpuEvaluator() {
        GlobalValueQueue::get().queue().start();
        GlobalPolicyQueue::get().queue().start();
    }

    [[nodiscard]] f64 value(const BoardState &state) const {
        auto &queue = GlobalValueQueue::get().queue();
        const auto slot = queue.reserve_value_slot(state);
        queue.mark_ready(slot);
        const auto result = queue.wait_for_result(slot);
        queue.mark_consumed(slot);
        return result.value;
    }

    [[nodiscard]] QueuedCpuPolicyContext policy_context(const BoardState &state) const {
        return QueuedCpuPolicyContext(state);
    }
};

class GpuEvaluator {
  public:
    GpuEvaluator() {
        if (!value::cuda_available() || !policy::cuda_available()) {
            throw std::runtime_error("CUDA backend unavailable");
        }
    }

    [[nodiscard]] f64 value(const BoardState &state) const {
        value::CudaBoardInput input{};
        std::copy(state.piece_type_on_sq.begin(), state.piece_type_on_sq.end(), std::begin(input.pieces));
        input.side_to_move = static_cast<u8>(state.side_to_move);
        f32 output = 0.0f;
        value::evaluate_many(&input, &output, 1);
        return output;
    }

    [[nodiscard]] GpuPolicyContext policy_context(const BoardState &state) const {
        return GpuPolicyContext(state);
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
        return result.value;
    }

    [[nodiscard]] QueuedGpuPolicyContext policy_context(const BoardState &state) const {
        return QueuedGpuPolicyContext(state);
    }
};

} // namespace network

#endif // EVALUATOR_HPP
