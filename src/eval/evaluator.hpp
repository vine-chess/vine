#ifndef EVALUATOR_HPP
#define EVALUATOR_HPP

#include "policy_network.hpp"
#include "value_network.hpp"

#include "../util/assert.hpp"
#include "../util/static_vector.hpp"

namespace network {

class CpuPolicyContext {
  public:
    explicit CpuPolicyContext(const BoardState &state) : ctx_(state) {}
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
        ready_ = true;
        next_idx_ = 0;
    }

    [[nodiscard]] f32 logit() {
        vine_assert(ready_);
        vine_assert(next_idx_ < moves_.size());
        const usize idx = next_idx_++;
        return ctx_.logit(moves_[idx], moving_pieces_[idx]);
    }

  private:
    policy::PolicyContext ctx_;
    util::StaticVector<Move, 256> moves_;
    util::StaticVector<PieceType, 256> moving_pieces_;
    usize next_idx_ = 0;
    bool ready_ = false;
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

} // namespace network

#endif // EVALUATOR_HPP
