#ifndef EVALUATOR_HPP
#define EVALUATOR_HPP

#include "../chess/move_gen.hpp"
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
