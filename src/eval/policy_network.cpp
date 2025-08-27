#include "policy_network.hpp"
#include "../chess/move_gen.hpp"

#include <array>
#include <cstring>

namespace network::policy {

const extern PolicyNetwork *const network;

namespace detail {

[[nodiscard]] const util::MultiArray<i8Vec, L1_SIZE / VECTOR_SIZE> &feature(Square sq, PieceType piece,
                                                                            Color piece_color, Color perspective) {
    usize flip = 0b111000 * perspective;
    return network->ft_weights_vec[piece_color != perspective][piece - 1][sq ^ flip];
}

constexpr std::array<Bitboard, 64> ALL_DESTINATIONS = [] {
    std::array<Bitboard, 64> arr{};
    for (i32 sq = 0; sq < 64; ++sq) {
        arr[sq] = ROOK_RAYS[sq] | BISHOP_RAYS[sq] | KNIGHT_MOVES[sq] | KING_MOVES[sq];
    }
    return arr;
}();

constexpr std::array<usize, 65> OFFSETS = [] {
    std::array<usize, 65> off{};
    usize cur = 0;
    for (i32 sq = 0; sq < 64; ++sq) {
        off[sq] = cur;
        cur += static_cast<usize>(ALL_DESTINATIONS[sq].pop_count());
    }
    off[64] = cur; // Start of the promotion buckets
    return off;
}();

[[nodiscard]] i32 promo_kind_id(PieceType pt) {
    return static_cast<i32>(pt) - 2;
}

[[nodiscard]] usize move_output_idx(Color stm, Move move) {
    const i32 flipper = stm == Color::BLACK ? 56 : 0;
    const Square from = move.from() ^ flipper;
    const Square to = move.to() ^ flipper;
    if (move.is_promo()) {
        constexpr usize PROMO_STRIDE = 22;
        const i32 promo_id = 2 * from.file() + to.file();
        const i32 kind = promo_kind_id(move.promo_type());
        return OFFSETS[64] + static_cast<usize>(kind * PROMO_STRIDE + promo_id);
    } else {
        const u64 all = static_cast<u64>(ALL_DESTINATIONS[from]);
        const u64 below = to == 0 ? 0ull : (all & (to.to_bb() - 1ull));
        return OFFSETS[from] + static_cast<usize>(std::popcount(below));
    }
}

} // namespace detail

PolicyContext::PolicyContext(const BoardState &state) : stm_(state.side_to_move) {
    for (usize i = 0; i < L1_SIZE / VECTOR_SIZE; ++i) {
        feature_accumulator_[i] = util::convert_vector<i16, i8, VECTOR_SIZE>(network->ft_biases_vec[i]);
    }

    util::StaticVector<std::reference_wrapper<const util::MultiArray<i8Vec, L1_SIZE / VECTOR_SIZE>>, 32> features;

    // Accumulate features for both sides, viewed from side-to-move's perspective
    for (PieceType piece = PieceType::PAWN; piece <= PieceType::KING; piece = PieceType(piece + 1)) {
        // Our pieces
        for (auto sq : state.piece_bbs[piece - 1] & state.occupancy(stm_)) {
            features.push_back(std::ref(detail::feature(sq, piece, stm_, stm_)));
        }
        // Opponent pieces
        for (auto sq : state.piece_bbs[piece - 1] & state.occupancy(~stm_)) {
            features.push_back(std::ref(detail::feature(sq, piece, ~stm_, stm_)));
        }
    }

    // the actual adding of features to the accumulator is unrolled
    // to save on redundant loads and stores to the accumulator

    for (usize feature_idx = 0; auto unroll : {8, 1}) {
        for (; feature_idx + unroll <= features.size(); feature_idx += unroll) {
            for (usize i = 0; i < L1_SIZE / VECTOR_SIZE; ++i) {
                i16Vec sum = util::set1_epi16<VECTOR_SIZE>(0);
                for (usize j = 0; j < unroll; ++j) {
                    sum += util::convert_vector<i16, i8, VECTOR_SIZE>(features[feature_idx + j].get()[i]);
                }
                feature_accumulator_[i] += sum;
            }
        }
    }
}

f32 PolicyContext::logit(Move move) const {
    const usize idx = detail::move_output_idx(stm_, move);

    util::SimdVector<i32, VECTOR_SIZE / 2> sum{};
    const auto zero = util::set1_epi16<VECTOR_SIZE>(0);
    const auto one = util::set1_epi16<VECTOR_SIZE>(Q);

    for (usize i = 0; i < L1_SIZE / VECTOR_SIZE; ++i) {
        const auto clamped =
            util::min_epi16<VECTOR_SIZE>(util::max_epi16<VECTOR_SIZE>(feature_accumulator_[i], zero), one);
        sum += util::madd_epi16(clamped, util::convert_vector<i16, i8, VECTOR_SIZE>(network->l1_weights_vec[idx][i]));
    }

    const i32 dot = util::reduce_vector<i32, VECTOR_SIZE / 2>(sum);
    const i32 bias = network->l1_biases[idx];
    return static_cast<f32>(dot + bias) / static_cast<f32>(Q * Q);
}

} // namespace network::policy
