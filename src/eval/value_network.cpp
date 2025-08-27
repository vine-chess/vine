#include "value_network.hpp"
#include "../util/static_vector.hpp"
#include <array>
#include <cstring>

namespace network::value {

const extern ValueNetwork *const network;

namespace detail {

[[nodiscard]] const util::MultiArray<i16Vec, L1_SIZE / VECTOR_SIZE> &feature(Square sq, PieceType piece,
                                                                             Color piece_color, Color perspective,
                                                                             Square king_sq) {
    usize flip = 0b111000 * perspective ^ 0b000111 * (king_sq.file() >= File::E);
    return network->ft_weights_vec[piece_color != perspective][piece - 1][sq ^ flip];
}

} // namespace detail

f64 evaluate(const BoardState &state) {
    std::array<i16Vec, L1_SIZE / VECTOR_SIZE> accumulator;
    std::memcpy(accumulator.data(), network->ft_biases.data(), sizeof(accumulator));

    const auto stm = state.side_to_move;
    const auto king_sq = state.king(stm).lsb();

    util::StaticVector<std::reference_wrapper<const util::MultiArray<i16Vec, L1_SIZE / VECTOR_SIZE>>, 32> features;
    // Accumulate features for both sides, viewed from side-to-move's perspective
    for (PieceType piece = PieceType::PAWN; piece <= PieceType::KING; piece = PieceType(piece + 1)) {
        // Our pieces
        for (auto sq : state.piece_bbs[piece - 1] & state.occupancy(stm)) {
            features.push_back(std::ref(detail::feature(sq, piece, stm, stm, king_sq)));
        }
        // Opponent pieces
        for (auto sq : state.piece_bbs[piece - 1] & state.occupancy(~stm)) {
            features.push_back(std::ref(detail::feature(sq, piece, ~stm, stm, king_sq)));
        }
    }
    usize feature_idx = 0;
    constexpr auto UNROLL = 8;
    for (; feature_idx + UNROLL <= features.size(); feature_idx += UNROLL) {
        for (usize i = 0; i < L1_SIZE / VECTOR_SIZE; ++i) {
            i16Vec sum = util::set1_epi16<VECTOR_SIZE>(0);
            for (usize j = 0; j < UNROLL; ++j) {
                sum += features[feature_idx + j].get()[i];
            }
            accumulator[i] += sum;
        }
    }
    for (; feature_idx < features.size(); feature_idx += 1) {
        for (usize i = 0; i < L1_SIZE / VECTOR_SIZE; ++i) {
            accumulator[i] += features[feature_idx].get()[i];
        }
    }

    util::SimdVector<i32, VECTOR_SIZE / 2> sum{};
    const auto zero = util::set1_epi16<VECTOR_SIZE>(0);
    const auto one = util::set1_epi16<VECTOR_SIZE>(QA);

    for (usize i = 0; i < L1_SIZE / VECTOR_SIZE; ++i) {
        const auto clamped = util::min_epi16<VECTOR_SIZE>(util::max_epi16<VECTOR_SIZE>(accumulator[i], zero), one);
        sum += util::madd_epi16(clamped, clamped * network->l1_weights_vec[i]);
    }

    const i32 dot = util::reduce_vector<i32, VECTOR_SIZE / 2>(sum);
    const i32 bias = network->l1_biases[0];
    return (dot / static_cast<f64>(QA) + bias) / static_cast<f64>(QA * QB);
}

} // namespace network::value
