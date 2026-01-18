#include "value_network.hpp"

#include <algorithm>
#include <array>
#include <cstring>

namespace network::value {

const extern ValueNetwork *const network;

namespace detail {

[[nodiscard]] const util::MultiArray<i16Vec, L1_SIZE / VECTOR_SIZE> &feature(Square sq, PieceType piece,
                                                                             Color piece_color, Color perspective,
                                                                             Square king_sq, Bitboard threats,
                                                                             Bitboard defences) {
    usize flip = 0b111000 * perspective ^ 0b000111 * (king_sq.file() >= File::E);
    return network
        ->ft_weights_vec[defences.is_set(sq)][threats.is_set(sq)][piece_color != perspective][piece - 1][sq ^ flip];
}

} // namespace detail

f64 evaluate(const BoardState &state) {
    std::array<i16Vec, L1_SIZE / VECTOR_SIZE> accumulator;
    std::memcpy(accumulator.data(), network->ft_biases.data(), sizeof(accumulator));

    const auto stm = state.side_to_move;
    const auto king_sq = state.king(stm).lsb();

    const std::array<Bitboard, 2> threats = {state.pinned_threats_by(Color::WHITE),
                                             state.pinned_threats_by(Color::BLACK)};

    // Accumulate features for both sides, viewed from side-to-move's perspective
    for (PieceType piece = PieceType::PAWN; piece <= PieceType::KING; piece = PieceType(piece + 1)) {
        // Our pieces
        for (auto sq : state.piece_bbs[piece - 1] & state.occupancy(stm)) {
            const auto feat = detail::feature(sq, piece, stm, stm, king_sq, threats[~stm], threats[stm]);
            for (usize i = 0; i < L1_SIZE / VECTOR_SIZE; ++i) {
                accumulator[i] += feat[i];
            }
        }

        // Opponent pieces
        for (auto sq : state.piece_bbs[piece - 1] & state.occupancy(~stm)) {
            const auto feat = detail::feature(sq, piece, ~stm, stm, king_sq, threats[stm], threats[~stm]);
            for (usize i = 0; i < L1_SIZE / VECTOR_SIZE; ++i) {
                accumulator[i] += feat[i];
            }
        }
    }

    const f32 dequantisation_constant = 1.0 / (QA * QA * QB);

    const i16 *l1 = reinterpret_cast<const i16 *>(accumulator.data());

    std::array<i32, L2_SIZE> l2_int{};
    for (usize i = 0; i < L1_SIZE / 2 / L2_REG_SIZE; ++i) {
        // Load register values for pairwise
        auto left = util::loadu<i16, L2_REG_SIZE>(l1 + L2_REG_SIZE * i);
        auto right = util::loadu<i16, L2_REG_SIZE>(l1 + L2_REG_SIZE * i + L1_SIZE / 2);

        // Clamp to [0, 1] (quantized)
        left = util::clamp_scalar<i16, L2_REG_SIZE>(left, 0, QA);
        right = util::clamp_scalar<i16, L2_REG_SIZE>(right, 0, QA);

        // Widen so pairwise doesnt overflow the i16s, using u16s here is neutral
        const auto left_widened = util::convert_vector<u16, i16, L2_REG_SIZE>(left);
        const auto right_widened = util::convert_vector<u16, i16, L2_REG_SIZE>(right);

        // Pairwise multiply the clamped values
        const auto activated = left_widened * right_widened;

        //  Matrix multiply l1 -> l2
        for (usize j = 0; j < L2_REG_SIZE; ++j) {
            const auto idx = i * L2_REG_SIZE + j;
            for (usize k = 0; k < L2_SIZE; ++k) {
                l2_int[k] += activated[j] * network->l1_weights[idx][k];
            }
        }
    }

    std::array<f32, L2_SIZE> l2;
    for (usize i = 0; i < L2_SIZE; ++i) {
        l2[i] = l2_int[i] * dequantisation_constant + network->l1_biases[i];
    }

    // Activate l2
    for (usize i = 0; i < L2_SIZE / L2_REG_SIZE; ++i) {
        auto v = util::loadu<f32, L2_REG_SIZE>(l2.data() + L2_REG_SIZE * i);
        v = util::clamp_scalar<f32, L2_REG_SIZE>(v, 0, 1);
        v *= v;
        util::storeu<f32, L2_REG_SIZE>(l2.data() + L2_REG_SIZE * i, v);
    }

    std::array<f32, L3_SIZE> l3{};
    for (usize i = 0; i < L3_SIZE / L3_REG_SIZE; ++i) {
        auto v = util::loadu<f32, L3_REG_SIZE>(network->l2_biases.data() + L3_REG_SIZE * i);

        // Matrix multiply l2 -> l3
        for (usize j = 0; j < L2_SIZE; ++j) {
            const auto l2_val = util::set1<f32, L3_REG_SIZE>(l2[j]);
            const auto w = network->l2_weights_vec[j][i];
            v = util::fma<f32, L3_REG_SIZE>(l2_val, w, v);
        }

        // Activate l3
        v = util::clamp_scalar<f32, L3_REG_SIZE>(v, 0, 1);
        v *= v;

        // Matrix multiply l3 -> out
        const auto l3_val = util::loadu<f32, L3_REG_SIZE>(l3.data() + L3_REG_SIZE * i);
        util::storeu<f32, L3_REG_SIZE>(l3.data() + L3_REG_SIZE * i,
                                       util::fma<f32, L3_REG_SIZE>(v, network->l3_weights_vec[i], l3_val));
    }

    f32 final_sum = network->l3_biases[0];
    for (usize i = 0; i < L3_SIZE; ++i) {
        final_sum += l3[i];
    }
    return final_sum;
}

} // namespace network::value
