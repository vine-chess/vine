#include "value_network.hpp"

#include <algorithm>
#include <array>
#include <bit>
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
    std::array<const i16Vec *, 64> features;
    usize feature_count = 0;

    const auto stm = state.side_to_move;
    const auto king_sq = state.king(stm).lsb();

    const std::array<Bitboard, 2> threats = {state.pinned_threats_by(Color::WHITE),
                                             state.pinned_threats_by(Color::BLACK)};

    // Accumulate features for both sides, viewed from side-to-move's perspective
    for (PieceType piece = PieceType::PAWN; piece <= PieceType::KING; piece = PieceType(piece + 1)) {
        // Our pieces
        for (auto sq : state.piece_bbs[piece - 1] & state.occupancy(stm)) {
            features[feature_count++] = detail::feature(sq, piece, stm, stm, king_sq, threats[~stm], threats[stm]).data();
        }

        // Opponent pieces
        for (auto sq : state.piece_bbs[piece - 1] & state.occupancy(~stm)) {
            features[feature_count++] = detail::feature(sq, piece, ~stm, stm, king_sq, threats[stm], threats[~stm]).data();
        }
    }

    std::memcpy(accumulator.data(), network->ft_biases.data(), sizeof(accumulator));
    constexpr usize UNROLL = 4;
    usize j = 0;
    for (; j + UNROLL <= feature_count; j += UNROLL) {
        for (usize i = 0; i < L1_SIZE / VECTOR_SIZE; ++i) {
            for (usize k = 0; k < UNROLL; ++k) {
                accumulator[i] += features[j + k][i];
            }
        }
    }
    for (; j < feature_count; ++j) {
        for (usize i = 0; i < L1_SIZE / VECTOR_SIZE; ++i) {
            accumulator[i] += features[j][i];
        }
    }

    constexpr usize ACTIVATION_SHIFT = 9;
    constexpr usize PACK_SIZE = 64;
    constexpr f32 DEQUANTISATION = f32(1 << ACTIVATION_SHIFT) / (QA * QA * QB);

    std::array<u8, L1_SIZE / 2> activated;
    const auto activate = [&](usize i) {
        const auto left = util::clamp_scalar<i16>(accumulator[i / VECTOR_SIZE], 0, QA);
        const auto right = util::clamp_scalar<i16>(accumulator[(i + L1_SIZE / 2) / VECTOR_SIZE], 0, QA);
        return util::mulhi_round_epi16(left << (15 - ACTIVATION_SHIFT), right);
    };
    for (usize i = 0; i < L1_SIZE / 2; i += PACK_SIZE) {
        for (usize j = 0; j < PACK_SIZE / 2; j += VECTOR_SIZE) {
            util::storeu<u8>(activated.data() + i + 2 * j,
                             util::packus(activate(i + j), activate(i + j + PACK_SIZE / 2)));
        }
    }

    constexpr usize L1_UNROLL = 4;
    constexpr usize L2_REGS = L2_SIZE / L2_REG_SIZE;
    std::array<std::array<util::NativeVector<i32>, L2_REGS>, L1_UNROLL> sums{};
    const auto inputs = std::bit_cast<std::array<i32, L1_SIZE / 8>>(activated);
    for (usize i = 0; i < inputs.size(); ++i) {
        const auto input = util::set1<i32>(inputs[i]);
        for (usize k = 0; k < L2_REGS; ++k) {
            sums[i % L1_UNROLL][k] = util::dpbusd(sums[i % L1_UNROLL][k], input, network->l1_weights_vec[i][k]);
        }
    }

    for (usize j = 1; j < L1_UNROLL; ++j) {
        for (usize k = 0; k < L2_REGS; ++k) {
            sums[0][k] += sums[j][k];
        }
    }
    std::array<f32, L2_SIZE> l2;
    for (usize i = 0; i < L2_REGS; ++i) {
        auto v = util::convert_vector<f32, i32, L2_REG_SIZE>(sums[0][i]) * DEQUANTISATION +
                 util::loadu<f32, L2_REG_SIZE>(network->l1_biases.data() + L2_REG_SIZE * i);
        const auto scaled = util::fma<f32, L2_REG_SIZE>(v, util::set1<f32, L2_REG_SIZE>(1.0f / 6.0f),
                                                        util::set1<f32, L2_REG_SIZE>(0.5f));
        v *= util::clamp_scalar<f32, L2_REG_SIZE>(scaled, 0, 1);
        util::storeu<f32, L2_REG_SIZE>(l2.data() + L2_REG_SIZE * i, v);
    }

    std::array<f32, L3_SIZE> l3{};
    for (usize i = 0; i < L3_SIZE / L3_REG_SIZE; ++i) {
        auto v = util::loadu<f32, L3_REG_SIZE>(network->l2_biases.data() + L3_REG_SIZE * i);
        auto g = util::loadu<f32, L3_REG_SIZE>(network->l2_biases.data() + L3_REG_SIZE * i + L3_SIZE);

        // Matrix multiply l2 -> l3
        for (usize j = 0; j < L2_SIZE; ++j) {
            const auto l2_val = util::set1<f32, L3_REG_SIZE>(l2[j]);
            const auto w1 = network->l2_weights_vec[j][i];
            const auto w2 = network->l2_weights_vec[j][i + L3_SIZE / L3_REG_SIZE];
            v = util::fma<f32, L3_REG_SIZE>(l2_val, w1, v);
            g = util::fma<f32, L3_REG_SIZE>(l2_val, w2, g);
        }

        const auto g_scaled = util::fma<f32, L3_REG_SIZE>(g, util::set1<f32, L3_REG_SIZE>(1.0f / 6.0f),
                                                          util::set1<f32, L3_REG_SIZE>(0.5f));
        // Activate l3
        v *= util::clamp_scalar<f32, L3_REG_SIZE>(g_scaled, 0, 1);

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
