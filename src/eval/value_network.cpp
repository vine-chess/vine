#include "value_network.hpp"
#include "value_network.cuh"

#include <algorithm>

namespace network::value {

const extern ValueNetwork *const network;

#ifdef MIXER_VALUE_NETWORK

namespace detail {

using MixerVector = std::array<f32, MIXER_SIZE>;
using MixerWeights = util::MultiArray<f32, MIXER_D, MIXER_D>;

[[nodiscard]] usize feature_index(Square sq, PieceType piece, Color piece_color, Color perspective, Square king_sq,
                                  Bitboard threats, Bitboard defences) {
    const usize flip = shared::mixer::feature_flip(perspective == Color::BLACK, king_sq.file() >= File::E);
    const usize defended = defences.is_set(sq);
    const usize threatened = threats.is_set(sq);
    const usize opposite_color = piece_color != perspective;
    const usize feature_class = shared::mixer::feature_class(defended, threatened, opposite_color, piece - 1);
    return shared::mixer::feature_index(feature_class, static_cast<u8>(sq), flip);
}

void add_feature(std::array<i32, MIXER_SIZE> &accumulator, const usize feature_idx) {
    for (usize i = 0; i < MIXER_SIZE; ++i) {
        accumulator[i] += network->ft_weights[feature_idx][i];
    }
}

template <shared::mixer::MixSide Side>
void apply_mix(MixerVector &x, const MixerWeights &weights) {
    MixerVector mix{};
    for (usize row = 0; row < MIXER_D; ++row) {
        for (usize col = 0; col < MIXER_D; ++col) {
            f32 sum = 0.0f;
            for (usize k = 0; k < MIXER_D; ++k) {
                if constexpr (Side == shared::mixer::MixSide::Left) {
                    sum += weights[k][row] * x[shared::mixer::matrix_index(k, col)];
                } else {
                    sum += x[shared::mixer::matrix_index(row, k)] * weights[col][k];
                }
            }
            mix[shared::mixer::matrix_index(row, col)] = sum;
        }
    }

    for (usize i = 0; i < MIXER_SIZE; ++i) {
        x[i] += shared::mixer::crelu(mix[i]);
    }
}

} // namespace detail

namespace cuda_detail {

void export_cuda_network(CudaValueNetwork &dst) {
    std::ranges::copy(network->ft_weights.flat_span(), dst.ft_weights.begin());
    std::ranges::copy(network->ft_biases, dst.ft_biases.begin());
    std::ranges::copy(network->wl1.flat_span(), dst.wl1.begin());
    std::ranges::copy(network->wr1.flat_span(), dst.wr1.begin());
    std::ranges::copy(network->wl2.flat_span(), dst.wl2.begin());
    std::ranges::copy(network->wr2.flat_span(), dst.wr2.begin());
    std::ranges::copy(network->value_weights, dst.value_weights.begin());
    dst.value_bias = network->value_bias;
}

} // namespace cuda_detail

f64 evaluate(const BoardState &state) {
    std::array<i32, MIXER_SIZE> accumulator{};
    for (usize i = 0; i < MIXER_SIZE; ++i) {
        accumulator[i] = network->ft_biases[i];
    }

    const auto stm = state.side_to_move;
    const auto king_sq = state.king(stm).lsb();
    const std::array<Bitboard, 2> threats = {state.pinned_threats_by(Color::WHITE),
                                             state.pinned_threats_by(Color::BLACK)};

    for (PieceType piece = PieceType::PAWN; piece <= PieceType::KING; piece = PieceType(piece + 1)) {
        for (auto sq : state.piece_bbs[piece - 1] & state.occupancy(stm)) {
            detail::add_feature(
                accumulator, detail::feature_index(sq, piece, stm, stm, king_sq, threats[~stm], threats[stm]));
        }

        for (auto sq : state.piece_bbs[piece - 1] & state.occupancy(~stm)) {
            detail::add_feature(
                accumulator, detail::feature_index(sq, piece, ~stm, stm, king_sq, threats[stm], threats[~stm]));
        }
    }

    std::array<f32, MIXER_SIZE> x{};
    for (usize i = 0; i < MIXER_SIZE; ++i) {
        x[i] = static_cast<f32>(std::clamp(accumulator[i], 0, static_cast<i32>(QA))) / static_cast<f32>(QA);
    }

    detail::apply_mix<shared::mixer::MixSide::Left>(x, network->wl1);
    detail::apply_mix<shared::mixer::MixSide::Right>(x, network->wr1);
    detail::apply_mix<shared::mixer::MixSide::Left>(x, network->wl2);
    detail::apply_mix<shared::mixer::MixSide::Right>(x, network->wr2);

    f32 value = network->value_bias;
    for (usize i = 0; i < MIXER_SIZE; ++i) {
        value += x[i] * network->value_weights[i];
    }
    return value;
}

#else

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

namespace cuda_detail {

void export_cuda_network(CudaValueNetwork &dst) {
    std::ranges::copy(network->ft_weights.flat_span(), dst.ft_weights.begin());
    std::ranges::copy(network->ft_biases, dst.ft_biases.begin());
    std::ranges::copy(network->l1_weights.flat_span(), dst.l1_weights.begin());
    std::ranges::copy(network->l1_biases, dst.l1_biases.begin());
    std::ranges::copy(network->l2_weights.flat_span(), dst.l2_weights.begin());
    std::ranges::copy(network->l2_biases, dst.l2_biases.begin());
    std::ranges::copy(network->l3_weights, dst.l3_weights.begin());
    dst.l3_bias = network->l3_biases[0];
}

} // namespace cuda_detail

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

#endif

} // namespace network::value
