#include "network.hpp"

#include "../third_party/incbin.h"
#include <algorithm>
#include <array>
#include <cstring>

#define EVALFILE "net.nnue"
INCBIN(EVAL, EVALFILE);

const Network *network = reinterpret_cast<const Network *>(gEVALData);

usize feature_idx(Square sq, PieceType piece, Color piece_color, Color perspective) {
    usize flip = 0b111000 * perspective;
    return piece_color * 6 * 64 + piece * 64 + sq;
}

i32 evaluate(const BoardState &state) {
    std::array<i16Vec, L1_SIZE / VECTOR_SIZE> stm_accumulator, nstm_accumulator;
    std::memcpy(stm_accumulator.data(), network->ft_biases.data(), sizeof(stm_accumulator));
    std::memcpy(nstm_accumulator.data(), network->ft_biases.data(), sizeof(nstm_accumulator));

    const auto stm = state.side_to_move;
    { // add all the features
        const auto stm_occ = state.side_bbs[stm];
        const auto nstm_occ = state.side_bbs[~stm];
        for (int pti = PieceType::PAWN; pti <= PieceType::KING; ++pti) {
            const auto piece = PieceType(pti);
            for (auto sq : state.piece_bbs[piece] & stm_occ) {
                for (usize i = 0; i < L1_SIZE / VECTOR_SIZE; ++i) {
                    stm_accumulator[i] += network->ft_weights_vec[feature_idx(sq, piece, stm, stm)][i];
                    nstm_accumulator[i] += network->ft_weights_vec[feature_idx(sq, piece, stm, ~stm)][i];
                }
            }
            for (auto sq : state.piece_bbs[piece] & nstm_occ) {
                for (usize i = 0; i < L1_SIZE / VECTOR_SIZE; ++i) {
                    stm_accumulator[i] += network->ft_weights_vec[feature_idx(sq, piece, ~stm, stm)][i];
                    nstm_accumulator[i] += network->ft_weights_vec[feature_idx(sq, piece, ~stm, ~stm)][i];
                }
            }
        }
    }

    util::SimdVector<i32, VECTOR_SIZE / 2> result_vec{};
    // TODO: make this nicer, not sure how
    // CReLU
    for (usize i = 0; i < L1_SIZE / VECTOR_SIZE; ++i) {
        const auto stm_val = util::max_epi16({}, stm_accumulator[i] * network->l1_weights[0][i]);
        const auto nstm_val = util::max_epi16({}, nstm_accumulator[i] * network->l1_weights[1][i]);
        result_vec += util::convert_vector<i32, i16, VECTOR_SIZE / 2>(util::lower_half<i16, VECTOR_SIZE>(stm_val));
        result_vec += util::convert_vector<i32, i16, VECTOR_SIZE / 2>(util::upper_half<i16, VECTOR_SIZE>(stm_val));
        result_vec += util::convert_vector<i32, i16, VECTOR_SIZE / 2>(util::lower_half<i16, VECTOR_SIZE>(nstm_val));
        result_vec += util::convert_vector<i32, i16, VECTOR_SIZE / 2>(util::upper_half<i16, VECTOR_SIZE>(nstm_val));
    }
    const auto result = util::reduce_vector<i32, VECTOR_SIZE / 2>(result_vec) + network->l1_biases[0];

    return result * SCALE / (QA * QB);
}
