#include "network.hpp"

#include "../third_party/incbin.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>

INCBIN(EVAL, EVALFILE);

namespace network {

const Network *network = reinterpret_cast<const Network *>(gEVALData);

const util::MultiArray<i16Vec, L1_SIZE / VECTOR_SIZE> &feature(Square sq, PieceType piece, Color piece_color,
                                                               Color perspective) {
    usize flip = 0b111000 * perspective;
    return network->ft_weights_vec[piece_color != perspective][piece - 1][sq ^ flip];
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
            for (auto sq : state.piece_bbs[piece - 1] & stm_occ) {
                for (usize i = 0; i < L1_SIZE / VECTOR_SIZE; ++i) {
                    stm_accumulator[i] += feature(sq, piece, stm, stm)[i];
                    nstm_accumulator[i] += feature(sq, piece, stm, ~stm)[i];
                }
            }
            for (auto sq : state.piece_bbs[piece - 1] & nstm_occ) {
                for (usize i = 0; i < L1_SIZE / VECTOR_SIZE; ++i) {
                    stm_accumulator[i] += feature(sq, piece, ~stm, stm)[i];
                    nstm_accumulator[i] += feature(sq, piece, ~stm, ~stm)[i];
                }
            }
        }
    }

    // TODO: make this nicer, not sure how
    // CReLU
    const auto zero = util::set1_epi16<VECTOR_SIZE>(0);
    const auto one = util::set1_epi16<VECTOR_SIZE>(QA);

    util::SimdVector<i32, VECTOR_SIZE / 2> result_vec{};
    for (usize i = 0; i < L1_SIZE / VECTOR_SIZE; ++i) {
        const auto stm_clamped =
            util::min_epi16<VECTOR_SIZE>(util::max_epi16<VECTOR_SIZE>(stm_accumulator[i], zero), one);
        const auto nstm_clamped =
            util::min_epi16<VECTOR_SIZE>(util::max_epi16<VECTOR_SIZE>(nstm_accumulator[i], zero), one);
        result_vec += util::madd_epi16(stm_clamped, network->l1_weights_vec[0][i]);
        result_vec += util::madd_epi16(nstm_clamped, network->l1_weights_vec[1][i]);
    }
    const auto result = util::reduce_vector<i32, VECTOR_SIZE / 2>(result_vec) ;

    return (result + network->l1_biases[0]) * SCALE / (QA * QB);
}
} // namespace network
