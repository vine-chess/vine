#ifndef EVAL_VALUE_DENSE_GPU_CUH
#define EVAL_VALUE_DENSE_GPU_CUH

#include "../../board_cuda.cuh"
#include "../shared.hpp"

namespace network::value {

using namespace network::cuda_common;

constexpr i32 WARPS_PER_BLOCK = 4;
constexpr i32 THREADS_PER_BLOCK = network::cuda_common::WARP_SIZE * WARPS_PER_BLOCK;
constexpr usize L1_CHUNK_SIZE = 64;

static_assert((L1_SIZE / 2) % L1_CHUNK_SIZE == 0);
static_assert(network::cuda_common::WARP_SIZE == L2_SIZE * 2);

[[nodiscard]] __device__ __forceinline__ f32 clamp_f32(f32 value, f32 min_value, f32 max_value) {
    if (value < min_value) {
        return min_value;
    }
    if (value > max_value) {
        return max_value;
    }
    return value;
}

__global__ void evaluate_kernel(const CudaBoardInput *inputs, f32 *outputs, i32 count,
                                const CudaValueNetwork *network) {
    const i32 warp_in_block = threadIdx.x / WARP_SIZE;
    const u8 lane = threadIdx.x % WARP_SIZE;
    const i32 idx = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
    if (idx >= count) {
        return;
    }

    __shared__ i16 shared_l1_left[WARPS_PER_BLOCK][L1_CHUNK_SIZE];
    __shared__ i16 shared_l1_right[WARPS_PER_BLOCK][L1_CHUNK_SIZE];
    __shared__ u16 shared_l1_activations[WARPS_PER_BLOCK][L1_CHUNK_SIZE];
    __shared__ u16 shared_feature_indices[WARPS_PER_BLOCK][BOARD_SIZE];
    __shared__ i32 shared_l2_int[WARPS_PER_BLOCK][L2_SIZE];
    __shared__ f32 shared_l2[WARPS_PER_BLOCK][L2_SIZE];

    const CudaBoardInput &input = inputs[idx];
    i16 *l1_left = shared_l1_left[warp_in_block];
    i16 *l1_right = shared_l1_right[warp_in_block];
    u16 *l1_activations = shared_l1_activations[warp_in_block];
    u16 *feature_indices = shared_feature_indices[warp_in_block];
    const CompressedMailbox &pieces = input.pieces;
    i32 *l2_int = shared_l2_int[warp_in_block];
    f32 *l2 = shared_l2[warp_in_block];

    const i16 *ft_weights = reinterpret_cast<const i16 *>(&network->ft_weights);
    const i16 *ft_biases = reinterpret_cast<const i16 *>(&network->ft_biases);
    const i8 *l1_weights = reinterpret_cast<const i8 *>(&network->l1_weights);
    const f32 *l1_biases = reinterpret_cast<const f32 *>(&network->l1_biases);
    const f32 *l2_weights = reinterpret_cast<const f32 *>(&network->l2_weights);
    const f32 *l2_biases = reinterpret_cast<const f32 *>(&network->l2_biases);
    const f32 *l3_weights = reinterpret_cast<const f32 *>(&network->l3_weights);

    PackedBoard board = build_board_warp(pieces, lane);
    const PackedColor perspective = static_cast<PackedColor>(input.side_to_move);
    const i32 king_sq = lsb(piece_bb(board, PackedPieceType::KING, perspective));
    const i32 flip = (perspective == PackedColor::BLACK ? 0b111000 : 0) ^ (file_of(king_sq) >= 4 ? 0b111 : 0);
    const u64 white_threats = warp_broadcast(warp_or(pinned_threats_by_warp(board, pieces, PackedColor::WHITE, lane)));
    const u64 black_threats = warp_broadcast(warp_or(pinned_threats_by_warp(board, pieces, PackedColor::BLACK, lane)));
    __syncwarp();

    warp_for_each_bit(occupancy(board), lane, [&](int sq, bool active) {
        if (active) {
            const u8 piece_byte = pieces.at(static_cast<u8>(sq));
            const PackedPieceType piece = decode_piece_type(piece_byte);
            const PackedColor color = decode_color(piece_byte);
            const i32 defended = color == PackedColor::WHITE ? is_set(white_threats, sq) : is_set(black_threats, sq);
            const i32 threatened = color == PackedColor::WHITE ? is_set(black_threats, sq) : is_set(white_threats, sq);
            const i32 opposite_color = color != perspective;
            const u8 fc = ft_feature_class(defended, threatened, opposite_color, piece);
            feature_indices[lane] =
                static_cast<u16>((static_cast<u16>(fc) << 8) | static_cast<u16>(sq ^ flip));
        }
    });
    const int active_feature_count = __popcll(occupancy(board));

    __syncwarp();

    constexpr f32 DEQUANTISATION = 1.0f / (QA * QA * QB);

    if (lane < L2_SIZE) {
        l2_int[lane] = 0;
    }

    for (usize chunk_start = 0; chunk_start < L1_SIZE / 2; chunk_start += L1_CHUNK_SIZE) {
#pragma unroll
        for (usize i = lane; i < L1_CHUNK_SIZE; i += WARP_SIZE) {
            l1_left[i] = ft_biases[chunk_start + i];
            l1_right[i] = ft_biases[L1_SIZE / 2 + chunk_start + i];
        }

        __syncwarp();

        for (usize i = lane; i < L1_CHUNK_SIZE; i += WARP_SIZE) {
            i16 left = l1_left[i];
            i16 right = l1_right[i];
            for (int j = 0; j < active_feature_count; ++j) {
                const u16 packed = feature_indices[j];
                const u8 fc = static_cast<u8>(packed >> 8);
                const i32 sq_flipped = static_cast<i32>(packed & 0xff);
                const usize base = ft_offset<L1_SIZE>(fc, sq_flipped);
                left += ft_weights[base + chunk_start + i];
                right += ft_weights[base + L1_SIZE / 2 + chunk_start + i];
            }
            l1_left[i] = left;
            l1_right[i] = right;
        }

        __syncwarp();

#pragma unroll
        for (usize i = lane; i < L1_CHUNK_SIZE; i += WARP_SIZE) {
            const u16 left = static_cast<u16>(clamp_i32(l1_left[i], 0, QA));
            const u16 right = static_cast<u16>(clamp_i32(l1_right[i], 0, QA));
            l1_activations[i] = static_cast<u16>(left * right);
        }

        __syncwarp();

        const usize l2_idx = lane % L2_SIZE;
        i32 sum = 0;
#pragma unroll
        for (usize i = lane / L2_SIZE; i < L1_CHUNK_SIZE; i += 2) {
            sum += static_cast<i32>(l1_activations[i]) * l1_weights[(chunk_start + i) * L2_SIZE + l2_idx];
        }

        sum += __shfl_xor_sync(0xffffffff, sum, L2_SIZE);

        if (lane < L2_SIZE) {
            l2_int[lane] += sum;
        }

        __syncwarp();
    }

    __syncwarp();

    if (lane < L2_SIZE) {
        f32 value = l2_int[lane] * DEQUANTISATION + l1_biases[lane];
        value = hard_silu(value);
        l2[lane] = value;
    }

    __syncwarp();

    f32 final_sum = lane == 0 ? network->l3_bias : 0.0f;
    for (usize i = lane; i < L3_SIZE; i += WARP_SIZE) {
        f32 v = l2_biases[i];
        f32 g = l2_biases[i + L3_SIZE];

        for (usize j = 0; j < L2_SIZE; ++j) {
            v += l2[j] * l2_weights[j * (L3_SIZE * 2) + i];
            g += l2[j] * l2_weights[j * (L3_SIZE * 2) + i + L3_SIZE];
        }

        v *= hard_sigmoid(g);
        final_sum += v * l3_weights[i];
    }

    final_sum = warp_sum(final_sum);

    if (lane == 0) {
        outputs[idx] = final_sum;
    }
}

} // namespace network::value

#endif // EVAL_VALUE_DENSE_GPU_CUH
