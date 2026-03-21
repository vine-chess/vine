#include "../chess/board_state.hpp"

#include "cuda_common.cuh"
#include "policy_network.cuh"

namespace network::policy {

namespace {

constexpr usize ACTIVATED_SIZE = L1_SIZE / 2;
constexpr i32 WARPS_PER_BLOCK = 4;
constexpr i32 THREADS_PER_BLOCK = network::cuda_common::WARP_SIZE * WARPS_PER_BLOCK;
constexpr i32 MOVES_PER_ROUND = 2;
constexpr usize L1_CHUNK_SIZE = 64;
static_assert(ACTIVATED_SIZE % L1_CHUNK_SIZE == 0);
using namespace network::cuda_common;

[[nodiscard]] __device__ u64 threats_by_warp(const PackedBoard &board, const ColoredPiece *pieces, PackedColor color,
                                             i32 lane) {
    const i32 king_sq = lsb(piece_bb(board, PackedPieceType::KING, color));
    const u64 occ = occupancy(board) ^ piece_bb(board, PackedPieceType::KING, opposite(color));

    u64 threats = lane == 0 ? king_attacks(king_sq) : 0;
    for (i32 sq = lane; sq < BOARD_SIZE; sq += WARP_SIZE) {
        if (decode_color(pieces[sq]) != color) {
            continue;
        }

        switch (decode_piece_type(pieces[sq])) {
        case PackedPieceType::NONE:
            break;
        case PackedPieceType::PAWN:
            threats |= pawn_attacks(sq, color);
            break;
        case PackedPieceType::KNIGHT:
            threats |= knight_attacks(sq);
            break;
        case PackedPieceType::BISHOP:
            threats |= bishop_attacks(sq, occ);
            break;
        case PackedPieceType::ROOK:
            threats |= rook_attacks(sq, occ);
            break;
        case PackedPieceType::QUEEN:
            threats |= bishop_attacks(sq, occ) | rook_attacks(sq, occ);
            break;
        case PackedPieceType::KING:
            break;
        }
    }

    return threats;
}

__global__ void evaluate_kernel(const CudaPolicyInput *inputs, const u16 *move_indices, f32 *outputs,
                                i32 position_count, const cuda_detail::CudaPolicyNetwork *network) {
    const i32 warp_in_block = threadIdx.x / WARP_SIZE;
    const i32 lane = threadIdx.x % WARP_SIZE;
    const i32 position_idx = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
    if (position_idx >= position_count) {
        return;
    }

    __shared__ i16 shared_l1_left[WARPS_PER_BLOCK][L1_CHUNK_SIZE];
    __shared__ i16 shared_l1_right[WARPS_PER_BLOCK][L1_CHUNK_SIZE];
    __shared__ u16 shared_activated_chunk[WARPS_PER_BLOCK][L1_CHUNK_SIZE];
    __shared__ u8 shared_feature_classes[WARPS_PER_BLOCK][BOARD_SIZE];
    __shared__ u64 shared_occupied[WARPS_PER_BLOCK];

    const CudaPolicyInput &input = inputs[position_idx];
    const ColoredPiece *pieces = input.pieces;
    i16 *l1_left = shared_l1_left[warp_in_block];
    i16 *l1_right = shared_l1_right[warp_in_block];
    u16 *activated_chunk = shared_activated_chunk[warp_in_block];
    u8 *feature_classes = shared_feature_classes[warp_in_block];
    i32 flip = 0;

    {
        const PackedColor perspective = static_cast<PackedColor>(input.side_to_move);
        const PackedBoard board = build_board_warp(pieces, lane);
        const i32 king_sq = lsb(piece_bb(board, PackedPieceType::KING, perspective));
        flip = (perspective == PackedColor::BLACK ? 0b111000 : 0) ^ (file_of(king_sq) >= 4 ? 0b111 : 0);
        const u64 white_threats = warp_broadcast(warp_or(threats_by_warp(board, pieces, PackedColor::WHITE, lane)));
        const u64 black_threats = warp_broadcast(warp_or(threats_by_warp(board, pieces, PackedColor::BLACK, lane)));
        if (lane == 0) {
            shared_occupied[warp_in_block] = occupancy(board);
        }

        for (i32 sq = lane; sq < BOARD_SIZE; sq += WARP_SIZE) {
            const PackedPieceType piece = decode_piece_type(pieces[sq]);
            if (piece == PackedPieceType::NONE) {
                continue;
            }

            const PackedColor color = decode_color(pieces[sq]);
            const i32 defended =
                perspective == PackedColor::WHITE ? is_set(white_threats, sq) : is_set(black_threats, sq);
            const i32 threatened =
                perspective == PackedColor::WHITE ? is_set(black_threats, sq) : is_set(white_threats, sq);
            const i32 opposite_color = color != perspective;
            feature_classes[sq] = ft_feature_class(defended, threatened, opposite_color, piece);
        }
    }

    const i8 *ft_weights = reinterpret_cast<const i8 *>(&network->ft_weights);
    const i8 *ft_biases = reinterpret_cast<const i8 *>(&network->ft_biases);
    const i8 *l1_weights = reinterpret_cast<const i8 *>(&network->l1_weights);
    const i8 *l1_biases = reinterpret_cast<const i8 *>(&network->l1_biases);
    __syncwarp();

    constexpr f32 DEQUANTISATION = 1.0f / static_cast<f32>(Q * Q * Q);
    constexpr i32 MOVE_ROUND_STRIDE = MOVES_PER_ROUND * WARP_SIZE;
    for (i32 move_local_0 = lane; move_local_0 < input.move_count; move_local_0 += MOVE_ROUND_STRIDE) {
        const i32 move_local_1 = move_local_0 + WARP_SIZE;
        const bool has_move_1 = move_local_1 < input.move_count;
        const usize output_idx_0 = input.move_offset + static_cast<usize>(move_local_0);
        const usize output_idx_1 = input.move_offset + static_cast<usize>(move_local_1);
        const u16 move_idx_0 = move_indices[output_idx_0];
        const u16 move_idx_1 = has_move_1 ? move_indices[output_idx_1] : 0;
        i32 lane_sum_0 = static_cast<i32>(l1_biases[move_idx_0]) * Q * Q;
        i32 lane_sum_1 = has_move_1 ? static_cast<i32>(l1_biases[move_idx_1]) * Q * Q : 0;

        for (usize chunk_start = 0; chunk_start < ACTIVATED_SIZE; chunk_start += L1_CHUNK_SIZE) {
            for (usize i = lane; i < L1_CHUNK_SIZE; i += WARP_SIZE) {
                l1_left[i] = ft_biases[chunk_start + i];
                l1_right[i] = ft_biases[ACTIVATED_SIZE + chunk_start + i];
            }

            __syncwarp();

            for (usize i = lane; i < L1_CHUNK_SIZE; i += WARP_SIZE) {
                i16 left = l1_left[i];
                i16 right = l1_right[i];
                u64 occupied = shared_occupied[warp_in_block];
                while (occupied != 0) {
                    const i32 sq = pop_lsb(occupied);
                    const u8 feature_class = feature_classes[sq];
                    const usize base = ft_offset<L1_SIZE>(feature_class, sq ^ flip);
                    left += ft_weights[base + chunk_start + i];
                    right += ft_weights[base + ACTIVATED_SIZE + chunk_start + i];
                }
                l1_left[i] = left;
                l1_right[i] = right;
            }

            __syncwarp();

            for (usize i = lane; i < L1_CHUNK_SIZE; i += WARP_SIZE) {
                const i16 left = l1_left[i] < 0 ? 0 : (l1_left[i] > Q ? Q : l1_left[i]);
                const i16 right = l1_right[i] < 0 ? 0 : (l1_right[i] > Q ? Q : l1_right[i]);
                activated_chunk[i] = static_cast<u16>(left * right);
            }

            __syncwarp();

            const i8 *weights_0 = l1_weights + static_cast<usize>(move_idx_0) * ACTIVATED_SIZE + chunk_start;
#pragma unroll
            for (usize i = 0; i < L1_CHUNK_SIZE; ++i) {
                lane_sum_0 += static_cast<i32>(activated_chunk[i]) * static_cast<i32>(weights_0[i]);
            }

            if (has_move_1) {
                const i8 *weights_1 = l1_weights + static_cast<usize>(move_idx_1) * ACTIVATED_SIZE + chunk_start;
#pragma unroll
                for (usize i = 0; i < L1_CHUNK_SIZE; ++i) {
                    lane_sum_1 += static_cast<i32>(activated_chunk[i]) * static_cast<i32>(weights_1[i]);
                }
            }
        }

        outputs[output_idx_0] = static_cast<f32>(lane_sum_0) * DEQUANTISATION;
        if (has_move_1) {
            outputs[output_idx_1] = static_cast<f32>(lane_sum_1) * DEQUANTISATION;
        }
    }
}

} // namespace

bool cuda_available() {
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

void evaluate_many(const CudaPolicyInput *inputs, const u16 *move_indices, f32 *outputs, usize position_count) {
    if (position_count == 0) {
        return;
    }

    static DeviceCached<cuda_detail::CudaPolicyNetwork> cached_network;

    usize total_move_count = 0;
    for (usize i = 0; i < position_count; ++i) {
        const usize move_end = static_cast<usize>(inputs[i].move_offset) + inputs[i].move_count;
        total_move_count = total_move_count > move_end ? total_move_count : move_end;
    }
    if (total_move_count == 0) {
        return;
    }

    const i32 position_count_i32 = static_cast<i32>(position_count);
    CudaArray<CudaPolicyInput> device_inputs(position_count);
    CudaArray<u16> device_move_indices(total_move_count);
    CudaArray<f32> device_outputs(total_move_count);

    device_inputs.set(inputs);
    device_move_indices.set(move_indices);
    auto *device_network =
        cached_network.get_or_init([](auto &host_network) { cuda_detail::export_cuda_network(host_network); });

    const i32 block_count = (position_count_i32 + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    evaluate_kernel<<<block_count, THREADS_PER_BLOCK>>>(device_inputs, device_move_indices, device_outputs,
                                                        position_count_i32, device_network);
    cuda_check(cudaGetLastError());
    cuda_check(cudaDeviceSynchronize());

    device_outputs.get(outputs);
}

} // namespace network::policy
