#include "../board_cuda.cuh"
#include "gpu.cuh"

#include "../../chess/board_state.hpp"

namespace network::policy {

namespace {

constexpr usize ACTIVATED_SIZE = L1_SIZE / 2;
constexpr i32 WARPS_PER_BLOCK = 4;
constexpr i32 THREADS_PER_BLOCK = network::cuda_common::WARP_SIZE * WARPS_PER_BLOCK;
constexpr i32 MOVES_PER_ROUND = 2;
constexpr usize L1_CHUNK_SIZE = 64;
static_assert(ACTIVATED_SIZE % L1_CHUNK_SIZE == 0);
using namespace network::cuda_common;

[[nodiscard]] __device__ u64 threats_by_warp(const PackedBoard &board, const CompressedMailbox &pieces,
                                             PackedColor color, u8 lane) {
    const i32 king_sq = lsb(piece_bb(board, PackedPieceType::KING, color));
    const u64 occ = occupancy(board) ^ piece_bb(board, PackedPieceType::KING, opposite(color));

    u64 threats = lane == 0 ? king_attacks(king_sq) : 0;
    warp_for_each_bit(board.side_bbs[to_index(color)], lane, [&](int sq, bool active) {
        if (active) {
            const u8 piece_byte = pieces.at(static_cast<u8>(sq));
            switch (decode_piece_type(piece_byte)) {
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
    });

    return threats;
}

__global__ void evaluate_kernel(const CudaPolicyInput *inputs, const u16 *move_indices, f32 *outputs,
                                i32 position_count, const CudaPolicyNetwork *network) {
    const i32 warp_in_block = threadIdx.x / WARP_SIZE;
    const u8 lane = threadIdx.x % WARP_SIZE;
    const i32 position_idx = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
    if (position_idx >= position_count) {
        return;
    }

    __shared__ i16 shared_l1_left[WARPS_PER_BLOCK][L1_CHUNK_SIZE];
    __shared__ i16 shared_l1_right[WARPS_PER_BLOCK][L1_CHUNK_SIZE];
    __shared__ u16 shared_activated_chunk[WARPS_PER_BLOCK][L1_CHUNK_SIZE];
    __shared__ u16 shared_feature_indices[WARPS_PER_BLOCK][BOARD_SIZE];

    const CudaPolicyInput &input = inputs[position_idx];
    i16 *l1_left = shared_l1_left[warp_in_block];
    i16 *l1_right = shared_l1_right[warp_in_block];
    u16 *activated_chunk = shared_activated_chunk[warp_in_block];
    u16 *feature_indices = shared_feature_indices[warp_in_block];
    const CompressedMailbox &pieces = input.pieces;
    i32 flip = 0;
    int active_feature_count = 0;

    {
        const PackedColor perspective = static_cast<PackedColor>(input.side_to_move);
        const PackedBoard board = build_board_warp(pieces, lane);
        const i32 king_sq = lsb(piece_bb(board, PackedPieceType::KING, perspective));
        flip = (perspective == PackedColor::BLACK ? 0b111000 : 0) ^ (file_of(king_sq) >= 4 ? 0b111 : 0);
        const u64 white_threats = warp_broadcast(warp_or(threats_by_warp(board, pieces, PackedColor::WHITE, lane)));
        const u64 black_threats = warp_broadcast(warp_or(threats_by_warp(board, pieces, PackedColor::BLACK, lane)));

        const u64 occ = occupancy(board);
        active_feature_count = __popcll(occ);

        warp_for_each_bit(occ, lane, [&](int sq, bool active) {
            if (active) {
                const u8 piece_byte = pieces.at(static_cast<u8>(sq));
                const PackedPieceType piece = decode_piece_type(piece_byte);
                const PackedColor color = decode_color(piece_byte);
                const i32 defended =
                    perspective == PackedColor::WHITE ? is_set(white_threats, sq) : is_set(black_threats, sq);
                const i32 threatened =
                    perspective == PackedColor::WHITE ? is_set(black_threats, sq) : is_set(white_threats, sq);
                const i32 opposite_color = color != perspective;
                const u8 feature_class = ft_feature_class(defended, threatened, opposite_color, piece);
                feature_indices[lane] =
                    static_cast<u16>((static_cast<u16>(feature_class) << 8) | static_cast<u16>(sq ^ flip));
            }
        });
    }

    const i8 *ft_weights = reinterpret_cast<const i8 *>(&network->ft_weights);
    const i8 *ft_biases = reinterpret_cast<const i8 *>(&network->ft_biases);
    const i8 *l1_weights = reinterpret_cast<const i8 *>(&network->l1_weights);
    const i8 *l1_biases = reinterpret_cast<const i8 *>(&network->l1_biases);
    __syncwarp();

    constexpr f32 DEQUANTISATION = 1.0f / static_cast<f32>(Q * Q * Q);
    constexpr i32 MOVE_ROUND_STRIDE = MOVES_PER_ROUND * WARP_SIZE;
    for (i32 move_base = 0; move_base < input.move_count; move_base += MOVE_ROUND_STRIDE) {
        const i32 idx0 = move_base + lane;
        const i32 idx1 = idx0 + WARP_SIZE;
        const bool has0 = idx0 < input.move_count;
        const bool has1 = idx1 < input.move_count;
        const usize out0 = input.move_offset + static_cast<usize>(idx0);
        const usize out1 = input.move_offset + static_cast<usize>(idx1);
        const u16 move_idx0 = has0 ? move_indices[out0] : 0;
        const u16 move_idx1 = has1 ? move_indices[out1] : 0;
        i32 sum0 = has0 ? static_cast<i32>(l1_biases[move_idx0]) * Q * Q : 0;
        i32 sum1 = has1 ? static_cast<i32>(l1_biases[move_idx1]) * Q * Q : 0;

        for (usize chunk_start = 0; chunk_start < ACTIVATED_SIZE; chunk_start += L1_CHUNK_SIZE) {
            for (usize i = lane; i < L1_CHUNK_SIZE; i += WARP_SIZE) {
                l1_left[i] = ft_biases[chunk_start + i];
                l1_right[i] = ft_biases[ACTIVATED_SIZE + chunk_start + i];
            }

            __syncwarp();

            for (usize i = lane; i < L1_CHUNK_SIZE; i += WARP_SIZE) {
                i16 left = l1_left[i];
                i16 right = l1_right[i];
                for (int j = 0; j < active_feature_count; ++j) {
                    const u16 packed = feature_indices[j];
                    const u8 feature_class = static_cast<u8>(packed >> 8);
                    const i32 sq_flipped = static_cast<i32>(packed & 0xff);
                    const usize base = ft_offset<L1_SIZE>(feature_class, sq_flipped);
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

            if (has0) {
                const i8 *weights = l1_weights + static_cast<usize>(move_idx0) * ACTIVATED_SIZE + chunk_start;
#pragma unroll
                for (usize i = 0; i < L1_CHUNK_SIZE; ++i) {
                    sum0 += static_cast<i32>(activated_chunk[i]) * static_cast<i32>(weights[i]);
                }
            }

            if (has1) {
                const i8 *weights = l1_weights + static_cast<usize>(move_idx1) * ACTIVATED_SIZE + chunk_start;
#pragma unroll
                for (usize i = 0; i < L1_CHUNK_SIZE; ++i) {
                    sum1 += static_cast<i32>(activated_chunk[i]) * static_cast<i32>(weights[i]);
                }
            }
        }

        if (has0) {
            outputs[out0] = static_cast<f32>(sum0) * DEQUANTISATION;
        }
        if (has1) {
            outputs[out1] = static_cast<f32>(sum1) * DEQUANTISATION;
        }
    }
}

cuda_common::DeviceCached<CudaPolicyNetwork> &cached_policy_network() {
    static cuda_common::DeviceCached<CudaPolicyNetwork> cached_network;
    return cached_network;
}

} // namespace

bool cuda_available() {
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

CudaExecutor::CudaExecutor() {
    cuda_common::cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    device_network_ =
        cached_policy_network().get_or_init([](auto &host_network) { export_cuda_network(host_network); });
}

CudaExecutor::~CudaExecutor() {
    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
    }
}

void CudaExecutor::reserve(usize position_count, usize total_move_count) {
    host_inputs_.reserve(position_count);
    host_move_indices_.reserve(total_move_count);
    host_outputs_.reserve(total_move_count);
    device_inputs_.reserve(position_count);
    device_move_indices_.reserve(total_move_count);
    device_outputs_.reserve(total_move_count);
}

CudaPolicyInput *CudaExecutor::inputs() {
    return host_inputs_.data();
}

u16 *CudaExecutor::move_indices() {
    return host_move_indices_.data();
}

f32 *CudaExecutor::outputs() {
    return host_outputs_.data();
}

void CudaExecutor::launch(usize position_count, usize total_move_count) {
    if (position_count == 0 || total_move_count == 0) {
        return;
    }

    reserve(position_count, total_move_count);

    cuda_common::cuda_check(cudaMemcpyAsync(device_inputs_.data(), host_inputs_.data(),
                                            position_count * sizeof(CudaPolicyInput), cudaMemcpyHostToDevice, stream_));
    cuda_common::cuda_check(cudaMemcpyAsync(device_move_indices_.data(), host_move_indices_.data(),
                                            total_move_count * sizeof(u16), cudaMemcpyHostToDevice, stream_));

    const i32 position_count_i32 = static_cast<i32>(position_count);
    const i32 block_count = (position_count_i32 + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    evaluate_kernel<<<block_count, THREADS_PER_BLOCK, 0, stream_>>>(device_inputs_.data(), device_move_indices_.data(),
                                                                    device_outputs_.data(), position_count_i32,
                                                                    device_network_);
    cuda_common::cuda_check(cudaGetLastError());

    cuda_common::cuda_check(cudaMemcpyAsync(host_outputs_.data(), device_outputs_.data(),
                                            total_move_count * sizeof(f32), cudaMemcpyDeviceToHost, stream_));
}

void CudaExecutor::wait() {
    cuda_common::cuda_check(cudaStreamSynchronize(stream_));
}

void evaluate_many(const CudaPolicyInput *inputs, const u16 *move_indices, f32 *outputs, usize position_count) {
    if (position_count == 0) {
        return;
    }

    usize total_move_count = 0;
    for (usize i = 0; i < position_count; ++i) {
        const usize move_end = static_cast<usize>(inputs[i].move_offset) + inputs[i].move_count;
        total_move_count = total_move_count > move_end ? total_move_count : move_end;
    }
    if (total_move_count == 0) {
        return;
    }

    static thread_local CudaExecutor executor;
    executor.reserve(position_count, total_move_count);
    for (usize i = 0; i < position_count; ++i) {
        executor.inputs()[i] = inputs[i];
    }
    for (usize i = 0; i < total_move_count; ++i) {
        executor.move_indices()[i] = move_indices[i];
    }
    executor.launch(position_count, total_move_count);
    executor.wait();
    for (usize i = 0; i < total_move_count; ++i) {
        outputs[i] = executor.outputs()[i];
    }
}

} // namespace network::policy
