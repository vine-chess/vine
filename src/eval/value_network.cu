#include "../chess/board_state.hpp"

#include "cuda_common.cuh"
#include "value_network.cuh"

namespace network::value {

namespace {

constexpr usize L1_SIZE = 4096;
constexpr usize L2_SIZE = 16;
constexpr usize L3_SIZE = 128;
constexpr i16 QA = 255;
constexpr i16 QB = 64;
constexpr i32 WARPS_PER_BLOCK = 4;
constexpr i32 THREADS_PER_BLOCK = network::cuda_common::WARP_SIZE * WARPS_PER_BLOCK;
constexpr usize L1_CHUNK_SIZE = 64;
static_assert((L1_SIZE / 2) % L1_CHUNK_SIZE == 0);
static_assert(network::cuda_common::WARP_SIZE == L2_SIZE * 2);
using namespace network::cuda_common;

[[nodiscard]] __device__ __forceinline__ i32 clamp_i32(i32 value, i32 min_value, i32 max_value) {
    if (value < min_value) {
        return min_value;
    }
    if (value > max_value) {
        return max_value;
    }
    return value;
}

[[nodiscard]] __device__ __forceinline__ f32 clamp_f32(f32 value, f32 min_value, f32 max_value) {
    if (value < min_value) {
        return min_value;
    }
    if (value > max_value) {
        return max_value;
    }
    return value;
}

[[nodiscard]] __device__ u64 ray_between(i32 from, i32 to) {
    if (from == to) {
        return 0;
    }

    const i32 dr = (to >> 3) - (from >> 3);
    const i32 df = file_of(to) - file_of(from);
    i32 step = 0;
    if (dr == 0) {
        step = df > 0 ? 1 : -1;
    } else if (df == 0) {
        step = dr > 0 ? 8 : -8;
    } else if (dr == df) {
        step = dr > 0 ? 9 : -9;
    } else if (dr == -df) {
        step = dr > 0 ? 7 : -7;
    } else {
        return 0;
    }

    u64 mask = 0;
    for (i32 sq = from + step; sq != to; sq += step) {
        mask |= square_bb(sq);
    }
    return mask;
}

[[nodiscard]] __device__ __forceinline__ u64 xray_attacks(u64 attacks, i32 king_sq, u64 occ, u64 blockers,
                                                          bool diagonal) {
    const u64 first_blockers = attacks & blockers;
    if (first_blockers == 0) {
        return 0;
    }

    const u64 occ_without_blockers = occ ^ first_blockers;
    const u64 attacks_without_blockers =
        diagonal ? bishop_attacks(king_sq, occ_without_blockers) : rook_attacks(king_sq, occ_without_blockers);
    return attacks ^ attacks_without_blockers;
}

[[nodiscard]] __device__ u64 compute_pinned_pieces(const PackedBoard &board, PackedColor color) {
    const i32 king_sq = lsb(piece_bb(board, PackedPieceType::KING, color));
    const u64 occ = occupancy(board);
    const u64 own_occ = board.side_bbs[to_index(color)];
    const u64 enemy_diag_sliders = piece_bb(board, PackedPieceType::BISHOP, opposite(color)) |
                                   piece_bb(board, PackedPieceType::QUEEN, opposite(color));
    const u64 enemy_ortho_sliders = piece_bb(board, PackedPieceType::ROOK, opposite(color)) |
                                    piece_bb(board, PackedPieceType::QUEEN, opposite(color));

    const u64 diag_pinners =
        xray_attacks(bishop_attacks(king_sq, occ), king_sq, occ, own_occ, true) & enemy_diag_sliders;
    const u64 ortho_pinners =
        xray_attacks(rook_attacks(king_sq, occ), king_sq, occ, own_occ, false) & enemy_ortho_sliders;

    u64 pinned = 0;
    u64 pinners = diag_pinners | ortho_pinners;
    while (pinners != 0) {
        const i32 pinner_sq = pop_lsb(pinners);
        pinned |= ray_between(king_sq, pinner_sq) & own_occ;
    }
    return pinned;
}

[[nodiscard]] __device__ u64 pinned_threats_by_warp(const PackedBoard &board, const ColoredPiece *pieces,
                                                    PackedColor color, i32 lane) {
    const i32 king_sq = lsb(piece_bb(board, PackedPieceType::KING, color));
    const u64 occ = occupancy(board);
    const u64 pinned = lane == 0 ? compute_pinned_pieces(board, color) : 0;
    const u64 pinned_mask = warp_broadcast(pinned);

    u64 threats = lane == 0 ? king_attacks(king_sq) : 0;
    for (i32 sq = lane; sq < BOARD_SIZE; sq += WARP_SIZE) {
        if (decode_color(pieces[sq]) != color) {
            continue;
        }

        const PackedPieceType piece = decode_piece_type(pieces[sq]);
        u64 cur = 0;
        switch (piece) {
        case PackedPieceType::NONE:
        case PackedPieceType::KING:
            break;
        case PackedPieceType::PAWN:
            cur = pawn_attacks(sq, color);
            break;
        case PackedPieceType::KNIGHT:
            if ((pinned_mask & square_bb(sq)) == 0) {
                cur = knight_attacks(sq);
            }
            break;
        case PackedPieceType::BISHOP:
            cur = bishop_attacks(sq, occ);
            break;
        case PackedPieceType::ROOK:
            cur = rook_attacks(sq, occ);
            break;
        case PackedPieceType::QUEEN:
            cur = bishop_attacks(sq, occ) | rook_attacks(sq, occ);
            break;
        }

        if (pinned_mask & square_bb(sq)) {
            cur &= ray_between(king_sq, sq);
        }
        threats |= cur;
    }

    return threats;
}

__global__ void evaluate_kernel(const CudaBoardInput *inputs, f32 *outputs, i32 count,
                                const cuda_detail::CudaValueNetwork *network) {
    const i32 warp_in_block = threadIdx.x / WARP_SIZE;
    const i32 lane = threadIdx.x % WARP_SIZE;
    const i32 idx = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
    if (idx >= count) {
        return;
    }

    __shared__ i16 shared_l1_left[WARPS_PER_BLOCK][L1_CHUNK_SIZE];
    __shared__ i16 shared_l1_right[WARPS_PER_BLOCK][L1_CHUNK_SIZE];
    __shared__ u16 shared_l1_activations[WARPS_PER_BLOCK][L1_CHUNK_SIZE];
    __shared__ u8 shared_feature_classes[WARPS_PER_BLOCK][BOARD_SIZE];
    __shared__ i32 shared_l2_int[WARPS_PER_BLOCK][L2_SIZE];
    __shared__ f32 shared_l2[WARPS_PER_BLOCK][L2_SIZE];

    const CudaBoardInput &input = inputs[idx];
    const ColoredPiece *pieces = input.pieces;
    i16 *l1_left = shared_l1_left[warp_in_block];
    i16 *l1_right = shared_l1_right[warp_in_block];
    u16 *l1_activations = shared_l1_activations[warp_in_block];
    u8 *feature_classes = shared_feature_classes[warp_in_block];
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

    for (i32 sq = lane; sq < BOARD_SIZE; sq += WARP_SIZE) {
        const PackedPieceType piece = decode_piece_type(pieces[sq]);
        if (piece == PackedPieceType::NONE) {
            feature_classes[sq] = 0xff;
            continue;
        }

        const PackedColor color = decode_color(pieces[sq]);
        const i32 defended = color == PackedColor::WHITE ? is_set(white_threats, sq) : is_set(black_threats, sq);
        const i32 threatened = color == PackedColor::WHITE ? is_set(black_threats, sq) : is_set(white_threats, sq);
        const i32 opposite_color = color != perspective;
        feature_classes[sq] = ft_feature_class(defended, threatened, opposite_color, piece);
    }

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
            for (i32 sq = 0; sq < BOARD_SIZE; ++sq) {
                const u8 feature_class = feature_classes[sq];
                if (feature_class == 0xff) {
                    continue;
                }
                const usize base = ft_offset<L1_SIZE>(feature_class, sq ^ flip);
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
        value *= clamp_f32(value / 6.0f + 0.5f, 0.0f, 1.0f);
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

        v *= clamp_f32(g / 6.0f + 0.5f, 0.0f, 1.0f);
        final_sum += v * l3_weights[i];
    }

    final_sum = warp_sum(final_sum);

    if (lane == 0) {
        outputs[idx] = final_sum;
    }
}

DeviceCached<cuda_detail::CudaValueNetwork> &cached_value_network() {
    static DeviceCached<cuda_detail::CudaValueNetwork> cached_network;
    return cached_network;
}

} // namespace

bool cuda_available() {
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

class CudaExecutor {
  public:
    CudaExecutor() {
        cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        device_network_ =
            cached_value_network().get_or_init([](auto &host_network) { cuda_detail::export_cuda_network(host_network); });
    }

    ~CudaExecutor() {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }

    void reserve(usize count) {
        host_inputs_.reserve(count);
        host_outputs_.reserve(count);
        device_inputs_.reserve(count);
        device_outputs_.reserve(count);
    }

    [[nodiscard]] CudaBoardInput *inputs() {
        return host_inputs_.data();
    }

    [[nodiscard]] f32 *outputs() {
        return host_outputs_.data();
    }

    void launch(usize count) {
        if (count == 0) {
            return;
        }

        reserve(count);

        const i32 count_i32 = static_cast<i32>(count);
        cuda_check(cudaMemcpyAsync(device_inputs_.data(), host_inputs_.data(), count * sizeof(CudaBoardInput),
                                   cudaMemcpyHostToDevice, stream_));

        const i32 block_count = static_cast<i32>((count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
        evaluate_kernel<<<block_count, THREADS_PER_BLOCK, 0, stream_>>>(
            device_inputs_.data(), device_outputs_.data(), count_i32, device_network_);
        cuda_check(cudaGetLastError());

        cuda_check(cudaMemcpyAsync(host_outputs_.data(), device_outputs_.data(), count * sizeof(f32),
                                   cudaMemcpyDeviceToHost, stream_));
    }

    void wait() {
        cuda_check(cudaStreamSynchronize(stream_));
    }

  private:
    PinnedArray<CudaBoardInput> host_inputs_;
    PinnedArray<f32> host_outputs_;
    DeviceBuffer<CudaBoardInput> device_inputs_;
    DeviceBuffer<f32> device_outputs_;
    cudaStream_t stream_ = nullptr;
    cuda_detail::CudaValueNetwork *device_network_ = nullptr;
};

void evaluate_many(const CudaBoardInput *inputs, f32 *outputs, usize count) {
    if (count == 0) {
        return;
    }

    static thread_local CudaExecutor executor;
    executor.reserve(count);
    for (usize i = 0; i < count; ++i) {
        executor.inputs()[i] = inputs[i];
    }
    executor.launch(count);
    executor.wait();
    for (usize i = 0; i < count; ++i) {
        outputs[i] = executor.outputs()[i];
    }
}

} // namespace network::value
