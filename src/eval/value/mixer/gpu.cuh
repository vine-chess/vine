#ifndef EVAL_VALUE_MIXER_GPU_CUH
#define EVAL_VALUE_MIXER_GPU_CUH

#include "../../board_cuda.cuh"
#include "../shared.hpp"

#ifdef MIXER_VALUE_WMMA
#include <mma.h>
#endif

namespace network::value {

using namespace network::cuda_common;

constexpr usize VALUES_PER_LANE = (MIXER_SIZE + WARP_SIZE - 1) / WARP_SIZE;

#ifdef MIXER_VALUE_WMMA

#ifndef MIXER_WMMA_PRECISION
#define MIXER_WMMA_PRECISION 0
#endif

enum class WmmaPrecision {
    TF32 = 0,
    BF16 = 1,
};

constexpr WmmaPrecision WMMA_PRECISION = static_cast<WmmaPrecision>(MIXER_WMMA_PRECISION);

constexpr i32 MAX_WARPS_PER_BLOCK = 4;
constexpr i32 SHARED_MEM_BYTES = 49152;

constexpr usize WMMA_SHARED_BYTES_PER_WARP = []() {
    usize bytes = MIXER_SIZE * sizeof(f32) + BOARD_SIZE * sizeof(u8);
    if (WMMA_PRECISION == WmmaPrecision::BF16) {
        bytes += (MIXER_TILE_SIZE * 4) * sizeof(f32);
    } else {
        bytes += MIXER_TILE_SIZE * sizeof(f32);
    }
    return bytes;
}();

constexpr i32 WARPS_PER_BLOCK =
    std::min(MAX_WARPS_PER_BLOCK, static_cast<i32>(SHARED_MEM_BYTES / WMMA_SHARED_BYTES_PER_WARP));
static_assert(WARPS_PER_BLOCK >= 1);

namespace wmma = nvcuda::wmma;

using operand_storage_t = std::conditional_t<WMMA_PRECISION == WmmaPrecision::TF32, f32, __nv_bfloat16>;
using operand_fragment_t =
    std::conditional_t<WMMA_PRECISION == WmmaPrecision::TF32, wmma::precision::tf32, __nv_bfloat16>;
using accumulator_t = f32;

constexpr i32 K = WMMA_PRECISION == WmmaPrecision::TF32 ? 8 : 16;

using accumulator_fragment_t = wmma::fragment<wmma::accumulator, 16, 16, K, accumulator_t>;
using matrix_a_fragment_t = wmma::fragment<wmma::matrix_a, 16, 16, K, operand_fragment_t, wmma::col_major>;
using matrix_b_fragment_t = wmma::fragment<wmma::matrix_b, 16, 16, K, operand_fragment_t, wmma::col_major>;

struct WmmaScratch {
    f32 tile_f32[MIXER_TILE_SIZE];
    operand_storage_t tile_operand[MIXER_TILE_SIZE];
    operand_storage_t a_operand[MIXER_TILE_SIZE];
    operand_storage_t b_operand[MIXER_TILE_SIZE];
};

#else

constexpr i32 WARPS_PER_BLOCK = 4;
constexpr usize MIXER_TMP_SIZE = []() {
    usize s1 = MIXER_INNER1 * MIXER_D2;
    usize s2 = MIXER_D1 * MIXER_INNER2;
    return std::max({s1, s2, MIXER_SIZE});
}();

#endif

constexpr i32 THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;

#ifdef MIXER_VALUE_WMMA
static_assert(MIXER_D1 % 16 == 0);
static_assert(MIXER_D2 % 16 == 0);
static_assert(MIXER_INNER1 % 16 == 0);
static_assert(MIXER_INNER2 % 16 == 0);
#endif

[[nodiscard]] __device__ inline f32 *get_shared_x(f32 *base, int warp_idx) {
    return base + warp_idx * MIXER_SIZE;
}

#ifndef MIXER_VALUE_WMMA

__device__ void apply_left_mix(f32 (&x)[VALUES_PER_LANE], f32 *shared_x, f32 *shared_tmp, f32 *shared_mix,
                               const f32 *wl_up, const f32 *wl_down, u8 lane) {
    for (usize local = 0; local < VALUES_PER_LANE; ++local) {
        const usize idx = lane + local * WARP_SIZE;
        if (idx < MIXER_SIZE)
            shared_x[idx] = x[local];
    }
    __syncwarp();

    constexpr usize UP_SIZE = MIXER_INNER1 * MIXER_D2;
    for (usize i = lane; i < UP_SIZE; i += WARP_SIZE) {
        const usize row = i % MIXER_INNER1;
        const usize col = i / MIXER_INNER1;
        f32 sum = 0.0f;
        for (usize k = 0; k < MIXER_D1; ++k) {
            sum += wl_up[matrix_index(row, k, MIXER_INNER1)] * shared_x[matrix_index(k, col, MIXER_D1)];
        }
        shared_tmp[i] = sum;
    }
    __syncwarp();

#if MIXER_DO_DOWN_PROJ
    for (usize i = lane; i < UP_SIZE; i += WARP_SIZE)
        shared_tmp[i] = activate(shared_tmp[i]);
    __syncwarp();

    for (usize i = lane; i < MIXER_SIZE; i += WARP_SIZE) {
        const usize row = i % MIXER_D1;
        const usize col = i / MIXER_D1;
        f32 sum = 0.0f;
        for (usize k = 0; k < MIXER_INNER1; ++k) {
            sum += wl_down[matrix_index(row, k, MIXER_D1)] * shared_tmp[matrix_index(k, col, MIXER_INNER1)];
        }
        shared_mix[i] = sum;
    }
    __syncwarp();
    for (usize local = 0; local < VALUES_PER_LANE; ++local) {
        const usize idx = lane + local * WARP_SIZE;
        if (idx < MIXER_SIZE)
            x[local] += shared_mix[idx];
    }
#else
    for (usize local = 0; local < VALUES_PER_LANE; ++local) {
        const usize idx = lane + local * WARP_SIZE;
        if (idx < MIXER_SIZE)
            x[local] += activate(shared_tmp[idx]);
    }
#endif
}

__device__ void apply_right_mix(f32 (&x)[VALUES_PER_LANE], f32 *shared_x, f32 *shared_tmp, f32 *shared_mix,
                                const f32 *wr_up, const f32 *wr_down, u8 lane) {
    for (usize local = 0; local < VALUES_PER_LANE; ++local) {
        const usize idx = lane + local * WARP_SIZE;
        if (idx < MIXER_SIZE)
            shared_x[idx] = x[local];
    }
    __syncwarp();

    constexpr usize UP_SIZE = MIXER_D1 * MIXER_INNER2;
    for (usize i = lane; i < UP_SIZE; i += WARP_SIZE) {
        const usize row = i % MIXER_D1;
        const usize col = i / MIXER_D1;
        f32 sum = 0.0f;
        for (usize k = 0; k < MIXER_D2; ++k) {
            sum += shared_x[matrix_index(row, k, MIXER_D1)] * wr_up[matrix_index(k, col, MIXER_D2)];
        }
        shared_tmp[i] = sum;
    }
    __syncwarp();

#if MIXER_DO_DOWN_PROJ
    for (usize i = lane; i < UP_SIZE; i += WARP_SIZE)
        shared_tmp[i] = activate(shared_tmp[i]);
    __syncwarp();

    for (usize i = lane; i < MIXER_SIZE; i += WARP_SIZE) {
        const usize row = i % MIXER_D1;
        const usize col = i / MIXER_D1;
        f32 sum = 0.0f;
        for (usize k = 0; k < MIXER_INNER2; ++k) {
            sum += shared_tmp[matrix_index(row, k, MIXER_D1)] * wr_down[matrix_index(k, col, MIXER_INNER2)];
        }
        shared_mix[i] = sum;
    }
    __syncwarp();
    for (usize local = 0; local < VALUES_PER_LANE; ++local) {
        const usize idx = lane + local * WARP_SIZE;
        if (idx < MIXER_SIZE)
            x[local] += shared_mix[idx];
    }
#else
    for (usize local = 0; local < VALUES_PER_LANE; ++local) {
        const usize idx = lane + local * WARP_SIZE;
        if (idx < MIXER_SIZE)
            x[local] += activate(shared_tmp[idx]);
    }
#endif
}

#endif // !MIXER_VALUE_WMMA

#ifdef MIXER_VALUE_WMMA

[[nodiscard]] __device__ inline operand_storage_t to_wmma_operand(f32 x) {
    if constexpr (WMMA_PRECISION == WmmaPrecision::TF32) {
        return nvcuda::wmma::__float_to_tf32(x);
    } else {
        return __float2bfloat16(x);
    }
}

__device__ inline void store_activate_tile(WmmaScratch &scratch, accumulator_fragment_t &fragment, u8 lane) {
    namespace wmma = nvcuda::wmma;

    if constexpr (WMMA_PRECISION == WmmaPrecision::TF32) {
        for (usize i = 0; i < fragment.num_storage_elements; ++i) {
            fragment.x[i] = to_wmma_operand(activate(fragment.x[i]));
        }
        wmma::store_matrix_sync(scratch.tile_f32, fragment, 16, wmma::mem_col_major);
    } else {
        wmma::store_matrix_sync(scratch.tile_f32, fragment, 16, wmma::mem_col_major);
        __syncwarp();
        for (usize i = lane; i < 16 * 16; i += WARP_SIZE) {
            scratch.tile_operand[i] = to_wmma_operand(activate(scratch.tile_f32[i]));
        }
    }
}

__device__ inline matrix_a_fragment_t load_a_from_f32(WmmaScratch &scratch, const f32 *src, usize ld, usize row0,
                                                      usize col0, u8 lane) {
    namespace wmma = nvcuda::wmma;
    matrix_a_fragment_t a;
#if MIXER_WMMA_PRECISION == 0
    wmma::load_matrix_sync(a, src + matrix_index(row0, col0, ld), ld);
#else
    for (usize i = lane; i < 16 * 16; i += WARP_SIZE) {
        scratch.a_operand[i] = to_wmma_operand(src[matrix_index(row0 + i % 16, col0 + i / 16, ld)]);
    }
    __syncwarp();
    wmma::load_matrix_sync(a, scratch.a_operand, 16);
#endif
    return a;
}

__device__ inline matrix_b_fragment_t load_b_from_f32(WmmaScratch &scratch, const f32 *src, usize ld, usize row0,
                                                      usize col0, u8 lane) {
    namespace wmma = nvcuda::wmma;
    matrix_b_fragment_t b;
#if MIXER_WMMA_PRECISION == 0
    wmma::load_matrix_sync(b, src + matrix_index(row0, col0, ld), ld);
#else
    for (usize i = lane; i < 16 * 16; i += WARP_SIZE) {
        scratch.b_operand[i] = to_wmma_operand(src[matrix_index(row0 + i % 16, col0 + i / 16, ld)]);
    }
    __syncwarp();
    wmma::load_matrix_sync(b, scratch.b_operand, 16);
#endif
    return b;
}

__device__ inline matrix_a_fragment_t load_tile_a(WmmaScratch &scratch, usize col0) {
    namespace wmma = nvcuda::wmma;
    matrix_a_fragment_t a;
#if MIXER_WMMA_PRECISION == 0
    wmma::load_matrix_sync(a, scratch.tile_f32 + col0 * 16, 16);
#else
    wmma::load_matrix_sync(a, scratch.tile_operand, 16);
#endif
    return a;
}

__device__ inline matrix_b_fragment_t load_tile_b(WmmaScratch &scratch, usize col0) {
    namespace wmma = nvcuda::wmma;
    matrix_b_fragment_t b;
#if MIXER_WMMA_PRECISION == 0
    wmma::load_matrix_sync(b, scratch.tile_f32 + col0, 16);
#else
    wmma::load_matrix_sync(b, scratch.tile_operand, 16);
#endif
    return b;
}

__device__ inline void scatter_tile(f32 (&x)[VALUES_PER_LANE], const f32 *tile, usize tile_row, usize tile_col,
                                    u8 lane) {
    for (usize l = 0; l < VALUES_PER_LANE; ++l) {
        const usize flat = lane + l * WARP_SIZE;
        if (flat < MIXER_SIZE) {
            const usize row = flat % MIXER_D1;
            const usize col = flat / MIXER_D1;
            if (row - tile_row < 16u && col - tile_col < 16u) {
                x[l] += tile[(col - tile_col) * 16 + (row - tile_row)];
            }
        }
    }
}

__device__ void apply_left_mix_wmma(f32 (&x)[VALUES_PER_LANE], f32 *shared_x, WmmaScratch &scratch, const f32 *wl_up,
                                    const f32 *wl_down, u8 lane) {
    namespace wmma = nvcuda::wmma;

    for (usize local = 0; local < VALUES_PER_LANE; ++local) {
        const usize idx = lane + local * WARP_SIZE;
        if (idx < MIXER_SIZE) {
            if constexpr (WMMA_PRECISION == WmmaPrecision::TF32) {
                shared_x[idx] = to_wmma_operand(x[local]);
            } else {
                shared_x[idx] = x[local];
            }
        }
    }
    __syncwarp();

    for (usize tile_col = 0; tile_col < MIXER_D2; tile_col += 16) {
        for (usize tile_row = 0; tile_row < MIXER_D1; tile_row += 16) {
            accumulator_fragment_t c;
            wmma::fill_fragment(c, 0.0f);

#if MIXER_DO_DOWN_PROJ
            for (usize k_chunk = 0; k_chunk < MIXER_INNER1; k_chunk += 16) {
                accumulator_fragment_t c_up;
                wmma::fill_fragment(c_up, 0.0f);
                for (usize k = 0; k < MIXER_D1; k += K) {
                    auto a = load_a_from_f32(scratch, wl_up, MIXER_INNER1, k_chunk, k, lane);
                    auto b = load_b_from_f32(scratch, shared_x, MIXER_D1, k, tile_col, lane);
                    wmma::mma_sync(c_up, a, b, c_up);
                    __syncwarp();
                }
                store_activate_tile(scratch, c_up, lane);
                __syncwarp();
                for (usize k2 = 0; k2 < 16; k2 += K) {
                    auto a = load_a_from_f32(scratch, wl_down, MIXER_D1, tile_row, k_chunk + k2, lane);
                    auto b = load_tile_b(scratch, k2);
                    wmma::mma_sync(c, a, b, c);
                    __syncwarp();
                }
            }
#else
            for (usize k = 0; k < MIXER_D1; k += K) {
                auto a = load_a_from_f32(scratch, wl_up, MIXER_INNER1, tile_row, k, lane);
                auto b = load_b_from_f32(scratch, shared_x, MIXER_D1, k, tile_col, lane);
                wmma::mma_sync(c, a, b, c);
                __syncwarp();
            }
            for (usize i = 0; i < c.num_storage_elements; ++i)
                c.x[i] = activate(c.x[i]);
#endif
            wmma::store_matrix_sync(scratch.tile_f32, c, 16, wmma::mem_col_major);
            __syncwarp();
            scatter_tile(x, scratch.tile_f32, tile_row, tile_col, lane);
        }
    }
}

__device__ void apply_right_mix_wmma(f32 (&x)[VALUES_PER_LANE], f32 *shared_x, WmmaScratch &scratch, const f32 *wr_up,
                                     const f32 *wr_down, u8 lane) {
    namespace wmma = nvcuda::wmma;

    for (usize local = 0; local < VALUES_PER_LANE; ++local) {
        const usize idx = lane + local * WARP_SIZE;
        if (idx < MIXER_SIZE) {
            if constexpr (WMMA_PRECISION == WmmaPrecision::TF32) {
                shared_x[idx] = to_wmma_operand(x[local]);
            } else {
                shared_x[idx] = x[local];
            }
        }
    }
    __syncwarp();

    for (usize tile_col = 0; tile_col < MIXER_D2; tile_col += 16) {
        for (usize tile_row = 0; tile_row < MIXER_D1; tile_row += 16) {
            accumulator_fragment_t c;
            wmma::fill_fragment(c, 0.0f);

#if MIXER_DO_DOWN_PROJ
            for (usize k_chunk = 0; k_chunk < MIXER_INNER2; k_chunk += 16) {
                accumulator_fragment_t c_up;
                wmma::fill_fragment(c_up, 0.0f);
                for (usize k = 0; k < MIXER_D2; k += K) {
                    auto a = load_a_from_f32(scratch, shared_x, MIXER_D1, tile_row, k, lane);
                    auto b = load_b_from_f32(scratch, wr_up, MIXER_D2, k, k_chunk, lane);
                    wmma::mma_sync(c_up, a, b, c_up);
                    __syncwarp();
                }
                store_activate_tile(scratch, c_up, lane);
                __syncwarp();
                for (usize k2 = 0; k2 < 16; k2 += K) {
                    auto a = load_tile_a(scratch, k2);
                    auto b = load_b_from_f32(scratch, wr_down, MIXER_INNER2, k_chunk + k2, tile_col, lane);
                    wmma::mma_sync(c, a, b, c);
                    __syncwarp();
                }
            }
#else
            for (usize k = 0; k < MIXER_D2; k += K) {
                auto a = load_a_from_f32(scratch, shared_x, MIXER_D1, tile_row, k, lane);
                auto b = load_b_from_f32(scratch, wr_up, MIXER_D2, k, tile_col, lane);
                wmma::mma_sync(c, a, b, c);
                __syncwarp();
            }
            for (usize i = 0; i < c.num_storage_elements; ++i)
                c.x[i] = activate(c.x[i]);
#endif
            wmma::store_matrix_sync(scratch.tile_f32, c, 16, wmma::mem_col_major);
            __syncwarp();
            scatter_tile(x, scratch.tile_f32, tile_row, tile_col, lane);
        }
    }
}

#endif // MIXER_VALUE_WMMA

__global__ void evaluate_kernel(const CudaBoardInput *inputs, f32 *outputs, i32 count,
                                const CudaValueNetwork *network) {
    const i32 warp_in_block = threadIdx.x / WARP_SIZE;
    const u8 lane = threadIdx.x % WARP_SIZE;
    const i32 idx = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
    if (idx >= count) {
        return;
    }

    __shared__ u16 shared_feature_indices[WARPS_PER_BLOCK][BOARD_SIZE];
    __shared__ f32 shared_x[WARPS_PER_BLOCK][MIXER_SIZE];
#ifdef MIXER_VALUE_WMMA
    __shared__ WmmaScratch shared_wmma[WARPS_PER_BLOCK];
    WmmaScratch &wmma_scratch = shared_wmma[warp_in_block];
#else
    __shared__ f32 shared_tmp[WARPS_PER_BLOCK][MIXER_TMP_SIZE];
    __shared__ f32 shared_mix[WARPS_PER_BLOCK][MIXER_SIZE];
    f32 *mixer_tmp = shared_tmp[warp_in_block];
    f32 *mixer_mix = shared_mix[warp_in_block];
#endif

    const CudaBoardInput &input = inputs[idx];
    u16 *feature_indices = shared_feature_indices[warp_in_block];
    f32 *mixer_x = shared_x[warp_in_block];
    const CompressedMailbox &pieces = input.pieces;
    const i16 *ft_weights = reinterpret_cast<const i16 *>(&network->ft_weights);
    const i16 *ft_biases = reinterpret_cast<const i16 *>(&network->ft_biases);
    const f32 *value_weights = reinterpret_cast<const f32 *>(&network->value_weights);

    PackedBoard board = build_board_warp(pieces, lane);
    const PackedColor perspective = static_cast<PackedColor>(input.side_to_move);
    const i32 king_sq = lsb(piece_bb(board, PackedPieceType::KING, perspective));
    const usize flip = feature_flip(perspective == PackedColor::BLACK, file_of(king_sq) >= 4);
    const u64 white_threats = warp_broadcast(warp_or(pinned_threats_by_warp(board, pieces, PackedColor::WHITE, lane)));
    const u64 black_threats = warp_broadcast(warp_or(pinned_threats_by_warp(board, pieces, PackedColor::BLACK, lane)));
    __syncwarp();

    i32 acc[VALUES_PER_LANE]{};
    f32 x[VALUES_PER_LANE]{};

#pragma unroll
    for (usize local = 0; local < VALUES_PER_LANE; ++local) {
        const usize out_idx = lane + local * WARP_SIZE;
        if (out_idx < MIXER_SIZE) {
            acc[local] = ft_biases[out_idx];
        }
    }

    warp_for_each_bit(occupancy(board), lane, [&](int sq, bool active) {
        if (active) {
            const u8 piece_byte = pieces.at(static_cast<u8>(sq));
            const PackedPieceType piece = decode_piece_type(piece_byte);
            const PackedColor color = decode_color(piece_byte);
            const i32 defended = color == PackedColor::WHITE ? is_set(white_threats, sq) : is_set(black_threats, sq);
            const i32 threatened = color == PackedColor::WHITE ? is_set(black_threats, sq) : is_set(white_threats, sq);
            const i32 opposite_color = color != perspective;
            const u8 fc = feature_class(defended, threatened, opposite_color, to_index(piece) - 1);
            feature_indices[lane] = static_cast<u16>(feature_index(fc, sq, flip));
        }
    });
    const int active_feature_count = __popcll(occupancy(board));

    __syncwarp();

    for (int i = 0; i < active_feature_count; ++i) {
        const usize feature_idx = feature_indices[i];
        const usize base = feature_idx * MIXER_SIZE;
#pragma unroll
        for (usize local = 0; local < VALUES_PER_LANE; ++local) {
            const usize out_idx = lane + local * WARP_SIZE;
            if (out_idx < MIXER_SIZE) {
                acc[local] += ft_weights[base + out_idx];
            }
        }
    }

#pragma unroll
    for (usize local = 0; local < VALUES_PER_LANE; ++local) {
        const usize out_idx = lane + local * WARP_SIZE;
        if (out_idx < MIXER_SIZE) {
            x[local] = static_cast<f32>(clamp_i32(acc[local], 0, QA)) / static_cast<f32>(QA);
        }
    }

    __syncwarp();

    for (usize layer_i = 0; layer_i < MIXER_NUM_LAYERS; ++layer_i) {
        const auto &layer = network->layers[layer_i];

        const f32 *wl_up = layer.wl_up;
        const f32 *wr_up = layer.wr_up;
#if MIXER_DO_DOWN_PROJ
        const f32 *wl_down = layer.wl_down;
        const f32 *wr_down = layer.wr_down;
#else
        const f32 *wl_down = nullptr;
        const f32 *wr_down = nullptr;
#endif

#ifdef MIXER_VALUE_WMMA
        apply_left_mix_wmma(x, mixer_x, wmma_scratch, wl_up, wl_down, lane);
        apply_right_mix_wmma(x, mixer_x, wmma_scratch, wr_up, wr_down, lane);
#else
        apply_left_mix(x, mixer_x, mixer_tmp, mixer_mix, wl_up, wl_down, lane);
        apply_right_mix(x, mixer_x, mixer_tmp, mixer_mix, wr_up, wr_down, lane);
#endif
    }

    f32 final_sum = lane == 0 ? network->value_bias : 0.0f;
#pragma unroll
    for (usize local = 0; local < VALUES_PER_LANE; ++local) {
        const usize out_idx = lane + local * WARP_SIZE;
        if (out_idx < MIXER_SIZE) {
            final_sum += x[local] * value_weights[out_idx];
        }
    }

    final_sum = warp_sum(final_sum);
    if (lane == 0) {
        outputs[idx] = final_sum;
    }
}

} // namespace network::value

#endif // EVAL_VALUE_MIXER_GPU_CUH
