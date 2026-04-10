#ifndef COMPRESSED_MAILBOX_HPP
#define COMPRESSED_MAILBOX_HPP

#include "../chess/board_state.hpp"
#include "../util/types.hpp"

#include <array>

#if defined(__CUDACC__)
#define VINE_CUDA_HOST __host__
#define VINE_CUDA_DEVICE __device__
#else
#define VINE_CUDA_HOST
#define VINE_CUDA_DEVICE
#endif

namespace network::cuda_common {

constexpr usize BOARD_SIZE = 64;

struct CompressedMailbox {
    alignas(u32) u8 data[32];

    VINE_CUDA_HOST VINE_CUDA_DEVICE void compress(const u8 *in) {
        const u64 *src = reinterpret_cast<const u64 *>(in);
        u32 *dst = reinterpret_cast<u32 *>(data);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const u64 v = src[i];
            const u64 lo = v & 0x000F000F000F000F;
            const u64 hi = v & 0x0F000F000F000F00;
            const u64 c = lo | (hi >> 4);
            dst[i] = static_cast<u32>((c & 0xFF) | (((c >> 16) & 0xFF) << 8) | (((c >> 32) & 0xFF) << 16) |
                                      (((c >> 48) & 0xFF) << 24));
        }
    }

    VINE_CUDA_HOST void compress(const std::array<ColoredPiece, BOARD_SIZE> &in) {
        compress(reinterpret_cast<const u8 *>(in.data()));
    }

    VINE_CUDA_HOST VINE_CUDA_DEVICE void decompress(u8 *out) const {
        const u32 *src = reinterpret_cast<const u32 *>(data);
        u64 *dst = reinterpret_cast<u64 *>(out);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const u64 v = src[i];
            const u64 e = (v & 0xFF) | ((v & 0xFF00) << 8) | ((v & 0xFF0000) << 16) | ((v & 0xFF000000) << 24);
            dst[i] = (e & 0x000F000F000F000F) | ((e & 0x00F000F000F000F0) << 4);
        }
    }

    [[nodiscard]] VINE_CUDA_HOST VINE_CUDA_DEVICE u8 at(const u8 sq) const {
        return (data[sq / 2] >> (4 * (sq % 2))) & 0x0f;
    }
};

} // namespace network::cuda_common

#undef VINE_CUDA_HOST
#undef VINE_CUDA_DEVICE

#endif // COMPRESSED_MAILBOX_HPP
