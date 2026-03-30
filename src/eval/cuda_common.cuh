#ifndef CUDA_COMMON_CUH
#define CUDA_COMMON_CUH

#include "../chess/board_state.hpp"
#include "../util/types.hpp"

#include <cstdlib>
#include <cstddef>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <source_location>
#include <type_traits>

namespace network::cuda_common {

constexpr usize BOARD_SIZE = 64;
constexpr i32 WARP_SIZE = 32;

enum class PackedColor : u8 {
    WHITE = 0,
    BLACK = 1,
};

enum class PackedPieceType : u8 {
    NONE = 0,
    PAWN = 1,
    KNIGHT = 2,
    BISHOP = 3,
    ROOK = 4,
    QUEEN = 5,
    KING = 6,
};

inline void cuda_check(cudaError_t err, const std::source_location &loc = std::source_location::current()) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << "\n  File: " << loc.file_name()
                  << "\n  Line: " << loc.line() << "\n  Func: " << loc.function_name() << "\n";
        std::exit(1);
    }
}

template <class T>
struct CudaArray {
    T *ptr;
    std::size_t size;

    explicit CudaArray(std::size_t n) : ptr(nullptr), size(n) {
        cuda_check(cudaMalloc(reinterpret_cast<void **>(&ptr), size_bytes()));
    }

    ~CudaArray() {
        cudaFree(ptr);
    }

    [[nodiscard]] std::size_t size_bytes() const {
        return size * sizeof(T);
    }

    void set(const T *host_ptr) {
        cuda_check(cudaMemcpy(ptr, host_ptr, size_bytes(), cudaMemcpyHostToDevice));
    }

    void get(T *host_ptr) const {
        cuda_check(cudaMemcpy(host_ptr, ptr, size_bytes(), cudaMemcpyDeviceToHost));
    }

    operator T *() const {
        return ptr;
    }
};

template <class T>
struct PinnedArray {
    T *ptr = nullptr;
    std::size_t size = 0;

    PinnedArray() = default;

    PinnedArray(const PinnedArray &) = delete;
    PinnedArray &operator=(const PinnedArray &) = delete;

    ~PinnedArray() {
        reset();
    }

    void reserve(std::size_t n) {
        if (size >= n) {
            return;
        }

        reset();
        cuda_check(cudaMallocHost(reinterpret_cast<void **>(&ptr), n * sizeof(T)));
        size = n;
    }

    void reset() {
        if (ptr != nullptr) {
            cudaFreeHost(ptr);
            ptr = nullptr;
            size = 0;
        }
    }

    [[nodiscard]] T *data() {
        return ptr;
    }
};

template <class T>
struct DeviceBuffer {
    T *ptr = nullptr;
    std::size_t size = 0;

    DeviceBuffer() = default;

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    ~DeviceBuffer() {
        reset();
    }

    void reserve(std::size_t n) {
        if (size >= n) {
            return;
        }

        reset();
        cuda_check(cudaMalloc(reinterpret_cast<void **>(&ptr), n * sizeof(T)));
        size = n;
    }

    void reset() {
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
            size = 0;
        }
    }

    [[nodiscard]] T *data() {
        return ptr;
    }
};

template <class T>
class DeviceCached {
  public:
    static_assert(std::is_trivially_copyable_v<T>);

    DeviceCached() = default;
    DeviceCached(const DeviceCached &) = delete;
    DeviceCached &operator=(const DeviceCached &) = delete;

    template <class Init>
    T *get_or_init(Init init) {
        std::call_once(init_once_, [this, &init] {
            auto host_value = std::make_unique<T>();
            init(*host_value);
            cuda_check(cudaMalloc(reinterpret_cast<void **>(&device_ptr_), sizeof(T)));
            cuda_check(cudaMemcpy(device_ptr_, host_value.get(), sizeof(T), cudaMemcpyHostToDevice));
            initialized_ = true;
        });
        return device_ptr_;
    }

    ~DeviceCached() {
        if (initialized_) {
            cudaFree(device_ptr_);
        }
    }

  private:
    std::once_flag init_once_{};
    T *device_ptr_ = nullptr;
    bool initialized_ = false;
};

struct PackedBoard {
    u64 piece_bbs[6];
    u64 side_bbs[2];
};

[[nodiscard]] __device__ __host__ constexpr usize to_index(PackedColor color) {
    return static_cast<usize>(color);
}

[[nodiscard]] __device__ __host__ constexpr usize to_index(PackedPieceType piece_type) {
    return static_cast<usize>(piece_type);
}

[[nodiscard]] __device__ __host__ constexpr PackedColor opposite(PackedColor color) {
    return color == PackedColor::WHITE ? PackedColor::BLACK : PackedColor::WHITE;
}

[[nodiscard]] __device__ __forceinline__ PackedPieceType decode_piece_type(ColoredPiece piece) {
    return static_cast<PackedPieceType>((*reinterpret_cast<const u8 *>(&piece)) >> 1);
}

[[nodiscard]] __device__ __forceinline__ PackedColor decode_color(ColoredPiece piece) {
    return static_cast<PackedColor>((*reinterpret_cast<const u8 *>(&piece)) & 1);
}

[[nodiscard]] __device__ __host__ constexpr u64 square_bb(i32 sq) {
    return u64{1} << sq;
}

[[nodiscard]] __device__ __forceinline__ u64 reverse_bits(u64 x) {
    return __brevll(x);
}

[[nodiscard]] __device__ __forceinline__ i32 file_of(i32 sq) {
    return sq & 7;
}

[[nodiscard]] __device__ __forceinline__ i32 lsb(u64 bb) {
    return __ffsll(static_cast<long long>(bb)) - 1;
}

[[nodiscard]] __device__ __forceinline__ i32 pop_lsb(u64 &bb) {
    const i32 sq = lsb(bb);
    bb &= bb - 1;
    return sq;
}

[[nodiscard]] __device__ __forceinline__ bool is_set(u64 bb, i32 sq) {
    return ((bb >> sq) & 1) != 0;
}

[[nodiscard]] __device__ __host__ constexpr u64 file_mask(i32 file) {
    return 0x0101010101010101ULL << file;
}

[[nodiscard]] __device__ __host__ constexpr u64 rank_mask(i32 rank) {
    return 0xffULL << (8 * rank);
}

[[nodiscard]] __device__ __host__ constexpr u64 diagonal_mask(i32 sq) {
    const i32 rank = sq / 8;
    const i32 file = sq % 8;
    u64 mask = 0;
    for (i32 r = rank + 1, f = file + 1; r < 8 && f < 8; ++r, ++f) {
        mask |= square_bb(r * 8 + f);
    }
    for (i32 r = rank - 1, f = file - 1; r >= 0 && f >= 0; --r, --f) {
        mask |= square_bb(r * 8 + f);
    }
    return mask;
}

[[nodiscard]] __device__ __host__ constexpr u64 anti_diagonal_mask(i32 sq) {
    const i32 rank = sq / 8;
    const i32 file = sq % 8;
    u64 mask = 0;
    for (i32 r = rank + 1, f = file - 1; r < 8 && f >= 0; ++r, --f) {
        mask |= square_bb(r * 8 + f);
    }
    for (i32 r = rank - 1, f = file + 1; r >= 0 && f < 8; --r, ++f) {
        mask |= square_bb(r * 8 + f);
    }
    return mask;
}

[[nodiscard]] __device__ __host__ constexpr u64 rook_rank_mask(i32 sq) {
    return rank_mask(sq >> 3) ^ square_bb(sq);
}

[[nodiscard]] __device__ __host__ constexpr u64 rook_file_mask(i32 sq) {
    return file_mask(sq & 7) ^ square_bb(sq);
}

[[nodiscard]] __device__ __host__ constexpr u64 bishop_diag_mask(i32 sq) {
    return diagonal_mask(sq);
}

[[nodiscard]] __device__ __host__ constexpr u64 bishop_anti_diag_mask(i32 sq) {
    return anti_diagonal_mask(sq);
}

[[nodiscard]] __device__ __forceinline__ u64 piece_bb(const PackedBoard &board, PackedPieceType piece_type,
                                                      PackedColor color) {
    return board.piece_bbs[to_index(piece_type) - 1] & board.side_bbs[to_index(color)];
}

[[nodiscard]] __device__ __forceinline__ u64 occupancy(const PackedBoard &board) {
    return board.side_bbs[to_index(PackedColor::WHITE)] | board.side_bbs[to_index(PackedColor::BLACK)];
}

[[nodiscard]] __device__ __forceinline__ u64 line_attacks(u64 occ, u64 mask, u64 from_bb, u64 from_bb_reversed) {
    const u64 occ_masked = occ & mask;
    const u64 forward = occ_masked - 2 * from_bb;
    const u64 backward = reverse_bits(reverse_bits(occ_masked) - 2 * from_bb_reversed);
    return (forward ^ backward) & mask;
}

[[nodiscard]] __device__ __forceinline__ u64 warp_or(u64 value) {
    for (i32 offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        value |= __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

template <class T>
[[nodiscard]] __device__ __forceinline__ T warp_broadcast(T value) {
    return __shfl_sync(0xffffffff, value, 0);
}

template <class T>
[[nodiscard]] __device__ __forceinline__ T warp_sum(T value) {
    for (i32 offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

[[nodiscard]] __device__ __forceinline__ u64 rook_attacks(i32 sq, u64 occ) {
    const u64 from_bb = square_bb(sq);
    const u64 from_bb_reversed = reverse_bits(from_bb);
    return line_attacks(occ, rook_rank_mask(sq), from_bb, from_bb_reversed) |
           line_attacks(occ, rook_file_mask(sq), from_bb, from_bb_reversed);
}

[[nodiscard]] __device__ __forceinline__ u64 bishop_attacks(i32 sq, u64 occ) {
    const u64 from_bb = square_bb(sq);
    const u64 from_bb_reversed = reverse_bits(from_bb);
    return line_attacks(occ, bishop_diag_mask(sq), from_bb, from_bb_reversed) |
           line_attacks(occ, bishop_anti_diag_mask(sq), from_bb, from_bb_reversed);
}

[[nodiscard]] __device__ __forceinline__ u64 king_attacks(i32 sq) {
    constexpr u64 NOT_A_FILE = 0xfefefefefefefefeULL;
    constexpr u64 NOT_H_FILE = 0x7f7f7f7f7f7f7f7fULL;

    const u64 king = square_bb(sq);
    u64 attacks = king;
    attacks |= attacks << 8;
    attacks |= attacks >> 8;
    attacks |= (attacks & NOT_A_FILE) >> 1;
    attacks |= (attacks & NOT_H_FILE) << 1;
    return attacks ^ king;
}

[[nodiscard]] __device__ __forceinline__ u64 knight_attacks(i32 sq) {
    constexpr u64 NOT_A_FILE = 0xfefefefefefefefeULL;
    constexpr u64 NOT_H_FILE = 0x7f7f7f7f7f7f7f7fULL;
    constexpr u64 NOT_AB_FILES = 0xfcfcfcfcfcfcfcfcULL;
    constexpr u64 NOT_GH_FILES = 0x3f3f3f3f3f3f3f3fULL;

    const u64 knight = square_bb(sq);
    u64 one_file = ((knight & NOT_A_FILE) >> 1) | ((knight & NOT_H_FILE) << 1);
    u64 two_files = ((knight & NOT_AB_FILES) >> 2) | ((knight & NOT_GH_FILES) << 2);
    return (one_file << 16) | (one_file >> 16) | (two_files << 8) | (two_files >> 8);
}

[[nodiscard]] __device__ __forceinline__ u64 pawn_attacks(i32 sq, PackedColor color) {
    constexpr u64 NOT_A_FILE = 0xfefefefefefefefeULL;
    constexpr u64 NOT_H_FILE = 0x7f7f7f7f7f7f7f7fULL;

    u64 attacks = square_bb(sq);
    if (color == PackedColor::WHITE) {
        attacks <<= 8;
    } else {
        attacks >>= 8;
    }
    return ((attacks & NOT_A_FILE) >> 1) | ((attacks & NOT_H_FILE) << 1);
}

[[nodiscard]] __device__ __forceinline__ PackedBoard build_board_warp(const ColoredPiece *pieces, i32 lane) {
    PackedBoard board{};

    for (i32 sq = lane; sq < static_cast<i32>(BOARD_SIZE); sq += WARP_SIZE) {
        const PackedPieceType piece = decode_piece_type(pieces[sq]);
        if (piece == PackedPieceType::NONE) {
            continue;
        }

        const PackedColor color = decode_color(pieces[sq]);
        const u64 sq_bb = square_bb(sq);
        board.piece_bbs[to_index(piece) - 1] |= sq_bb;
        board.side_bbs[to_index(color)] |= sq_bb;
    }

    for (usize i = 0; i < 6; ++i) {
        board.piece_bbs[i] = warp_broadcast(warp_or(board.piece_bbs[i]));
    }
    for (usize i = 0; i < 2; ++i) {
        board.side_bbs[i] = warp_broadcast(warp_or(board.side_bbs[i]));
    }

    return board;
}

[[nodiscard]] __device__ __forceinline__ u8 ft_feature_class(i32 defended, i32 threatened, i32 opposite_color,
                                                             PackedPieceType piece_type) {
    return static_cast<u8>((((defended * 2 + threatened) * 2 + opposite_color) * 6 + (to_index(piece_type) - 1)));
}

template <usize HiddenSize>
[[nodiscard]] __device__ __forceinline__ usize ft_offset(u8 feature_class, i32 sq) {
    return ((static_cast<usize>(feature_class) << 6) + static_cast<usize>(sq)) * HiddenSize;
}

} // namespace network::cuda_common

#endif // CUDA_COMMON_CUH
