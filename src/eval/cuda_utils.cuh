#ifndef EVAL_CUDA_UTILS_CUH
#define EVAL_CUDA_UTILS_CUH

#include "../util/types.hpp"

#include <cstddef>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <source_location>
#include <type_traits>

namespace network::cuda_common {

constexpr i32 WARP_SIZE = 32;

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

[[nodiscard]] __device__ __forceinline__ int fns64(u64 val, int n) {
    const uint32_t low = static_cast<uint32_t>(val);
    const uint32_t high = static_cast<uint32_t>(val >> 32);
    const int c = __popc(low);
    n++;
    return (n <= c) ? static_cast<int>(__fns(low, 0, n)) : static_cast<int>(__fns(high, 0, n - c) | 32);
}

template <typename Func>
__device__ __forceinline__ void warp_for_each_bit(u64 occ, int lane, Func func) {
    const int sq = fns64(occ, lane);
    func(sq, sq != -1);
}

[[nodiscard]] __device__ __forceinline__ i32 clamp_i32(i32 value, i32 min_value, i32 max_value) {
    if (value < min_value) return min_value;
    if (value > max_value) return max_value;
    return value;
}

} // namespace network::cuda_common

#endif // EVAL_CUDA_UTILS_CUH
