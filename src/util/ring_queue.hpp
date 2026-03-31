#ifndef RING_QUEUE_HPP
#define RING_QUEUE_HPP

#include "assert.hpp"
#include "types.hpp"
#include <atomic>
#include <bit>
#include <memory>
#include <type_traits>

namespace util {

template <class T>
class RingQueue {
    static_assert(std::is_trivially_copyable_v<T>, "RingQueue requires trivially copyable values");

  public:
    RingQueue() = default;

    explicit RingQueue(usize capacity) {
        reset(capacity);
    }

    void reset(usize capacity) {
        vine_assert(capacity > 0);

        capacity_ = std::bit_ceil(capacity);
        mask_ = capacity_ - 1;
        sequences_ = std::make_unique<std::atomic<usize>[]>(capacity_);
        values_ = std::make_unique<T[]>(capacity_);
        enqueue_pos_.store(0, std::memory_order_relaxed);
        dequeue_pos_.store(0, std::memory_order_relaxed);

        for (usize i = 0; i < capacity_; ++i) {
            sequences_[i].store(i, std::memory_order_relaxed);
        }
    }

    [[nodiscard]] bool try_push(T value) {
        usize pos = enqueue_pos_.load(std::memory_order_relaxed);

        while (true) {
            const usize idx = pos & mask_;
            const usize sequence = sequences_[idx].load(std::memory_order_acquire);
            const i64 diff = static_cast<i64>(sequence) - static_cast<i64>(pos);

            if (diff == 0) {
                if (enqueue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    values_[idx] = value;
                    sequences_[idx].store(pos + 1, std::memory_order_release);
                    return true;
                }
            } else if (diff < 0) {
                return false;
            } else {
                pos = enqueue_pos_.load(std::memory_order_relaxed);
            }
        }
    }

    [[nodiscard]] bool try_pop(T &value) {
        usize pos = dequeue_pos_.load(std::memory_order_relaxed);

        while (true) {
            const usize idx = pos & mask_;
            const usize sequence = sequences_[idx].load(std::memory_order_acquire);
            const i64 diff = static_cast<i64>(sequence) - static_cast<i64>(pos + 1);

            if (diff == 0) {
                if (dequeue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    value = values_[idx];
                    sequences_[idx].store(pos + capacity_, std::memory_order_release);
                    return true;
                }
            } else if (diff < 0) {
                return false;
            } else {
                pos = dequeue_pos_.load(std::memory_order_relaxed);
            }
        }
    }

    [[nodiscard]] usize capacity() const {
        return capacity_;
    }

  private:
    usize capacity_ = 0;
    usize mask_ = 0;
    std::unique_ptr<std::atomic<usize>[]> sequences_;
    std::unique_ptr<T[]> values_;
    alignas(64) std::atomic<usize> enqueue_pos_ = 0;
    alignas(64) std::atomic<usize> dequeue_pos_ = 0;
};

} // namespace util

#endif // RING_QUEUE_HPP
