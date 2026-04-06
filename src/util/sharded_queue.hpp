#ifndef SHARDED_QUEUE_HPP
#define SHARDED_QUEUE_HPP

#include "assert.hpp"
#include "types.hpp"
#include <atomic>
#include <bit>
#include <memory>
#include <span>
#include <type_traits>

namespace util {

template <class T>
class ShardedQueue {
    static_assert(std::is_trivially_copyable_v<T>, "ShardedQueue requires trivially copyable values");

    struct Shard {
        std::unique_ptr<T[]> buffer;
        usize capacity = 0;
        usize mask = 0;
        alignas(64) std::atomic<usize> head{0};
        alignas(64) std::atomic<usize> tail{0};
        alignas(64) std::atomic<bool> lock{false};
    };

    struct Local {
        usize head = 0;
        usize tail = 0;
    };

  public:
    class Sender;
    class Receiver;

    ShardedQueue() = default;

    void reset(usize n, usize cap) {
        vine_assert(n > 0);
        vine_assert(cap > 0);

        num_shards_ = n;
        shards_ = std::make_unique<Shard[]>(n);
        next_tx_.store(0, std::memory_order_relaxed);

        cap = std::bit_ceil(cap + 1);
        for (usize i = 0; i < n; ++i) {
            auto &s = shards_[i];
            s.capacity = cap;
            s.mask = cap - 1;
            s.buffer = std::make_unique<T[]>(cap);
            s.head.store(0, std::memory_order_relaxed);
            s.tail.store(0, std::memory_order_relaxed);
            s.lock.store(false, std::memory_order_relaxed);
        }
    }

    [[nodiscard]] Sender sender() {
        const usize idx = next_tx_.fetch_add(1, std::memory_order_relaxed);
        vine_assert(idx < num_shards_);
        return Sender(shards_[idx]);
    }

    [[nodiscard]] Receiver receiver() {
        return Receiver(shards_.get(), num_shards_);
    }

    class Sender {
      public:
        explicit Sender(Shard &shard) : shard_(shard), head_(0), tail_(0) {}

        [[nodiscard]] bool try_push(T value) {
            const usize next = (tail_ + 1) & shard_.mask;

            if (next == head_) {
                head_ = shard_.head.load(std::memory_order_acquire);
                if (next == head_) {
                    return false;
                }
            }

            shard_.buffer[tail_] = value;
            shard_.tail.store(next, std::memory_order_release);
            tail_ = next;
            return true;
        }

      private:
        Shard &shard_;
        usize head_;
        usize tail_;
    };

    class Receiver {
      public:
        Receiver(Shard *shards, usize n) : shards_(shards), num_shards_(n), next_(0) {
            locals_ = std::make_unique<Local[]>(n);
        }

        [[nodiscard]] bool try_pop(T &value) {
            return try_pop_some(std::span(&value, 1)) == 1;
        }

        [[nodiscard]] usize try_pop_some(std::span<T> values) {
            if (values.empty()) {
                return 0;
            }

            const usize start = next_;
            usize count = 0;

            do {
                const usize idx = next_;
                advance();

                auto &s = shards_[idx];
                auto &l = locals_[idx];

                if (l.head == l.tail) {
                    l.tail = s.tail.load(std::memory_order_acquire);
                    if (l.head == l.tail) {
                        continue;
                    }
                }

                if (s.lock.exchange(true, std::memory_order_acquire)) {
                    continue;
                }

                l.head = s.head.load(std::memory_order_relaxed);
                l.tail = s.tail.load(std::memory_order_acquire);

                while (l.head != l.tail && count < values.size()) {
                    values[count++] = s.buffer[l.head];
                    l.head = (l.head + 1) & s.mask;
                }

                s.head.store(l.head, std::memory_order_release);
                s.lock.store(false, std::memory_order_release);

                if (count != 0) {
                    return count;
                }
            } while (next_ != start);

            return count;
        }

      private:
        void advance() {
            ++next_;
            if (next_ == num_shards_) {
                next_ = 0;
            }
        }

        Shard *shards_;
        usize num_shards_;
        usize next_;
        std::unique_ptr<Local[]> locals_;
    };

  private:
    usize num_shards_ = 0;
    std::unique_ptr<Shard[]> shards_;
    std::atomic<usize> next_tx_{0};
};

} // namespace util

#endif // SHARDED_QUEUE_HPP
