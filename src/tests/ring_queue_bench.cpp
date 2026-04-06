#include "../util/ring_queue.hpp"
#include "../util/sharded_queue.hpp"
#include "../../../MPMCQueue/include/rigtorp/MPMCQueue.h"

#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::high_resolution_clock;
volatile u64 checksum_sink = 0;

void print_result(const char *name, const usize ops, const Clock::time_point begin, const u64 checksum) {
    checksum_sink ^= checksum;
    const f64 itemsps = ops / std::chrono::duration<f64>(Clock::now() - begin).count();
    std::cout << name << ": " << itemsps << "mitems/s\n";
}

template <class MakePusher, class MakePopper>
void bench(const char *name, const usize prod, const usize cons, const usize ops,
           MakePusher make_pusher, MakePopper make_popper) {
    const usize total = prod * ops;
    std::atomic<bool> start{false};
    std::atomic<usize> remaining{total};
    std::vector<u64> sums(cons, 0);
    std::vector<std::thread> threads;
    threads.reserve(prod + cons);

    for (usize p = 0; p < prod; ++p) {
        threads.emplace_back([&start, push = make_pusher(p), p, ops]() mutable {
            while (!start.load(std::memory_order_acquire)) {}
            const u64 base = p * ops;
            for (usize i = 0; i < ops; ++i) {
                while (!push(base + i)) {}
            }
        });
    }

    for (usize c = 0; c < cons; ++c) {
        threads.emplace_back([&start, &remaining, &sums, pop = make_popper(c), c]() mutable {
            while (!start.load(std::memory_order_acquire)) {}
            u64 sum = 0;
            std::array<u64, 128> values{};
            while (true) {
                const usize count = pop(std::span(values));
                if (count == 0) {
                    if (remaining.load(std::memory_order_relaxed) == 0) break;
                    continue;
                }
                for (usize i = 0; i < count; ++i) sum += values[i];
                if (remaining.fetch_sub(count, std::memory_order_relaxed) == count) break;
            }
            sums[c] = sum;
        });
    }

    const auto begin = Clock::now();
    start.store(true, std::memory_order_release);
    for (auto &t : threads) t.join();

    u64 sum = 0;
    for (const u64 s : sums) sum += s;
    print_result(name, total, begin, sum);
}

void bench_mpmc(const usize prod, const usize cons, const usize ops, const usize cap, const char *name) {
    util::RingQueue<u64> queue(cap);
    bench(name, prod, cons, ops,
        [&](usize) { return [&](u64 v) { return queue.try_push(v); }; },
        [&](usize) { return [&](std::span<u64> s) { return queue.try_pop_some(s); }; });
}

void bench_sharded(const usize prod, const usize cons, const usize ops, const usize cap, const char *name) {
    util::ShardedQueue<u64> queue;
    queue.reset(prod, cap);
    bench(name, prod, cons, ops,
        [&](usize) { return [tx = queue.sender()](u64 v) mutable { return tx.try_push(v); }; },
        [&](usize) { return [rx = queue.receiver()](std::span<u64> s) mutable { return rx.try_pop_some(s); }; });
}

void bench_rigtorp(const usize prod, const usize cons, const usize ops, const usize cap, const char *name) {
    rigtorp::mpmc::Queue<u64> queue(cap);
    bench(name, prod, cons, ops,
        [&](usize) { return [&](u64 v) { return queue.try_push(v); }; },
        [&](usize) {
            return [&](std::span<u64> s) {
                usize count = 0;
                while (count < s.size() && queue.try_pop(s[count])) ++count;
                return count;
            };
        });
}

} // namespace

int main() {
    constexpr usize ops = 2'000'000;
    constexpr usize small = 512;
    constexpr usize large = 4096;

    bench_mpmc(1, 1, ops, small, "mpmc 1x1 (512)");
    bench_mpmc(1, 1, ops, large, "mpmc 1x1 (4096)");
    bench_mpmc(4, 1, ops, small, "mpmc 4x1 (512)");
    bench_mpmc(4, 1, ops, large, "mpmc 4x1 (4096)");
    bench_mpmc(8, 1, ops, small, "mpmc 8x1 (512)");
    bench_mpmc(8, 1, ops, large, "mpmc 8x1 (4096)");
    bench_mpmc(4, 4, ops, small, "mpmc 4x4 (512)");
    bench_mpmc(4, 4, ops, large, "mpmc 4x4 (4096)");
    bench_mpmc(8, 8, ops, small, "mpmc 8x8 (512)");
    bench_mpmc(8, 8, ops, large, "mpmc 8x8 (4096)");
    bench_rigtorp(1, 1, ops, small, "rigtorp 1x1 (512)");
    bench_rigtorp(1, 1, ops, large, "rigtorp 1x1 (4096)");
    bench_rigtorp(4, 1, ops, small, "rigtorp 4x1 (512)");
    bench_rigtorp(4, 1, ops, large, "rigtorp 4x1 (4096)");
    bench_rigtorp(8, 1, ops, small, "rigtorp 8x1 (512)");
    bench_rigtorp(8, 1, ops, large, "rigtorp 8x1 (4096)");
    bench_rigtorp(4, 4, ops, small, "rigtorp 4x4 (512)");
    bench_rigtorp(4, 4, ops, large, "rigtorp 4x4 (4096)");
    bench_rigtorp(8, 8, ops, small, "rigtorp 8x8 (512)");
    bench_rigtorp(8, 8, ops, large, "rigtorp 8x8 (4096)");
    bench_sharded(1, 1, ops, small, "sharded 1x1 (512)");
    bench_sharded(1, 1, ops, large, "sharded 1x1 (4096)");
    bench_sharded(4, 1, ops, small, "sharded 4x1 (512)");
    bench_sharded(4, 1, ops, large, "sharded 4x1 (4096)");
    bench_sharded(8, 1, ops, small, "sharded 8x1 (512)");
    bench_sharded(8, 1, ops, large, "sharded 8x1 (4096)");
    bench_sharded(4, 4, ops, small, "sharded 4x4 (512)");
    bench_sharded(4, 4, ops, large, "sharded 4x4 (4096)");
    bench_sharded(8, 8, ops, small, "sharded 8x8 (512)");
    bench_sharded(8, 8, ops, large, "sharded 8x8 (4096)");
    return 0;
}
