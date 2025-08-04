#ifndef RANDOM_HPP
#define RANDOM_HPP

#include "types.hpp"
#include <iostream>
#include <mutex>
#include <random>
#include <vector>

namespace rng {

constexpr int DEFAULT_SEED = 0x1337;
inline thread_local std::mt19937_64 generator(DEFAULT_SEED);

template <typename... Args>
static void seed_generator(Args... args) {
    static_assert((std::is_convertible_v<Args, u32> && ...),
                  "All arguments to seed_generator must be convertible to u32");

    constexpr size_t N = sizeof...(Args);
    std::array<u32, N> seeds = {static_cast<u32>(args)...};

    std::seed_seq seq(seeds.begin(), seeds.end());
    generator.seed(seq);
}

static u64 next_u64() {
    return generator();
}

static u64 next_u64(u64 min, u64 max) {
    std::uniform_int_distribution dist(min, max);
    return dist(generator);
}

static f64 next_double() {
    std::uniform_real_distribution dist(0.0, 1.0);
    return dist(generator);
}

} // namespace rng

#endif // RANDOM_HPP
