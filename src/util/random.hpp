#ifndef RANDOM_HPP
#define RANDOM_HPP

#include "types.hpp"
#include <mutex>
#include <random>
#include <vector>

namespace rng {

constexpr int DEFAULT_SEED = 0x1337;
inline thread_local std::mt19937_64 generator(DEFAULT_SEED);

template <typename... Args>
inline void seed(u64 base_seed, Args... additional_seeds) {
    std::random_device rd;
    std::vector<unsigned> seed_data{static_cast<unsigned>(rd()),      static_cast<unsigned>(rd()),
                                    static_cast<unsigned>(rd()),      static_cast<unsigned>(rd()),
                                    static_cast<unsigned>(base_seed), static_cast<unsigned>(additional_seeds)...};
    std::seed_seq seq(seed_data.begin(), seed_data.end());
    generator.seed(seq);
}

inline u64 next_u64() {
    return generator();
}

inline u64 next_u64(u64 min, u64 max) {
    std::uniform_int_distribution<u64> dist(min, max - 1);
    return dist(generator);
}

inline double next_double() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(generator);
}

} // namespace rng

#endif // RANDOM_HPP
