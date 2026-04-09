#ifndef RANDOM_HPP
#define RANDOM_HPP

#include "types.hpp"
#include <array>
#include <limits>
#include <random>
#include <type_traits>
#include <utility>

namespace rng {

class FNVMixer {
  public:
    constexpr explicit FNVMixer(const u64 seed) : state_(seed) {}

    constexpr void add(const u64 seed) {
        state_ ^= seed;
        state_ *= 1099511628211ULL;
    }

    [[nodiscard]] constexpr u64 value() const {
        return state_;
    }

  private:
    u64 state_;
};

class SplitMix64 {
  public:
    constexpr explicit SplitMix64(const u64 seed) : state_(seed) {}

    [[nodiscard]] constexpr u64 next() {
        u64 z = (state_ += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

  private:
    u64 state_;
};

class Xoshiro256pp {
  public:
    using result_type = u64; // for STL compat

    constexpr Xoshiro256pp() {
        reset();
    }

    template <typename... Args>
    constexpr void seed(Args... args) {
        reset();
        add_entropy(std::forward<Args>(args)...);
    }

    template <typename... Args>
    constexpr void add_entropy(Args... args) {
        static_assert(sizeof...(Args) > 0);
        static_assert((std::is_convertible_v<Args, u64> && ...), "All arguments must be convertible to u64");

        FNVMixer mixer(DEFAULT_SEED);
        (mixer.add(static_cast<u64>(args)), ...);
        SplitMix64 sm(mixer.value());
        for (auto &v : state_) {
            v ^= sm.next();
        }
        ensure_nonzero();
    }

    [[nodiscard]] constexpr u64 operator()() {
        const u64 result = std::rotl(state_[0] + state_[3], 23) + state_[0];
        const u64 t = state_[1] << 17;

        state_[2] ^= state_[0];
        state_[3] ^= state_[1];
        state_[1] ^= state_[2];
        state_[0] ^= state_[3];
        state_[2] ^= t;
        state_[3] = std::rotl(state_[3], 45);

        return result;
    }

    [[nodiscard]] static constexpr u64 min() {
        return std::numeric_limits<u64>::min();
    }

    [[nodiscard]] static constexpr u64 max() {
        return std::numeric_limits<u64>::max();
    }

  private:
    static constexpr u64 DEFAULT_SEED = 0x1337;

    std::array<u64, 4> state_;

    constexpr void reset() {
        SplitMix64 sm(DEFAULT_SEED);
        for (auto &v : state_) {
            v = sm.next();
        }
    }

    [[nodiscard]] constexpr bool all_zero() const {
        u64 x = 0;
        for (const u64 v : state_) {
            x |= v;
        }
        return x == 0;
    }

    constexpr void ensure_nonzero() {
        if (all_zero()) {
            reset();
        }
    }
};

extern thread_local Xoshiro256pp generator;

template <typename... Args>
static void seed(Args... args) {
    generator.seed(args...);
}

template <typename... Args>
static void add_entropy(Args... args) {
    generator.add_entropy(args...);
}

template <typename... Args>
static void seed_generator(Args... args) {
    seed(args...);
}

static u64 next_u64() {
    return generator();
}

static u64 next_u64(u64 min, u64 max) {
    std::uniform_int_distribution dist(min, max);
    return dist(generator);
}

static f64 next_f64() {
    std::uniform_real_distribution dist(0.0, 1.0);
    return dist(generator);
}

static f64 next_f64_gamma(f64 alpha) {
    std::gamma_distribution<> dist(alpha, 1.0);
    return dist(generator);
}

} // namespace rng

#endif // RANDOM_HPP
