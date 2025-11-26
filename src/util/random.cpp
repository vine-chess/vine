#include "random.hpp"

namespace rng {

thread_local std::mt19937_64 generator(DEFAULT_SEED);

} // namespace rng
