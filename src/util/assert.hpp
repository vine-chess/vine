#ifndef ASSERT_HPP
#define ASSERT_HPP

#include <cstdlib>

namespace util {
constexpr void assert_impl(bool x) {
    if (!x) {
        std::exit(-1);
    }
}
#ifdef NDEBUG
#define vine_assert(x)
#else
#define vine_assert(x) util::assert_impl(x)
#endif

} // namespace util

#endif // ASSERT_HPP
