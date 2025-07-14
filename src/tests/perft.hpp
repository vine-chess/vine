#ifndef TESTS_HPP
#define TESTS_HPP

#include "../chess/board.hpp"
#include "../util/types.hpp"

namespace tests {

struct PerftTestCase {
    std::string fen;
    int depth;
    u64 expected;
};

[[nodiscard]] u64 perft(Board &board, i32 depth);
[[nodiscard]] u64 perft_print(Board &board, i32 depth, std::ostream &out);

void run_perft_tests(const std::vector<PerftTestCase> &tests, std::ostream &out);

} // namespace tests

#endif // TESTS_HPP