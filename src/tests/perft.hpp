#ifndef TESTS_HPP
#define TESTS_HPP

#include "../chess/board.hpp"
#include "../util/types.hpp"

namespace tests {

struct PerftBatch {
    std::string_view fen;
    std::vector<std::pair<i32, u64>> depths; // {depth, expected nodes}
};

const std::array PERFT_BATCHES = {
    PerftBatch{"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
               {{0, 1}, {1, 20}, {2, 400}, {3, 8902}, {4, 197281}, {5, 4865609}, {6, 119060324}, {7, 3195901860}}},
    PerftBatch{"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -",
               {{1, 48}, {2, 2039}, {3, 97862}, {4, 4085603}, {5, 193690690}}}};

[[nodiscard]] u64 perft(Board &board, i32 depth);
[[nodiscard]] u64 perft_print(Board &board, i32 depth, std::ostream &out);

void run_perft_tests(std::ostream &out);

} // namespace tests

#endif // TESTS_HPP