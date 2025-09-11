#ifndef ZOBRIST_HPP
#define ZOBRIST_HPP

#include "../util/multi_array.hpp"
#include "../util/random.hpp"
#include "../util/types.hpp"

using HashKey = u64;

namespace zobrist {

using PieceTable = util::MultiArray<HashKey, 6, 2, 64>;
using CastleRightsTable = std::array<HashKey, 16>;
using EnPassantTable = std::array<HashKey, 8>;

constexpr u64 murmur3(u64 x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccd;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53;
    x ^= x >> 33;
    return x;
}

constexpr auto side_to_move = [] {
    auto next_u64 = [hash = 0xDEADBEEFCAFEBABEull]() mutable { return hash = murmur3(hash); };
    return next_u64();
}();

constexpr auto pieces = [] {
    auto next_u64 = [hash = 12477279837831370886ull]() mutable { return hash = murmur3(hash); };
    PieceTable piece_table;
    for (auto &table : piece_table)
        for (auto &color : table)
            for (u64 &square : color)
                square = next_u64();
    return piece_table;
}();

constexpr auto castle_rights = [] {
    auto next_u64 = [hash = 13400036725371814030ull]() mutable { return hash = murmur3(hash); };
    CastleRightsTable castle_rights_table;
    castle_rights_table[1] = next_u64();
    castle_rights_table[2] = next_u64();
    castle_rights_table[4] = next_u64();
    castle_rights_table[8] = next_u64();

    for (i32 i = 0; i < 16; ++i) {
        if (i == 1 || i == 2 || i == 4 || i == 8) {
            continue;
        }

        HashKey key = 0;
        for (i32 bit = 1; bit <= 8; bit <<= 1) {
            if ((i & bit) != 0) {
                key ^= castle_rights_table[bit];
            }
        }
        castle_rights_table[i] = key;
    }

    return castle_rights_table;
}();

constexpr auto en_passant = [] {
    auto next_u64 = [hash = 2997978520062052832ull]() mutable { return hash = murmur3(hash); };
    EnPassantTable en_passant_table;
    for (u64 &entry : en_passant_table)
        entry = next_u64();
    return en_passant_table;
}();

} // namespace zobrist

#endif // ZOBRIST_HPP
