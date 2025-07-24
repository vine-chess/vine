#ifndef ZOBRIST_HPP
#define ZOBRIST_HPP

#include "../util/multi_array.hpp"
#include "../util/random.hpp"
#include "../util/types.hpp"

using HashKey = u64;

namespace zobrist {

using PieceTable = MultiArray<HashKey, 6, 2, 64>;
using CastleRightsTable = std::array<HashKey, 16>;
using EnPassantTable = std::array<HashKey, 8>;

const auto side_to_move = rng::next_u64();

const auto pieces = [] {
    PieceTable piece_table;
    for (auto &table : piece_table)
        for (auto &color : table)
            for (u64 &square : color)
                square = rng::next_u64();
    return piece_table;
}();

const auto castle_rights = [] {
    CastleRightsTable castle_rights_table;
    castle_rights_table[1] = rng::next_u64();
    castle_rights_table[2] = rng::next_u64();
    castle_rights_table[4] = rng::next_u64();
    castle_rights_table[8] = rng::next_u64();

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

    return castle_rights_table;;
}();

const auto en_passant = [] {
    EnPassantTable en_passant_table;
    for (u64 &entry : en_passant_table)
        entry = rng::next_u64();
    return en_passant_table;
}();

} // namespace zobrist

#endif // ZOBRIST_HPP
