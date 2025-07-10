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
    for (u64 &entry : castle_rights_table)
        entry = rng::next_u64();
    return castle_rights_table;
}();

const auto en_passant = [] {
    EnPassantTable en_passant_table;
    for (u64 &entry : en_passant_table)
        entry = rng::next_u64();
    return en_passant_table;
}();

} // namespace zobrist

#endif // ZOBRIST_HPP
