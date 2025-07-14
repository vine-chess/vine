#include "magics.hpp"
#include "move_gen.hpp"
#include <vector>
#include <functional>

std::vector<Bitboard> create_blockers(Bitboard moves) {
    std::vector<u8> set_bits;
    set_bits.reserve(moves.pop_count());

    // store the indices (from the lsb) of each set bit in the moves bitboard
    for (int square = 0; square < 64; square++) {
        if (moves.is_set(square)) {
            set_bits.push_back(square);
        }
    }

    std::vector<Bitboard> blockers;
    Bitboard subset = moves;

    for (u64 i = 0; i <= 1ull << set_bits.size(); i++) {
        Bitboard blocker;
        for (const auto bit : set_bits) {
            // check if the bit is set in subset, not in the index
            if (subset.is_set(bit)) {
                blocker.set(bit);
            }
        }

        blockers.push_back(blocker);

        // Carey-Ripley method to get next blocker subset
        subset = (subset - 1) & moves;
    }

    return blockers;
}

Bitboard compute_bishop_attacks(Square sq, Bitboard occ) {
    auto up_left = Bitboard::get_ray_precomputed<UP, LEFT>(sq);
    auto up_right = Bitboard::get_ray_precomputed<UP, RIGHT>(sq);
    auto down_left = Bitboard::get_ray_precomputed<DOWN, LEFT>(sq);
    auto down_right = Bitboard::get_ray_precomputed<DOWN, RIGHT>(sq);

    up_left &= ~Bitboard::get_ray_precomputed<UP, LEFT>((up_left & occ).lsb());
    up_right &= ~Bitboard::get_ray_precomputed<UP, RIGHT>((up_right & occ).lsb());
    down_left &= ~Bitboard::get_ray_precomputed<DOWN, LEFT>((down_left & occ).msb());
    down_right &= ~Bitboard::get_ray_precomputed<DOWN, RIGHT>((down_right & occ).msb());

    return up_left | up_right | down_left | down_right;
}

Bitboard compute_rook_attacks(Square sq, Bitboard occ) {
    auto up = Bitboard::get_ray_precomputed<UP, 0>(sq);
    auto right = Bitboard::get_ray_precomputed<0, RIGHT>(sq);
    auto left = Bitboard::get_ray_precomputed<0, LEFT>(sq);
    auto down = Bitboard::get_ray_precomputed<DOWN, 0>(sq);

    up &= ~Bitboard::get_ray_precomputed<UP, 0>((up & occ).lsb());
    right &= ~Bitboard::get_ray_precomputed<0, RIGHT>((right & occ).lsb());
    left &= ~Bitboard::get_ray_precomputed<0, LEFT>((left & occ).msb());
    down &= ~Bitboard::get_ray_precomputed<DOWN, 0>((down & occ).msb());

    return up | right | left | down;
}

u32 get_bishop_attack_idx(Square sq, Bitboard occ) {
#ifdef USE_PEXT
    return _pext_u64(static_cast<u64>(occupied), BISHOP_MAGICS[square].mask);
#else
    const auto &entry = BISHOP_MAGICS[sq];
    return (static_cast<u64>(occ & entry.mask) * entry.magic) >> entry.shift;
#endif
}

u32 get_rook_attack_idx(Square sq, Bitboard occ) {
#ifdef USE_PEXT
    return _pext_u64(static_cast<u64>(occupied), ROOK_MAGICS[square].mask);
#else
    const auto &entry = ROOK_MAGICS[sq];
    return (static_cast<u64>(occ & entry.mask) * entry.magic) >> entry.shift;
#endif
}

template <typename AttacksTable, typename MagicEntry, usize BLOCKER_COMBINATIONS>
AttacksTable generate_attacks_table(const std::array<MagicEntry, 64> &magics,
                                    const std::function<Bitboard(Square, Bitboard)> &generate_moves_fn,
                                    const std::function<u64(Square, Bitboard)> &get_idx_fn) {
    AttacksTable attacks{};

    for (int square = 0; square < 64; square++) {
        auto blockers = create_blockers(magics[square].mask);
        std::array<Bitboard, BLOCKER_COMBINATIONS> square_attacks{};

        for (const auto &occupied : blockers) {
            const u64 index = get_idx_fn(Square(square), occupied);
            square_attacks[index] = generate_moves_fn(Square(square), occupied);
        }

        attacks[square] = square_attacks;
    }

    return attacks;
}

MultiArray<Bitboard, 64, 512> BISHOP_ATTACKS = []() {
    return generate_attacks_table<decltype(BISHOP_ATTACKS), decltype(BISHOP_MAGICS)::value_type, 512>(
        BISHOP_MAGICS, compute_bishop_attacks, get_bishop_attack_idx);
}();

MultiArray<Bitboard, 64, 4096> ROOK_ATTACKS = []() {
    return generate_attacks_table<decltype(ROOK_ATTACKS), decltype(ROOK_MAGICS)::value_type, 4096>(
        ROOK_MAGICS, compute_rook_attacks, get_rook_attack_idx);
}();
