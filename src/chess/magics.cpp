#include "magics.hpp"
#include "move_gen.hpp"
#include <functional>
#include <vector>

#define USE_HYPERBOLA_QUINTESSENCE

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

Bitboard compute_bishop_attacks(Square sq, Bitboard occ) {
    const Bitboard s = sq.to_bb();
    const Bitboard diag = static_cast<u64>(BISHOP_RAYS[sq] & Bitboard::get_ray_precomputed<UP, LEFT>(sq) |
                                           Bitboard::get_ray_precomputed<DOWN, RIGHT>(sq));
    const Bitboard adiag = static_cast<u64>(BISHOP_RAYS[sq] & Bitboard::get_ray_precomputed<UP, RIGHT>(sq) |
                                            Bitboard::get_ray_precomputed<DOWN, LEFT>(sq));
    const Bitboard occ_diag = occ & diag;
    const Bitboard occ_adiag = occ & adiag;

    const Bitboard attacks_diag =
        (occ_diag - 2 * static_cast<u64>(s)) ^
        Bitboard(occ_diag.reverse_bits() - 2 * static_cast<u64>(s.reverse_bits())).reverse_bits();
    const Bitboard attacks_adiag =
        (occ_adiag - 2 * static_cast<u64>(s)) ^
        Bitboard(occ_adiag.reverse_bits() - 2 * static_cast<u64>(s.reverse_bits())).reverse_bits();

    return (attacks_diag & diag) | (attacks_adiag & adiag);
}

Bitboard compute_rook_attacks(Square sq, Bitboard occ) {
    const Bitboard s = sq.to_bb();
    const Bitboard rank = static_cast<u64>(ROOK_RAYS[sq] & Bitboard::get_ray_precomputed<0, LEFT>(sq) |
                                           Bitboard::get_ray_precomputed<0, RIGHT>(sq));
    const Bitboard file = static_cast<u64>(ROOK_RAYS[sq] & Bitboard::get_ray_precomputed<UP, 0>(sq) |
                                           Bitboard::get_ray_precomputed<DOWN, 0>(sq));

    const Bitboard occ_rank = occ & rank;
    const Bitboard occ_file = occ & file;

    const Bitboard attacks_rank =
        (occ_rank - 2 * static_cast<u64>(s)) ^
        Bitboard(occ_rank.reverse_bits() - 2 * static_cast<u64>(s.reverse_bits())).reverse_bits();

    const Bitboard attacks_file =
        (occ_file - 2 * static_cast<u64>(s)) ^
        Bitboard(occ_file.reverse_bits() - 2 * static_cast<u64>(s.reverse_bits())).reverse_bits();

    return (attacks_rank & rank) | (attacks_file & file);
}

util::MultiArray<Bitboard, 64, 512> BISHOP_ATTACKS = []() {
    return generate_attacks_table<decltype(BISHOP_ATTACKS), decltype(BISHOP_MAGICS)::value_type, 512>(
        BISHOP_MAGICS, compute_bishop_attacks, get_bishop_attack_idx);
}();

util::MultiArray<Bitboard, 64, 4096> ROOK_ATTACKS = []() {
    return generate_attacks_table<decltype(ROOK_ATTACKS), decltype(ROOK_MAGICS)::value_type, 4096>(
        ROOK_MAGICS, compute_rook_attacks, get_rook_attack_idx);
}();

[[nodiscard]] Bitboard get_bishop_attacks(Square sq, Bitboard occ) {
#ifdef USE_HYPERBOLA_QUINTESSENCE
    return compute_bishop_attacks(sq, occ);
#endif
    return BISHOP_ATTACKS[sq][get_bishop_attack_idx(sq, occ)];
}

[[nodiscard]] Bitboard get_rook_attacks(Square sq, Bitboard occ) {
#ifdef USE_HYPERBOLA_QUINTESSENCE
    return compute_rook_attacks(sq, occ);
#endif
    return ROOK_ATTACKS[sq][get_rook_attack_idx(sq, occ)];
}
