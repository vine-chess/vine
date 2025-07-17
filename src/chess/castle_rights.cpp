#include "castle_rights.hpp"

bool CastleRights::operator==(const CastleRights &other) const {
    return rights_ == other.rights_;
}

bool CastleRights::can_kingside_castle(Color color) const {
    return rights_ & MASKS[color][1];
}

bool CastleRights::can_queenside_castle(Color color) const {
    return rights_ & MASKS[color][0];
}

Square CastleRights::kingside_rook_sq(Color color) const {
    return rook_squares_[color][0];
}

Square CastleRights::queenside_rook_sq(Color color) const {
    return rook_squares_[color][1];
}

void CastleRights::clear_rights(Color color) {
    rights_ &= ~(MASKS[color][0] | MASKS[color][1]);
}

void CastleRights::set_kingside_castle(Color color, bool enabled) {
    const u8 mask = MASKS[color][1];
    enabled ? rights_ |= mask : rights_ &= ~mask;
}

void CastleRights::set_queenside_castle(Color color, bool enabled) {
    const u8 mask = MASKS[color][0];
    enabled ? rights_ |= mask : rights_ &= ~mask;
}

void CastleRights::set_kingside_rook_sq(Color color, Square sq) {
    rook_squares_[color][0] = sq;
}

void CastleRights::set_queenside_rook_sq(Color color, Square sq) {
    rook_squares_[color][1] = sq;
}

u8 CastleRights::operator&=(u8 mask) {
    rights_ &= mask;
    return rights_;
}