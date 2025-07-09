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

bool CastleRights::can_castle(Color color) const {
    return can_kingside_castle(color) || can_queenside_castle(color);
}

void CastleRights::set_kingside_castle(Color color, bool enabled) {
    const u8 mask = MASKS[color][1];
    enabled ? rights_ |= mask : rights_ &= ~mask;
}

void CastleRights::set_queenside_castle(Color color, bool enabled) {
    const u8 mask = MASKS[color][0];
    enabled ? rights_ |= mask : rights_ &= ~mask;
}

u8 CastleRights::operator&=(u8 mask) {
    rights_ &= mask;
    return rights_;
}