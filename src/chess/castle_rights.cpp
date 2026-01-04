#include "castle_rights.hpp"

bool CastleRights::can_kingside_castle(Color color) const {
    return (raw_ & (1 << (KINGSIDE + 2 * color))) != 0;
}

bool CastleRights::can_queenside_castle(Color color) const {
    return (raw_ & (1 << (QUEENSIDE + 2 * color))) != 0;
}

Square CastleRights::kingside_rook(Color color) const {
    return {color == Color::WHITE ? Rank::FIRST : Rank::EIGHTH, kingside_rook_file(color)};
}

Square CastleRights::queenside_rook(Color color) const {
    return {color == Color::WHITE ? Rank::FIRST : Rank::EIGHTH, queenside_rook_file(color)};
}

File CastleRights::kingside_rook_file(Color color) const {
    return File(raw_ >> (4 + 6 * color) & 0b111);
}

File CastleRights::queenside_rook_file(Color color) const {
    return File(raw_ >> (7 + 6 * color) & 0b111);
}

Square CastleRights::kingside_king_dest(Color color) const {
    return {color == Color::WHITE ? Rank::FIRST : Rank::EIGHTH, File::G};
}

Square CastleRights::queenside_king_dest(Color color) const {
    return {color == Color::WHITE ? Rank::FIRST : Rank::EIGHTH, File::C};
}

Square CastleRights::kingside_rook_dest(Color color) const {
    return {color == Color::WHITE ? Rank::FIRST : Rank::EIGHTH, File::F};
}

Square CastleRights::queenside_rook_dest(Color color) const {
    return {color == Color::WHITE ? Rank::FIRST : Rank::EIGHTH, File::D};
}

void CastleRights::set_kingside_rook_file(Color color, File file) {
    raw_ |= file << 4 + 6 * color;
    raw_ |= 1 << (KINGSIDE + 2 * color);
}

void CastleRights::set_queenside_rook_file(Color color, File file) {
    raw_ |= file << 7 + 6 * color;
    raw_ |= 1 << (QUEENSIDE + 2 * color);
}

void CastleRights::clear_kingside_availability(Color color) {
    raw_ &= ~(1 << (KINGSIDE + 2 * color));
}
void CastleRights::clear_queenside_availability(Color color) {
    raw_ &= ~(1 << (QUEENSIDE + 2 * color));
}

u8 CastleRights::to_monty_mask() const {
    return can_kingside_castle(Color::WHITE) << 2 | can_queenside_castle(Color::WHITE) << 3 |
           can_kingside_castle(Color::BLACK) << 0 | can_queenside_castle(Color::BLACK) << 1;
}

u8 CastleRights::to_mask() const {
    return can_kingside_castle(Color::WHITE) << 0 | can_queenside_castle(Color::WHITE) << 1 |
           can_kingside_castle(Color::BLACK) << 2 | can_queenside_castle(Color::BLACK) << 3;
}
