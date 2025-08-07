#include "castle_rights.hpp"

bool CastleRights::can_kingside_castle(Color color) const {
    return rook_files_[color][KINGSIDE] != File::NO_FILE;
}

bool CastleRights::can_queenside_castle(Color color) const {
    return rook_files_[color][QUEENSIDE] != File::NO_FILE;
}

Square CastleRights::kingside_rook(Color color) const {
    return {color == Color::WHITE ? Rank::FIRST : Rank::EIGHTH, rook_files_[color][KINGSIDE]};
}

Square CastleRights::queenside_rook(Color color) const {
    return {color == Color::WHITE ? Rank::FIRST : Rank::EIGHTH, rook_files_[color][QUEENSIDE]};
}

File CastleRights::kingside_rook_file(Color color) const {
    return rook_files_[color][KINGSIDE];
}

File CastleRights::queenside_rook_file(Color color) const {
    return rook_files_[color][QUEENSIDE];
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
    rook_files_[color][KINGSIDE] = file;
}

void CastleRights::set_queenside_rook_file(Color color, File file) {
    rook_files_[color][QUEENSIDE] = file;
}

u8 CastleRights::to_monty_mask() const {
    return can_kingside_castle(Color::WHITE) << 2 | can_queenside_castle(Color::WHITE) << 3 |
           can_kingside_castle(Color::BLACK) << 0 | can_queenside_castle(Color::BLACK) << 1;
}

u8 CastleRights::to_mask() const {
    return can_kingside_castle(Color::WHITE) << 0 | can_queenside_castle(Color::WHITE) << 1 |
           can_kingside_castle(Color::BLACK) << 2 | can_queenside_castle(Color::BLACK) << 3;
}
