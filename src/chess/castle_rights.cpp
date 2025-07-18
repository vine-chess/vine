#include "castle_rights.hpp"

constexpr i32 KINGSIDE = 0;
constexpr i32 QUEENSIDE = 1;

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
