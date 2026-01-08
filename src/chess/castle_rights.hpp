#ifndef CASTLE_RIGHTS_HPP
#define CASTLE_RIGHTS_HPP

#include "../util/types.hpp"

constexpr i32 KINGSIDE = 0;
constexpr i32 QUEENSIDE = 1;

class CastleRights {
  public:
    CastleRights() = default;
    ~CastleRights() = default;

    [[nodiscard]] bool can_kingside_castle(Color color) const;
    [[nodiscard]] bool can_queenside_castle(Color color) const;
    [[nodiscard]] Square kingside_rook(Color color) const;
    [[nodiscard]] Square queenside_rook(Color color) const;
    [[nodiscard]] File kingside_rook_file(Color color) const;
    [[nodiscard]] File queenside_rook_file(Color color) const;
    [[nodiscard]] Square kingside_rook_dest(Color color) const;
    [[nodiscard]] Square queenside_rook_dest(Color color) const;
    [[nodiscard]] Square kingside_king_dest(Color color) const;
    [[nodiscard]] Square queenside_king_dest(Color color) const;

    void clear_kingside_availability(Color color);
    void clear_queenside_availability(Color color);
    void set_kingside_rook_file(Color color, File file);
    void set_queenside_rook_file(Color color, File file);

    [[nodiscard]] u8 to_mask() const;
    [[nodiscard]] u8 to_monty_mask() const;

  private:
    u16 raw_ = 0;
};

#endif // CASTLE_RIGHTS_HPP
