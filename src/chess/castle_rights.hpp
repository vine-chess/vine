#ifndef CASTLE_RIGHTS_HPP
#define CASTLE_RIGHTS_HPP

#include "../util/types.hpp"
#include "../util/multi_array.hpp"

#include <array>

class CastleRights {
  public:
    CastleRights() = default;
    ~CastleRights() = default;

    bool operator==(const CastleRights &other) const;

    [[nodiscard]] bool can_kingside_castle(Color color) const;
    [[nodiscard]] bool can_queenside_castle(Color color) const;
    [[nodiscard]] Square kingside_rook_sq(Color color) const;
    [[nodiscard]] Square queenside_rook_sq(Color color) const;

    void clear_rights(Color color);
    void set_kingside_castle(Color color, bool value);
    void set_queenside_castle(Color color, bool value);
    void set_kingside_rook_sq(Color color, Square sq);
    void set_queenside_rook_sq(Color color, Square sq);

    u8 operator&=(u8 mask);

  private:
    u8 rights_;
    MultiArray<Square, 2, 2> rook_squares_;
    // [Color][0] = queenside, [Color][1] = kingside
    static constexpr std::array<std::array<u8, 2>, 2> MASKS = {{
        {{0b0001, 0b0010}}, // WHITE
        {{0b0100, 0b1000}}  // BLACK
    }};
};

#endif // CASTLE_RIGHTS_HPP