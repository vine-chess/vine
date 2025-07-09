#ifndef CASTLE_RIGHTS_HPP
#define CASTLE_RIGHTS_HPP

#include "../util/types.hpp"

#include <array>

class CastleRights {
  public:
    CastleRights() = default;
    ~CastleRights() = default;

    bool operator==(const CastleRights &other) const;

    [[nodiscard]] bool can_kingside_castle(Color turn) const;
    [[nodiscard]] bool can_queenside_castle(Color turn) const;
    [[nodiscard]] bool can_castle(Color turn) const;

    void set_kingside_castle(Color turn, bool value);
    void set_queenside_castle(Color turn, bool value);

    u8 operator&=(u8 mask);

  private:
    u8 rights_;
    // [Color][0] = queenside, [Color][1] = kingside
    static constexpr std::array<std::array<u8, 2>, 2> MASKS = {{
        {{0b0001, 0b0010}}, // WHITE
        {{0b0100, 0b1000}}  // BLACK
    }};
};

#endif // CASTLE_RIGHTS_HPP