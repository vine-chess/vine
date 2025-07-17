#ifndef CASTLE_RIGHTS_HPP
#define CASTLE_RIGHTS_HPP

#include "../util/types.hpp"
#include "../util/multi_array.hpp"

#include <array>

class CastleRights {
  public:
    CastleRights() = default;
    ~CastleRights() = default;

    [[nodiscard]] bool can_kingside_castle(Color color) const;
    [[nodiscard]] bool can_queenside_castle(Color color) const;
    [[nodiscard]] Square kingside_rook_sq(Color color) const;
    [[nodiscard]] Square queenside_rook_sq(Color color) const;

    void set_kingside_rook_file(Color color, File file);
    void set_queenside_rook_file(Color color, File file);

  private:
    MultiArray<File, 2, 2> rook_files_;
};

#endif // CASTLE_RIGHTS_HPP