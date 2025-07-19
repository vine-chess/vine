#include "move.hpp"
#include "../uci/uci.hpp"

std::string Move::to_string() const {
    std::ostringstream ss;
    ss << *this;
    return ss.str();
}

std::ostream &operator<<(std::ostream &out, const Move &mv) {
    out << mv.from();
    Square to_sq = mv.to();
    if (mv.is_castling() && !std::get<bool>(uci::options.get("UCI_Chess960")->value_as_variant())) {
        to_sq = mv.king_castling_to();
    }
    out << to_sq;
    if (mv.is_promo()) {
        out << mv.promo_type().to_char();
    }
    return out;
}