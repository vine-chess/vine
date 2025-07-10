#pragma once

#include "../chess/board.hpp"
#include <iostream>

namespace uci {

class Handler {
  public:
    void process_input(std::istream &in, std::ostream &out);

  private:
    void handle_perft(std::ostream &out, int depth);

    Board board_;
};

extern Handler handler;

} // namespace uci
