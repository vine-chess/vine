#ifndef UCI_HPP
#define UCI_HPP

#include "../chess/board.hpp"
#include "../search/searcher.hpp"
#include "options.hpp"

#include <iostream>

namespace uci {

class Handler {
  public:
    Handler();

    void process_input(std::istream &in, std::ostream &out);

  private:
    void handle_perft(std::ostream &out, int depth);
    void handle_setoption(std::ostream &out, const std::vector<std::string_view> &parts);
    void handle_go(std::ostream &out, const std::vector<std::string_view> &parts);

    Board board_;
    search::Searcher searcher_;
};

extern Handler handler;
extern Options options;

} // namespace uci

#endif // UCI_HPP
