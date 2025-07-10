#include "uci.hpp"
#include "../util/string.hpp"
#include "../util/types.hpp"

#include <chrono>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace uci {

Handler handler;

void Handler::handle_perft(std::ostream &out, int depth) {
    const auto start = std::chrono::high_resolution_clock::now();
    const auto nodes = 0;
    const auto end = std::chrono::high_resolution_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    const auto nps = static_cast<u64>(static_cast<double>(nodes) * 1e9 / elapsed.count());
    out << "Nodes searched: " << nodes << " (" << nps << "nps)\n";
}

void Handler::process_input(std::istream &in, std::ostream &out) {
    std::string line;
    while (std::getline(in, line)) {
        std::vector<std::string_view> parts = util::split_string(line);
        if (parts[0] == "uci") {
            out << "id name Vine\n";
            out << "id author Aron Petkovski, Jonathan Hallström\n";
            out << "uciok\n";
        } else if (parts[0] == "perft") {
            handle_perft(out, 0);
        } else if (parts[0] == "print") {
            std::cout << board_ << std::endl;
            board_.state().make_move(Move(Square::E2, Square::E4));
            std::cout << board_ << std::endl;
            board_.state().make_move(Move(Square::D7, Square::D5));
            std::cout << board_ << std::endl;
            board_.state().make_move(Move(Square::E4, Square::D5, MoveFlag::CAPTURE_BIT));
            std::cout << board_ << std::endl;
        }
    }
}

} // namespace uci
