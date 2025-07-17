#include "uci.hpp"
#include "../chess/move_gen.hpp"
#include "../tests/perft.hpp"
#include "../util/string.hpp"
#include "../util/types.hpp"

#include <chrono>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace uci {

Options options;
Handler handler;

Handler::Handler() {
    options.add(
        std::make_unique<IntegerOption>("Hash", 16, 1, std::numeric_limits<i32>::max(), [](const Option &option) {
            // resize hash table
        }));
    options.add(std::make_unique<BoolOption>("UCI_Chess960", false));
    board_ = Board(STARTPOS_FEN);
}

void Handler::handle_perft(std::ostream &out, int depth) {
    const auto start = std::chrono::high_resolution_clock::now();
    const auto nodes = tests::perft_print(board_, depth, out);
    const auto end = std::chrono::high_resolution_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    const auto nps = static_cast<u64>(static_cast<double>(nodes) * 1e9 / elapsed.count());
    out << "Nodes searched: " << nodes << " (" << nps << "nps)\n";
}

void Handler::handle_setoption(std::ostream &out, const std::vector<std::string_view> &parts) {
    if (parts[1] != "name") {
        out << "invalid second argument, expected 'name'" << std::endl;
        return;
    }

    if (parts[3] != "value") {
        out << "invalid fourth argument, expected 'value'" << std::endl;
        return;
    }

    options.get(parts[2])->set_value(parts[4]);
}

void Handler::process_input(std::istream &in, std::ostream &out) {
    std::string line;
    while (std::getline(in, line)) {
        const auto parts = util::split_string(line);
        if (parts[0] == "uci") {
            out << "id name Vine\n";
            out << "id author Aron Petkovski, Jonathan Hallström\n";
            out << options;
            out << "uciok\n";
        } else if (parts[0] == "perft") {
            handle_perft(out, *util::parse_int(parts[1]));
        } else if (parts[0] == "print") {
            out << board_ << std::endl;
        } else if (parts[0] == "setoption") {
            handle_setoption(out, parts);
        } else if (parts[0] == "position") {
            if (parts[1] == "fen" || parts[1] == "startpos") {
                std::string fen;
                size_t moves_pos = line.find(" moves ");

                if (parts[1] == "fen") {
                    const size_t fen_start = line.find("fen ") + 4;
                    if (moves_pos != std::string::npos) {
                        fen = line.substr(fen_start, moves_pos - fen_start);
                    } else {
                        fen = line.substr(fen_start);
                    }
                } else { // startpos
                    fen = std::string(STARTPOS_FEN);
                }

                board_ = Board(fen);

                if (moves_pos != std::string::npos) {
                    std::istringstream moves_stream(line.substr(moves_pos + 7));
                    std::string move_str;
                    while (moves_stream >> move_str) {
                        board_.make_move(board_.create_move(move_str));
                    }
                }
            }

        }
    }
}

} // namespace uci
