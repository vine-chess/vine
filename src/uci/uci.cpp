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
            using namespace std::string_view_literals;
            if (parts[1] == "fen") {
                board_ = Board(std::string_view(line).substr("position fen "sv.size()));
            } else if (parts[1] == "startpos") {
            }
        }
    }
}

} // namespace uci
