#include "uci.hpp"
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
    const auto nodes = 0;
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

    if (parts[3] != "value  ") {
        out << "invalid fourth argument, expected 'value'" << std::endl;
        return;
    }

    options.get(parts[2])->set_value(parts[4]);
}

void Handler::handle_go(std::ostream &out, const std::vector<std::string_view> &parts) {
    search::TimeSettings time_settings{};
    for (i32 i = 1; i < parts.size(); ++i) {
        if (parts[i] == "wtime") {
            time_settings.time_left_per_side[Color::WHITE] = std::atoi(parts[i + 1].data());
        } else if (parts[i] == "btime") {
            time_settings.time_left_per_side[Color::BLACK] = std::atoi(parts[i + 1].data());
        } else if (parts[i] == "winc") {
            time_settings.increment_per_side[Color::WHITE] = std::atoi(parts[i + 1].data());
        } else if (parts[i] == "binc") {
            time_settings.increment_per_side[Color::BLACK] = std::atoi(parts[i + 1].data());
        }
    }

    searcher_.go(board_, time_settings);
}

void Handler::process_input(std::istream &in, std::ostream &out) {
    std::string line;
    while (std::getline(in, line)) {
        const auto parts = util::split_string(line);
        if (parts[0] == "uci") {
            out << "id name Vine" << std::endl;
            out << "id author Aron Petkovski, Jonathan Hallström" << std::endl;
            out << options;
            out << "uciok" << std::endl;
        } else if (parts[0] == "perft") {
            handle_perft(out, 0);
        } else if (parts[0] == "print") {
            std::cout << board_ << std::endl;
        } else if (parts[0] == "setoption") {
            handle_setoption(out, parts);
        } else if (parts[0] == "go") {
            handle_go(out, parts);
        }
    }
}

} // namespace uci
