#include "uci.hpp"
#include "../util/string.hpp"
#include "../util/types.hpp"

#include <iostream>
#include <string>
#include <string_view>
#include <vector>
#include <chrono>

namespace uci {
	void UCIHandler::handle_perft(std::ostream &out, int depth) {
		const auto start = std::chrono::high_resolution_clock::now(); 
		const auto nodes = board_.perft(depth);
		const auto end = std::chrono::high_resolution_clock::now(); 
		const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
		const auto nps = static_cast<u64>(static_cast<double>(nodes) * 1e9 / elapsed.count());
		out << "Nodes searched: " << nodes << " (" << nps << "nps)\n";
	}

	void UCIHandler::process_input(std::istream& in, std::ostream& out) {
		std::string line;
		while (std::getline(in, line)) {
			std::vector<std::string_view> parts = util::string::split(line);
			if (parts[0] == "uci") {
				out << "id name Vine\n";
				out << "id author Aron Petkovski, Jonathan Hallström\n";
				out << "uciok\n";
			} else if (parts[0] == "perft") {
				handle_perft(out, 0);
			}
		}
	}

}
