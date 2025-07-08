#include <iostream>
#include <string>

namespace uci {
	void process_input(std::istream& in, std::ostream& out) {
		std::string line;
		while (std::getline(in, line)) {
			if (line == "uci") {
				out << "id name Vine\n";
				out << "id author Aron Petkovski, Jonathan Hallström\n";
				out << "uciok\n";
			}
		}
	}
}
