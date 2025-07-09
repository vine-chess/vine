#pragma once
#include <iostream>
#include "../chess/board.hpp"

namespace uci {
	class Handler {
	public:
		void process_input(std::istream& in, std::ostream& out);

	private:
		void handle_perft(std::ostream& out, int depth);

		Board board_;
	};
}