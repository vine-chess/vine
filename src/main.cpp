#include "uci/uci.hpp"

int main() {
	uci::UCIHandler{}.process_input(std::cin, std::cout);
}
