#include "uci/uci.hpp"

int main() {
	uci::Handler{}.process_input(std::cin, std::cout);
}