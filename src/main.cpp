#include "uci/uci.hpp"
#include <sstream>

int main(int argv, char **argc) {
    std::stringstream cli_arg_stream;
    for (int i = 1; i < argv; ++i) {
        cli_arg_stream << argc[i] << std::endl;
    }
    uci::handler.process_input(cli_arg_stream, std::cout);
    uci::handler.process_input(std::cin, std::cout);
}