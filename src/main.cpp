#include "eval/policy_queue.hpp"
#include "uci/uci.hpp"

int main(int argv, char **argc) {
    std::stringstream cli_arg_stream;
    for (int i = 1; i < argv; ++i) {
        cli_arg_stream << argc[i] << std::endl;
    }
    network::GlobalPolicyQueue::get().queue().start();
    network::GlobalValueQueue::get().queue().start();
    uci::handler.initialize_tunables();
    uci::handler.process_input(cli_arg_stream, std::cout);
    uci::handler.process_input(std::cin, std::cout);
}