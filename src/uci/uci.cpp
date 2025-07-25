#include "uci.hpp"
#include "../chess/move_gen.hpp"
#include "../tests/bench.hpp"
#include "../tests/perft.hpp"
#include "../util/string.hpp"
#include "../util/types.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <ostream>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace uci {

Options options;
Handler handler;

Handler::Handler() {
    options.add(
        std::make_unique<IntegerOption>("Hash", 16, 1, std::numeric_limits<i32>::max(), [&](const Option &option) {
            searcher_.set_hash_size(std::get<i32>(option.value_as_variant()));
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

void Handler::handle_genfens(std::ostream &out, const std::vector<std::string_view> &parts) {
    vine_assert(parts[0] == "genfens");
    const auto count = *util::parse_int<usize>(parts[1]);
    vine_assert(parts[2] == "seed");
    const auto seed = *util::parse_int<usize>(parts[3]);
    vine_assert(parts[4] == "book");
    const auto path = parts[5];
    vine_assert(path == "None"); // TODO: book support, needs test
    const auto random_moves = parts.size() >= 7 ? *util::parse_int<usize>(parts[6]) : 8; // TODO:

    auto rng = std::mt19937_64(seed);

    const auto pick_random_move_count = [&]() { return random_moves; };

    Board board(STARTPOS_FEN);
    for (usize num_generated = 0; num_generated < count;) {
        board.undo_n_moves(board.history().size() - 1);

        const auto random_move_count = pick_random_move_count();

        auto generated_successfully = true;
        for (usize i = 0; i <= random_move_count; ++i) {
            MoveList moves;
            generate_moves(board.state(), moves);
            if (moves.empty() || board.is_fifty_move_draw() || board.has_threefold_repetition()) {
                generated_successfully = false;
                break;
            }
            // generated enough moves and didn't reach a terminal node
            if (i == random_move_count) {
                break;
            }

            auto distr = std::uniform_int_distribution<i32>(-100, 100);
            util::StaticVector<std::pair<Move, i32>, 218> scored_moves;
            for (auto move : moves) {
                scored_moves.push_back({move, distr(rng)});
            }
            const auto move = std::max_element(std::begin(scored_moves), std::end(scored_moves),
                                               [](auto &lhs, auto &rhs) { return lhs.second < rhs.second; });
            board.make_move(move->first);
        }

        if (generated_successfully) {
            out << "info string genfens " << board.state().to_fen() << '\n';
            ++num_generated;
        }
    }
}

void Handler::process_input(std::istream &in, std::ostream &out) {
    std::string line;
    while (std::getline(in, line)) {
        const auto parts = util::split_string(line);
        if (parts.empty()) {
            continue;
        }
        if (parts[0] == "uci") {
            out << "id name Vine" << std::endl;
            out << "id author Aron Petkovski, Jonathan Hallström" << std::endl;
            out << options;
            out << "uciok" << std::endl;
        } else if (parts[0] == "isready") {
            out << "readyok" << std::endl;
        } else if (parts[0] == "perft") {
            handle_perft(out, *util::parse_int(parts[1]));
        } else if (parts[0] == "print") {
            out << board_ << std::endl;
        } else if (parts[0] == "setoption") {
            handle_setoption(out, parts);
        } else if (parts[0] == "go") {
            handle_go(out, parts);
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
        } else if (parts[0] == "bench") {
            tests::run_bench_tests(out);
        } else if (parts[0] == "quit") {
            std::exit(0);
        } else if (parts[0] == "genfens") {
            handle_genfens(out, parts);
        }
    }
}

} // namespace uci
