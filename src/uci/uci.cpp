#include "uci.hpp"
#include "../chess/move_gen.hpp"
#include "../data_gen/game_runner.hpp"
#include "../eval/policy_network.hpp"
#include "../eval/value_network.hpp"
#include "../tests/bench.hpp"
#include "../tests/perft.hpp"
#include "../util/string.hpp"
#include "../util/types.hpp"
#include "options.hpp"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

namespace uci {

Options options;
Handler handler;

Handler::Handler() {
    options.add(std::make_unique<IntegerOption>("Threads", 1, 1, 1));
    options.add(
        std::make_unique<IntegerOption>("Hash", 16, 1, std::numeric_limits<i32>::max(), [&](const Option &option) {
            searcher_.set_hash_size(std::get<i32>(option.value_as_variant()));
        }));
    options.add(std::make_unique<BoolOption>("Minimal", false, [&](const Option &option) {
        searcher_.set_verbosity(std::get<bool>(option.value_as_variant()) ? search::Verbosity::MINIMAL
                                                                          : search::Verbosity::VERBOSE);
    }));
    options.add(std::make_unique<BoolOption>("UCI_Chess960", false));
    options.add(std::make_unique<IntegerOption>("KldMinGain", 0, 0, 100));
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
            time_settings.time_left_per_side[Color::WHITE] = *util::parse_number<i64>(parts[i + 1].data());
        } else if (parts[i] == "btime") {
            time_settings.time_left_per_side[Color::BLACK] = *util::parse_number<i64>(parts[i + 1].data());
        } else if (parts[i] == "winc") {
            time_settings.increment_per_side[Color::WHITE] = *util::parse_number<i64>(parts[i + 1].data());
        } else if (parts[i] == "binc") {
            time_settings.increment_per_side[Color::BLACK] = *util::parse_number<i64>(parts[i + 1].data());
        } else if (parts[i] == "nodes") {
            time_settings.max_iters = *util::parse_number<u64>(parts[i + 1].data());
        }
        time_settings.min_kld_gain = std::get<i32>(uci::options.get("KldMinGain")->value_as_variant()) / 10000000.0;
    }
    searcher_.go(board_, time_settings);
}

void Handler::handle_genfens(std::ostream &out, const std::vector<std::string_view> &parts) {
    vine_assert(parts[0] == "genfens");
    const auto count = *util::parse_number<usize>(parts[1]);
    vine_assert(parts[2] == "seed");
    const auto seed = *util::parse_number<usize>(parts[3]);
    vine_assert(parts[4] == "book");
    const auto path = parts[5];
    std::vector<std::string> opening_fens;
    if (path == "None") {
        opening_fens.push_back(std::string(STARTPOS_FEN));
    } else {
        std::ifstream book{std::string(path)};
        for (std::string opening; std::getline(book, opening);) {
            opening_fens.push_back(opening);
        }
    };

    usize random_moves = 15;
    f64 temperature = 1.5;
    f64 gamma = 1.1;

    constexpr std::string_view random_moves_str = "ply=";
    constexpr std::string_view temperature_str = "temp=";
    constexpr std::string_view gamma_str = "gamma=";
    for (const auto part : parts) {
        if (part.starts_with(random_moves_str)) {
            random_moves = *util::parse_number<usize>(part.substr(random_moves_str.length()));
        }
        if (part.starts_with(temperature_str)) {
            char *dummy;
            temperature = std::strtod(std::string(part.substr(temperature_str.length())).c_str(), &dummy);
        }
        if (part.starts_with(gamma_str)) {
            char *dummy;
            gamma = std::strtod(std::string(part.substr(gamma_str.length())).c_str(), &dummy);
        }
    }
    rng::seed_generator(seed);

    for (usize i = 0; i < count; ++i) {
        const auto opening = opening_fens[rng::next_u64(0, opening_fens.size() - 1)];
        out << "info string genfens " << datagen::generate_opening(opening, random_moves, temperature, gamma).to_fen()
            << std::endl;
    }
}

void Handler::handle_datagen(std::ostream &out, const std::vector<std::string_view> &parts) {
    datagen::Settings settings{};
    settings.random_moves = 8;
    settings.num_games = 1000;
    settings.num_threads = 1;
    settings.hash_size = 16;
    settings.time_settings = search::TimeSettings{};
    settings.output_file = "output.bin";

    for (size_t i = 1; i + 1 < parts.size(); i += 2) {
        const auto key = parts[i];
        const auto value = parts[i + 1];

        if (key == "random_moves") {
            settings.random_moves = *util::parse_number<usize>(value);
        } else if (key == "games") {
            settings.num_games = *util::parse_number<usize>(value);
        } else if (key == "threads") {
            settings.num_threads = *util::parse_number<usize>(value);
        } else if (key == "hash") {
            settings.hash_size = *util::parse_number<usize>(value);
        } else if (key == "nodes") {
            settings.time_settings.max_iters = *util::parse_number<u64>(value);
        } else if (key == "depth") {
            settings.time_settings.max_depth = *util::parse_number<i32>(value);
        } else if (key == "out") {
            settings.output_file = std::string(value);
        } else if (key == "temp" || key == "temperature") {
            char *dummy;
            settings.temperature = std::strtod(std::string(value).c_str(), &dummy);
        } else if (key == "gamma") {
            char *dummy;
            settings.gamma = std::strtod(std::string(value).c_str(), &dummy);
        } else if (key == "min_kld_gain") {
            char *dummy;
            settings.time_settings.min_kld_gain = std::strtod(std::string(value).c_str(), &dummy);
        } else if (key == "book") {
            settings.book_path = value;
        } else {
            out << "info string warning: unknown datagen key: " << key << std::endl;
        }
    }

    std::ofstream file_out(settings.output_file, std::ios::binary);
    if (!file_out) {
        out << "info string error: failed to open output file: " << settings.output_file << std::endl;
        return;
    }

    datagen::run_games(settings, out);
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
            handle_perft(out, *util::parse_number(parts[1]));
        } else if (parts[0] == "print") {
            out << network::value::evaluate(board_.state()) << '\n';
            MoveList moves;
            generate_moves(board_.state(), moves);

            const network::policy::PolicyContext ctx(board_.state());

            std::vector<f64> logits;
            logits.reserve(moves.size());
            for (const auto move : moves) {
                logits.push_back(ctx.logit(move));
            }

            if (!logits.empty()) {
                const f64 max_logit = *std::max_element(logits.begin(), logits.end());
                f64 sum_exp = 0.0;
                for (auto &val : logits) {
                    val = std::exp(val - max_logit); // stability
                    sum_exp += val;
                }
                const f64 inv_sum = 1.0 / sum_exp;
                for (auto &val : logits) {
                    val *= inv_sum;
                }
            }

            for (usize i = 0; i < moves.size(); ++i) {
                out << moves[i] << ": " << std::fixed << std::setprecision(2) << (100.0 * logits[i]) << '%' << '\n';
            }

            out << board_.state().to_fen() << std::endl;
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
#ifdef DATAGEN
        else if (parts[0] == "datagen") {
            handle_datagen(out, parts);
        }
#endif
    }
}

} // namespace uci
