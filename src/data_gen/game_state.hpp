#ifndef DATAGEN_GAME_STATE_HPP
#define DATAGEN_GAME_STATE_HPP

#include "../search/searcher.hpp"
#include "../util/ring_queue.hpp"
#include "format/monty_format.hpp"

#include <memory>

namespace datagen {

enum class State : u8 {
    Free,
    Search,
    Value,
    Policy,
    Write,
};

struct GameState {
    State state = State::Free;
    search::Searcher searcher;
    Board board;
    std::unique_ptr<MontyFormatWriter> writer;
    search::NodeIndex pending_node = search::NodeIndex::none();
};

struct GamePool {
    std::unique_ptr<GameState[]> games;
    util::RingQueue<GameState *> free;
    util::RingQueue<GameState *> search;
    util::RingQueue<GameState *> value;
    util::RingQueue<GameState *> policy;
    util::RingQueue<GameState *> write;

    void reset(usize count) {
        vine_assert(count > 0);

        games = std::make_unique<GameState[]>(count);
        free.reset(count);
        search.reset(count);
        value.reset(count);
        policy.reset(count);
        write.reset(count);

        for (usize i = 0; i < count; ++i) {
            (void) free.try_push(&games[i]);
        }
    }
};

} // namespace datagen

#endif // DATAGEN_GAME_STATE_HPP
