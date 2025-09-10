#ifndef HASH_HPP
#define HASH_HPP

#include "../chess/zobrist.hpp"
#include "../util/types.hpp"

#include <optional>
#include <vector>

namespace search {

struct HashEntry {
    u16 compressed_hash_key = 0;
    u16 num_visits = 0;
    u16 policy_score = 32768;
    f64 q = 0.0;
};

class HashTable {
  public:
    void set_entry_capacity(usize capacity);

    void clear();

    [[nodiscard]] const HashEntry *probe(const HashKey &hash_key) const;

    void update_q(const HashKey &hash_key, f64 q, u16 num_visits);
    void update_policy_score(const HashKey &hash_key, f32 policy_score);

  private:
    [[nodiscard]] usize index(const HashKey &hash_key) const;

    std::vector<HashEntry> table_;
};

} // namespace search

#endif // HASH_HPP