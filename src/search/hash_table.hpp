#ifndef HASH_HPP
#define HASH_HPP

#include "../chess/zobrist.hpp"
#include "../util/types.hpp"

#include <optional>
#include <vector>

namespace search {

struct HashEntry {
    u16 compressed_hash_key = 0;
    f64 q = 0.0;
};

class HashTable {
  public:
    void set_entry_capacity(usize capacity);

    [[nodiscard]] const HashEntry *probe(const HashKey &hash_key) const;

    void update(const HashKey &hash_key, f64 q);

  private:
    [[nodiscard]] usize index(const HashKey &hash_key) const;

    std::vector<HashEntry> table_;
};

} // namespace search

#endif // HASH_HPP