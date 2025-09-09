#include "hash_table.hpp"

namespace search {

void HashTable::set_entry_capacity(usize capacity) {
    table_.clear();
    table_.shrink_to_fit();
    table_.resize(capacity);
}

std::optional<const HashEntry *> HashTable::probe(const HashKey &hash_key) const {
    const auto entry = &table_[index(hash_key)];
    if (entry->compressed_hash_key == static_cast<u16>(hash_key)) {
        return entry;
    }
    return std::nullopt;
}

void HashTable::update(const HashKey &hash_key, f64 q) {
    auto &[compressed_hash_key, entry_q] = table_[index(hash_key)];
    compressed_hash_key = static_cast<u16>(hash_key);
    entry_q = q;
}

usize HashTable::index(const HashKey &hash_key) const {
    return hash_key % table_.size();
}

} // namespace search