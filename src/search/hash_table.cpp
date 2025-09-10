#include "hash_table.hpp"
#include <algorithm>

namespace search {

void HashTable::set_entry_capacity(usize capacity) {
    table_.clear();
    table_.shrink_to_fit();
    table_.resize(capacity);
}

void HashTable::clear() {
    std::ranges::fill(table_, HashEntry{});
}

const HashEntry *HashTable::probe(const HashKey &hash_key) const {
    const auto entry = &table_[index(hash_key)];
    return entry->compressed_hash_key == static_cast<u16>(hash_key) ? entry : nullptr;
}

void HashTable::update_q(const HashKey &hash_key, f64 q, u16 num_visits) {
    auto &entry = table_[index(hash_key)];
    if (entry.compressed_hash_key != static_cast<u16>(hash_key) || num_visits >= entry.num_visits) {
        entry.compressed_hash_key = static_cast<u16>(hash_key);
        entry.num_visits = num_visits;
        entry.q = q;
    }
}

void HashTable::update_policy_score(const HashKey &hash_key, f32 policy_score) {
    auto &entry = table_[index(hash_key)];
    entry.compressed_hash_key = static_cast<u16>(hash_key);
    entry.policy_score = static_cast<u16>(policy_score / 32768.0);
}

usize HashTable::index(const HashKey &hash_key) const {
    return hash_key % table_.size();
}

} // namespace search