#ifndef STRING_HPP
#define STRING_HPP

#include <algorithm>
#include <cstddef>
#include <string_view>
#include <vector>

namespace util {

[[nodiscard]] inline std::vector<std::string_view> split_string(std::string_view input) {
    std::vector<std::string_view> res;
    size_t start_word = 0;
    while (start_word < input.size()) {
        const auto start_next_word = std::min(input.size(), input.find(' ', start_word + 1));
        const auto word = input.substr(start_word, start_next_word - start_word);
        if (!word.empty()) {
            res.push_back(word);
        }
        start_word = start_next_word + 1;
    }
    return res;
}

} // namespace util
#endif // STRING_HPP
