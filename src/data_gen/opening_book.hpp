#ifndef VINE_OPENING_BOOK_HPP
#define VINE_OPENING_BOOK_HPP

#include "../third_party/mio.h"
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace datagen {

class OpeningBook {
  public:
    explicit OpeningBook(const std::string &path) {
        mmap_ = mio::mmap_source(path);

        offsets_.reserve(60'000'000);
        offsets_.push_back(0);

        for (std::size_t i = 0; i < mmap_.size(); ++i) {
            if (mmap_[i] == '\n' && i + 1 < mmap_.size())
                offsets_.push_back(static_cast<uint64_t>(i + 1));
        }
    }

    std::string_view get(uint64_t idx) const {
        const char *base = mmap_.data();
        const uint64_t start = offsets_[idx];
        const uint64_t end = (idx + 1 < offsets_.size()) ? offsets_[idx + 1] - 1 : static_cast<uint64_t>(mmap_.size());

        const char *ptr = base + start;
        uint64_t len = end - start;

        // Handle CRLF
        if (len && ptr[len - 1] == '\r')
            --len;

        return std::string_view(ptr, len);
    }

    uint64_t size() const {
        return offsets_.size();
    }

  private:
    mio::mmap_source mmap_;
    std::vector<uint64_t> offsets_;
};

} // namespace datagen

#endif // VINE_OPENING_BOOK_HPP