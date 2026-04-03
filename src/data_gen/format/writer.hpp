
#ifndef DATAGEN_WRITER_HPP
#define DATAGEN_WRITER_HPP

#include "../../util/types.hpp"
#include <cstring>
#include <iostream>
#include <ostream>

namespace datagen {

class Writer {
  public:
    explicit Writer(std::ostream &out) : out_(out) {}

    void put_u8(u8 v) {
        out_.put(static_cast<char>(v));
    }

    void put_u16(u16 v) {
        put_raw_bytes(v);
    }

    void put_u64(u64 v) {
        put_raw_bytes(v);
    }

    template <class T>
    void put_raw_bytes(T const &x) {
        char buf[sizeof(T)];
        std::memcpy(buf, &x, sizeof(T));
        out_.write(buf, sizeof(T));
    }

    void flush() {
        out_.flush();
    }

  private:
    std::ostream &out_;
};

} // namespace datagen

#endif //  DATAGEN_WRITER_HPP
