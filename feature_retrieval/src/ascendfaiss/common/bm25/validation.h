#ifndef BM25_COMMON_VALIDATION_H
#define BM25_COMMON_VALIDATION_H
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ios>
#include <stdexcept>
#include <string>
namespace ascend::bm25 {
inline void Check(bool ok, const std::string &message)
{
    if (!ok)
        throw std::runtime_error(message);
}
inline size_t Bytes(uint64_t count, size_t width)
{
    Check(width > 0 && count <= std::numeric_limits<size_t>::max() / width &&
              count <= static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max()) / width,
          "Array byte count exceeds host limits");
    return static_cast<size_t>(count) * width;
}

} // namespace ascend::bm25
#endif // BM25_COMMON_VALIDATION_H
