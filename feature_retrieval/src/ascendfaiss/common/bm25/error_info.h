#ifndef BM25_ERROR_INFO_H
#define BM25_ERROR_INFO_H
#include <array>
#include <cstdint>
#include <cstring>

namespace ascend::bm25 {
struct ErrorInfo {
    int64_t aclCode = 0;
    std::array<char, 1024> message{};

    void Set(const char *text, int64_t code = 0) noexcept
    {
        aclCode = code;
        message.fill('\0');
        if (text) {
            std::strncpy(message.data(), text, message.size() - 1);
        }
    }
};
} // namespace ascend::bm25
#endif // BM25_ERROR_INFO_H
