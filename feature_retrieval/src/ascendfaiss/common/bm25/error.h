#ifndef BM25_COMMON_ERROR_H
#define BM25_COMMON_ERROR_H
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ascend::bm25 {
class InvalidState : public std::runtime_error {
public:
    explicit InvalidState(const std::string &message) : std::runtime_error(message) {}
};
class InvalidArgument : public std::runtime_error {
public:
    explicit InvalidArgument(const std::string &message) : std::runtime_error(message) {}
};

class DeviceError : public std::runtime_error {
public:
    DeviceError(int64_t code, const std::string &message) : std::runtime_error(message), code_(code) {}
    int64_t GetCode() const noexcept { return code_; }

private:
    int64_t code_;
};

inline void CheckArgument(bool valid, const std::string &message)
{
    if (!valid) {
        throw InvalidArgument(message);
    }
}
} // namespace ascend::bm25
#endif // BM25_COMMON_ERROR_H
