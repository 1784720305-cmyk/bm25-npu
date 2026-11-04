#ifndef BM25_TEST_SESSION_DOUBLE_H
#define BM25_TEST_SESSION_DOUBLE_H
// Used only by facade tests. Device work and lifecycle are tested separately.
#include "ascendhost/include/impl/Bm25RuntimeBinding.h"
#include "common/bm25/index_data.h"
#include "common/bm25/search_inputs.h"
#include <filesystem>

namespace bm25test {
inline bool failSearch = false;
inline bool failClose = false;
} // namespace bm25test
namespace bm25large {
class Session {
public:
    struct Result {
        std::vector<float> scores;
        std::vector<int32_t> ids;
    };
    Session(const ascend::bm25::Index &index, const std::filesystem::path &vendor, int, uint64_t, uint64_t, uint64_t,
            const ascend::bm25::RuntimeBinding &binding)
        : index_(index), binding_(binding)
    {
        if (vendor == "fail_init") {
            throw ascend::bm25::DeviceError(42, "initialization failed");
        }
    }
    Result Search(const bm25search::Queries &queries)
    {
        if (failed_) {
            throw ascend::bm25::InvalidState("failed session");
        }
        ascend::bm25::CheckArgument(queries.size() == 1 && queries[0] == std::vector<int32_t>{0}, "invalid query");
        if (bm25test::failSearch) {
            failed_ = true;
            throw ascend::bm25::DeviceError(161001, "execute failed");
        }
        return {{index_.idf[0]}, {0}};
    }
    void Close()
    {
        if (bm25test::failClose) {
            throw ascend::bm25::DeviceError(43, "synchronization failed");
        }
    }

private:
    const ascend::bm25::Index &index_;
    ascend::bm25::RuntimeBinding binding_;
    bool failed_ = false;
};
} // namespace bm25large
#endif // BM25_TEST_SESSION_DOUBLE_H
