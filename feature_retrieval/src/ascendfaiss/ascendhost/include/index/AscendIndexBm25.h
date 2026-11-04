#ifndef ASCEND_INDEX_BM25_H
#define ASCEND_INDEX_BM25_H
#include "common/ErrorCode.h"
#include "common/bm25/error_info.h"
#include "common/bm25/index_data.h"
#include "common/bm25/search_inputs.h"
#include <memory>
#include <cstdint>
#include <string>
#include <vector>

namespace ascend {
class AscendResourcesProxy;
}

namespace faiss::ascend {
struct Bm25Config {
    int deviceId = 0;
    uint64_t queryCount = 8;
    uint64_t maxQueryWords = 24;
    uint64_t topK = 10;
    std::string vendorDirectory;
    aclrtContext context = nullptr;
    std::shared_ptr<::ascend::AscendResourcesProxy> resources;
};

struct Bm25SearchResult {
    std::vector<float> scores;
    std::vector<int64_t> labels;
};

// Sparse, pre-tokenized BM25 interface; queryCount and topK are fixed at Init.
// The caller initializes SDK resources and activates config.context before Init,
// Search, Close and destruction. The class serializes calls to its own session.
// SDK resource uninitialization must occur after Close succeeds.
// Coordinate Init with other SDK activity because the current kernel uses global graph capture.
class AscendIndexBm25 {
public:
    AscendIndexBm25();
    ~AscendIndexBm25();
    AscendIndexBm25(const AscendIndexBm25 &) = delete;
    AscendIndexBm25 &operator=(const AscendIndexBm25 &) = delete;
    AscendIndexBm25(AscendIndexBm25 &&) = delete;
    AscendIndexBm25 &operator=(AscendIndexBm25 &&) = delete;

    // Copies and validates the in-memory index. Failed Init leaves the object uninitialized.
    ::ascend::APP_ERROR Init(const ::ascend::bm25::Index &index, const Bm25Config &config);
    // Output is replaced only on success. Input errors permit a subsequent valid request.
    ::ascend::APP_ERROR Search(const ::bm25search::Queries &queries, Bm25SearchResult &result);
    // Idempotent on a closed object. On failure, keeps state so Close can be retried.
    ::ascend::APP_ERROR Close();
    ::ascend::bm25::ErrorInfo GetLastError() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace faiss::ascend
#endif // ASCEND_INDEX_BM25_H
