#ifndef BM25_COMMON_INDEX_DATA_H
#define BM25_COMMON_INDEX_DATA_H
#include "common/bm25/validation.h"
#include "common/bm25/error.h"
#include <cmath>
#include <utility>
#include <vector>
namespace ascend::bm25 {
struct Index {
    std::vector<int64_t> offsets;
    std::vector<int32_t> docs;
    std::vector<float> tf, norm, idf;
    Index(std::vector<int64_t> termOffsets, std::vector<int32_t> postingIds, std::vector<float> termFrequencies,
          std::vector<float> documentNorms, std::vector<float> inverseFrequencies)
        : offsets(std::move(termOffsets)), docs(std::move(postingIds)), tf(std::move(termFrequencies)),
          norm(std::move(documentNorms)), idf(std::move(inverseFrequencies))
    {
        CheckArgument(!norm.empty() && !idf.empty() && norm.size() <= INT32_MAX && idf.size() < INT32_MAX &&
                          docs.size() <= INT32_MAX,
                      "Empty or oversized index");
        CheckArgument(offsets.size() == idf.size() + 1 && docs.size() == tf.size() && offsets.front() == 0 &&
                          offsets.back() == static_cast<int64_t>(docs.size()),
                      "Index dimensions/endpoints differ");
        for (float value : norm)
            CheckArgument(std::isfinite(value) && value > 0, "Invalid article norm");
        for (size_t term = 0; term < idf.size(); ++term) {
            CheckArgument(std::isfinite(idf[term]) && idf[term] >= 0, "Invalid IDF");
            const int64_t begin = offsets[term];
            const int64_t end = offsets[term + 1];
            CheckArgument(begin >= 0 && begin <= end && end <= static_cast<int64_t>(docs.size()),
                          "Invalid posting bounds");
            for (int64_t pos = begin; pos < end; ++pos) {
                CheckArgument(docs[pos] >= 0 && static_cast<size_t>(docs[pos]) < norm.size() &&
                                  (pos == begin || docs[pos - 1] < docs[pos]),
                              "Unsorted/duplicate/out-of-range posting ID");
                CheckArgument(std::isfinite(tf[pos]) && tf[pos] >= 1 && tf[pos] <= 16777216.0F &&
                                  std::floor(tf[pos]) == tf[pos],
                              "Invalid posting TF");
            }
        }
    }
};

} // namespace ascend::bm25
#endif // BM25_COMMON_INDEX_DATA_H
