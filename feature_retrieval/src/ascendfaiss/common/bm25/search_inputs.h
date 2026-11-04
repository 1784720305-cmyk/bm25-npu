#ifndef BM25_COMMON_BM25_SEARCH_INPUTS_H
#define BM25_COMMON_BM25_SEARCH_INPUTS_H
// Portable public input: already-known word IDs grouped by query. No ACL or index lookup.
#include "common/bm25/error.h"
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace bm25search {
using Queries = std::vector<std::vector<int32_t>>;
inline void InputCheck(bool ok, const char *message)
{
    if (!ok)
        throw ascend::bm25::InvalidArgument(message);
}
struct PackedQueries {
    std::vector<int64_t> offsets;
    std::vector<int32_t> words;
};

inline PackedQueries PackQueries(const Queries &queries, size_t maxQueries, size_t maxWords)
{
    InputCheck(!queries.empty() && queries.size() <= maxQueries, "Query batch exceeds declared capacity or is empty");
    size_t total = 0;
    for (const auto &row : queries) {
        InputCheck(row.size() <= maxWords - total, "Word IDs exceed declared query capacity");
        total += row.size();
        for (int32_t word : row)
            InputCheck(word >= 0 && word < INT32_MAX, "Invalid public query word ID");
    }
    PackedQueries result;
    result.offsets.reserve(queries.size() + 1);
    result.words.reserve(total);
    result.offsets.push_back(0);
    for (const auto &row : queries) {
        result.words.insert(result.words.end(), row.begin(), row.end());
        result.offsets.push_back(static_cast<int64_t>(result.words.size()));
    }
    return result;
}

} // namespace bm25search

#endif // BM25_COMMON_BM25_SEARCH_INPUTS_H
