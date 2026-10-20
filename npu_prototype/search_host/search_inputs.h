#pragma once
// Portable public input: already-known word IDs grouped by query. No ACL or index lookup.
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace bm25search {
using Queries = std::vector<std::vector<int32_t>>;
inline void InputCheck(bool ok,const char *message) { if (!ok) throw std::runtime_error(message); }
struct PackedQueries { std::vector<int64_t> offsets; std::vector<int32_t> words; };

inline PackedQueries PackQueries(const Queries &queries,size_t maxQueries,size_t maxWords)
{
    InputCheck(!queries.empty() && queries.size() <= maxQueries,"Query batch exceeds declared capacity or is empty");
    size_t total = 0;
    for (const auto &row : queries) {
        InputCheck(row.size() <= maxWords-total,"Word IDs exceed declared query capacity");
        total += row.size();
        for (int32_t word : row) InputCheck(word >= 0 && word < INT32_MAX,"Invalid public query word ID");
    }
    PackedQueries result; result.offsets.reserve(queries.size()+1); result.words.reserve(total);
    result.offsets.push_back(0);
    for (const auto &row : queries) {
        result.words.insert(result.words.end(),row.begin(),row.end());
        result.offsets.push_back(static_cast<int64_t>(result.words.size()));
    }
    return result;
}

// File parsing belongs to the benchmark caller, before it owns an in-memory Queries object.
// Conversion from that object to backend offsets/flat words MUST happen inside Search().
inline Queries LoadQueries(const std::string &path)
{
    std::ifstream file(path);
    std::string magic; int64_t q = 0;
    InputCheck(static_cast<bool>(file >> magic >> q) && magic == "BM25_QUERY_IDS_V1" && q >= 1 && q <= 10000,
               "Invalid raw query ID file header");
    Queries result(static_cast<size_t>(q));
    size_t total = 0;
    for (auto &row : result) {
        int64_t count = 0;
        InputCheck(static_cast<bool>(file >> count) && count >= 0 && count <= 16777216 &&
                   total <= 16777216-static_cast<size_t>(count),"Invalid raw query length/budget");
        total += static_cast<size_t>(count); row.reserve(static_cast<size_t>(count));
        for (int64_t i = 0; i < count; ++i) {
            int64_t word = 0;
            InputCheck(static_cast<bool>(file >> word) && word >= 0 && word < INT32_MAX,"Invalid raw query word ID");
            row.push_back(static_cast<int32_t>(word));
        }
    }
    file >> std::ws; InputCheck(file.eof(),"Unexpected trailing raw query data");
    return result;
}
}
