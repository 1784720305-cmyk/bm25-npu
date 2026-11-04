#ifndef BM25_TEST_INPUT_IO_H
#define BM25_TEST_INPUT_IO_H
#include "common/bm25/index_data.h"
#include "common/bm25/search_inputs.h"
#include <filesystem>
#include <fstream>
#include <cstring>
namespace ascend::bm25 {
namespace fs = std::filesystem;
template <class T> std::vector<T> Read(const fs::path &path, bool allowEmpty = false)
{
    const auto size = fs::file_size(path);
    Check((allowEmpty || size > 0) && size % sizeof(T) == 0 && size / sizeof(T) <= INT32_MAX,
          "Invalid binary input size: " + path.string());
    const size_t bytes = Bytes(size / sizeof(T), sizeof(T));
    std::vector<T> values(bytes / sizeof(T));
    std::ifstream file(path, std::ios::binary);
    Check(file.is_open(), "Cannot open " + path.string());
    if (bytes)
        Check(static_cast<bool>(file.read(reinterpret_cast<char *>(values.data()), bytes)),
              "Cannot read " + path.string());
    return values;
}
template <class T> void Write(const fs::path &path, const std::vector<T> &values)
{
    std::ofstream file(path, std::ios::binary);
    const size_t bytes = Bytes(values.size(), sizeof(T));
    if (bytes)
        file.write(reinterpret_cast<const char *>(values.data()), bytes);
    file.close();
    Check(static_cast<bool>(file), "Cannot save " + path.string());
}

inline int64_t Number(const char *text, int64_t low, int64_t high)
{
    size_t parsed = 0;
    const int64_t value = std::stoll(text, &parsed);
    Check(parsed == std::strlen(text) && value >= low && value <= high, "Invalid integer argument");
    return value;
}

inline Index LoadIndex(const fs::path &directory)
{
    return Index(Read<int64_t>(directory / "term_offsets.bin"), Read<int32_t>(directory / "posting_doc_ids.bin"),
                 Read<float>(directory / "posting_tf.bin"), Read<float>(directory / "doc_norm.bin"),
                 Read<float>(directory / "idf.bin"));
}
} // namespace ascend::bm25
namespace bm25search {
// File parsing belongs to the benchmark caller, before it owns an in-memory Queries object.
// Conversion from that object to backend offsets/flat words MUST happen inside Search().
inline Queries LoadQueries(const std::string &path)
{
    std::ifstream file(path);
    std::string magic;
    int64_t q = 0;
    InputCheck(static_cast<bool>(file >> magic >> q) && magic == "BM25_QUERY_IDS_V1" && q >= 1 && q <= 10000,
               "Invalid raw query ID file header");
    Queries result(static_cast<size_t>(q));
    size_t total = 0;
    for (auto &row : result) {
        int64_t count = 0;
        InputCheck(static_cast<bool>(file >> count) && count >= 0 && count <= 16777216 &&
                       total <= 16777216 - static_cast<size_t>(count),
                   "Invalid raw query length/budget");
        total += static_cast<size_t>(count);
        row.reserve(static_cast<size_t>(count));
        for (int64_t i = 0; i < count; ++i) {
            int64_t word = 0;
            InputCheck(static_cast<bool>(file >> word) && word >= 0 && word < INT32_MAX, "Invalid raw query word ID");
            row.push_back(static_cast<int32_t>(word));
        }
    }
    file >> std::ws;
    InputCheck(file.eof(), "Unexpected trailing raw query data");
    return result;
}

} // namespace bm25search
#endif // BM25_TEST_INPUT_IO_H
