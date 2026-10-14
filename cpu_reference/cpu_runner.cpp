// CPU computation only: the same prepared arrays and FP32 rules as the NPU pipeline.
// Never reads expected results; no CANN, search engine, or BM25S dependency.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);

void Require(bool ok, const std::string &message)
{
    if (!ok) throw std::runtime_error(message);
}

int64_t Integer(const char *value, const char *name)
{
    const std::string text(value);
    Require(!text.empty() && text.find_first_not_of("0123456789") == std::string::npos,
            std::string("Invalid integer: ") + name);
    size_t used = 0;
    const auto result = std::stoll(text, &used);
    Require(used == text.size(), std::string("Invalid integer: ") + name);
    return result;
}

template<class T> std::vector<T> Read(const fs::path &path)
{
    const auto bytes = fs::file_size(path);
    Require(bytes > 0 && bytes % sizeof(T) == 0 && bytes <= static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max()),
            "Invalid input size: " + path.string());
    std::vector<T> data(bytes / sizeof(T));
    std::ifstream stream(path, std::ios::binary);
    Require(static_cast<bool>(stream), "Cannot open: " + path.string());
    stream.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(bytes));
    Require(static_cast<bool>(stream), "Cannot read: " + path.string());
    return data;
}

template<class T> void Write(const fs::path &path, const std::vector<T> &values)
{
    std::ofstream stream(path, std::ios::binary);
    Require(static_cast<bool>(stream), "Cannot create: " + path.string());
    stream.write(reinterpret_cast<const char *>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(T)));
    stream.close();
    Require(static_cast<bool>(stream), "Cannot save: " + path.string());
}

struct Candidate {
    float score;
    int32_t id;
};
struct Better {
    bool operator()(const Candidate &a, const Candidate &b) const
    {
        return a.score > b.score || (a.score == b.score && a.id < b.id);
    }
};

void Contributions(const std::vector<float> &tf, const std::vector<float> &norm,
                   const std::vector<float> &idf, std::vector<float> &output)
{
    const auto count = static_cast<int64_t>(tf.size());
    // Independent records: SIMD is permitted here, with no fast-math or FMA contraction.
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < count; ++i) {
        const float numerator = tf[i] * 2.2F;
        output[i] = (idf[i] * numerator) / (tf[i] + norm[i]);
    }
}

void Accumulate(const std::vector<float> &values, const std::vector<int64_t> &offsets,
                std::vector<float> &scores)
{
    const auto groups = static_cast<int64_t>(scores.size());
    #pragma omp parallel for schedule(static)
    for (int64_t group = 0; group < groups; ++group) {
        float sum = 0.0F;
        // Preserve each group's addition order, exactly as in the first NPU operator.
        for (int64_t i = offsets[group]; i < offsets[group + 1]; ++i) sum += values[i];
        scores[group] = sum;
    }
}

void TopK(const std::vector<float> &scores, int64_t queries, int64_t docs, int64_t k,
          std::vector<float> &topScores, std::vector<int32_t> &topIds)
{
    #pragma omp parallel for schedule(static)
    for (int64_t query = 0; query < queries; ++query) {
        std::array<Candidate, 32> heap{};
        int64_t size = 0;
        for (int64_t doc = 0; doc < docs; ++doc) {
            const Candidate item{scores[query * docs + doc], static_cast<int32_t>(doc)};
            // Better as the heap comparator puts the worst retained candidate at the front.
            if (size < k) {
                heap[size++] = item;
                std::push_heap(heap.begin(), heap.begin() + size, Better{});
            } else if (Better{}(item, heap[0])) {
                std::pop_heap(heap.begin(), heap.begin() + size, Better{});
                heap[size - 1] = item;
                std::push_heap(heap.begin(), heap.begin() + size, Better{});
            }
        }
        std::sort_heap(heap.begin(), heap.begin() + size, Better{});
        for (int64_t rank = 0; rank < k; ++rank) {
            topScores[query * k + rank] = heap[rank].score;
            topIds[query * k + rank] = heap[rank].id;
        }
    }
}

double Percentile(std::vector<double> data, double fraction)
{
    std::sort(data.begin(), data.end());
    const size_t rank = static_cast<size_t>(std::ceil(fraction * data.size()));
    return data[std::max<size_t>(1, rank) - 1];
}

int main(int argc, char **argv)
{
    try {
        Require(argc == 9, "Usage: bm25_cpu_runner INPUT OUTPUT QUERIES DOCS K THREADS WARMUP REPEATS");
        const fs::path input = fs::canonical(argv[1]);
        const fs::path output = fs::absolute(argv[2]);
        Require(!fs::exists(output), "Output exists; choose a new directory");
        const auto queries = Integer(argv[3], "queries"), docs = Integer(argv[4], "docs");
        const auto k = Integer(argv[5], "k"), threads = Integer(argv[6], "threads");
        const auto warmup = Integer(argv[7], "warmup"), repeats = Integer(argv[8], "repeats");
        const int64_t limit = std::numeric_limits<int32_t>::max();
        Require(queries > 0 && docs > 0 && queries <= limit / docs, "Q*D exceeds current operator limits");
        Require(k >= 1 && k <= std::min<int64_t>(docs, 32), "Invalid K");
        Require(threads >= 1 && threads <= 65536 && warmup <= 10000 && repeats >= 1 && repeats <= 10000,
                "Invalid thread/iteration count");
        const uint16_t endian = 1;
        Require(*reinterpret_cast<const uint8_t *>(&endian) == 1, "This runner requires little-endian input/host");
        std::cout << "stage=validate_inputs\n";
        const auto tf = Read<float>(input / "tf.bin");
        const auto norm = Read<float>(input / "doc_norm.bin");
        const auto idf = Read<float>(input / "idf.bin");
        const auto offsets = Read<int64_t>(input / "offsets.bin");
        Require(tf.size() <= static_cast<size_t>(limit) && tf.size() == norm.size() && tf.size() == idf.size(),
                "TF/norm/IDF sizes mismatch");
        const int64_t groups = queries * docs;
        Require(offsets.size() == static_cast<size_t>(groups + 1) && offsets.front() == 0 &&
                offsets.back() == static_cast<int64_t>(tf.size()), "Offsets shape/endpoints mismatch");
        for (int64_t i = 0; i < groups; ++i) Require(offsets[i] <= offsets[i + 1], "Offsets are not monotone");
        for (size_t i = 0; i < tf.size(); ++i) {
            Require(std::isfinite(tf[i]) && tf[i] > 0 && tf[i] <= 16777216.0F && std::floor(tf[i]) == tf[i] &&
                    std::isfinite(norm[i]) && norm[i] >= 0 && std::isfinite(idf[i]) && idf[i] >= 0,
                    "Invalid TF/norm/IDF value");
        }
        std::vector<float> contributions(tf.size()), scores(groups), topScores(queries * k);
        std::vector<int32_t> topIds(queries * k);
        omp_set_dynamic(0);
        omp_set_num_threads(static_cast<int>(threads));
        int actualThreads = 0;
        #pragma omp parallel
        {
            #pragma omp single
            actualThreads = omp_get_num_threads();
        }
        const auto compute = [&]() {
            Contributions(tf, norm, idf, contributions);
            Accumulate(contributions, offsets, scores);
            TopK(scores, queries, docs, k, topScores, topIds);
        };
        std::cout << "values=" << tf.size() << " groups=" << groups << " requested_threads=" << threads
                  << " actual_threads=" << actualThreads << "\nstage=warmup\n";
        for (int64_t i = 0; i < warmup; ++i) compute();
        std::vector<double> timings;
        timings.reserve(repeats);
        std::cout << "stage=measure_resident_cpu_compute\n";
        for (int64_t i = 0; i < repeats; ++i) {
            const auto start = Clock::now();
            compute();
            timings.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
        }
        for (float value : contributions) Require(std::isfinite(value) && value >= 0, "Invalid computed contribution");
        for (float value : scores) Require(std::isfinite(value) && value >= 0, "Invalid computed score");
        Require(fs::create_directories(output), "Cannot create fresh output directory");
        Write(output / "contributions.bin", contributions);
        Write(output / "scores.bin", scores);
        Write(output / "top_scores.bin", topScores);
        Write(output / "top_ids.bin", topIds);
        std::ofstream report(output / "cpu_timing.json");
        report << std::setprecision(17)
               << "{\n  \"status\": \"cpu_computed_correctness_not_yet_checked\",\n"
               << "  \"scope\": \"resident three-stage CPU computation; excludes input packing, file IO and allocations; includes OpenMP scheduling\",\n"
               << "  \"fp32_flags\": \"-O3 -march=native -ffp-contract=off -fno-fast-math\",\n"
               << "  \"topk_algorithm\": \"bounded heap O(D log K) per query\",\n"
               << "  \"compiler\": \"" << __VERSION__ << "\",\n"
               << "  \"openmp_version\": " << _OPENMP << ",\n"
               << "  \"requested_threads\": " << threads << ",\n  \"actual_threads\": " << actualThreads << ",\n"
               << "  \"logical_processors\": " << omp_get_num_procs() << ",\n"
               << "  \"queries\": " << queries << ",\n  \"docs\": " << docs << ",\n  \"k\": " << k << ",\n"
               << "  \"warmup\": " << warmup << ",\n  \"repeats\": " << repeats << ",\n"
               << "  \"median_batch_ms\": " << Percentile(timings, .5) << ",\n"
               << "  \"p95_batch_ms\": " << Percentile(timings, .95) << ",\n"
               << "  \"samples_batch_ms\": [";
        for (size_t i = 0; i < timings.size(); ++i) report << (i ? ", " : "") << timings[i];
        report << "]\n}\n";
        report.close();
        Require(static_cast<bool>(report), "Cannot save timing report");
        std::cout << "stage=save_output\nresident_cpu_median_batch_ms=" << Percentile(timings, .5)
                  << "\nCPU computation completed; check all four output arrays separately. "
                  << "Do not compare with first-run NPU diagnostic timings.\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "CPU runner failed: " << error.what() << '\n';
        return 1;
    }
}
