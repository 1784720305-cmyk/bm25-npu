#include "acl/acl.h"
#include "ascendhost/src/bm25/large_search.h"
#include "common/bm25/large_search_plan.h"
#include "test/ascendhost/bm25/input_io.h"
#include <filesystem>
#include <cstddef>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <string>
#include <ios>
namespace bm25large {
using ascend::bm25::Write;
void Session::SaveDiagnostic(const Queries &queries, const PlannedResult &normal, const fs::path &path,
                             const fs::path &reference)
{
    Check(!closed && !failed, "Diagnostic on invalid large session");
    packet.Pack(queries, hostInput);
    PoisonForDiagnostic();
    const auto fullBytes = Bytes(plan.q * plan.padded, 4);
    MemoryGate(fullBytes + 128, "Full-score diagnostic exceeds device memory budget");
    Buffers extra;
    extra.stream = resources.runtime.stream;
    Tensors t;
    t.stream = resources.runtime.stream;
    auto *base = static_cast<unsigned char *>(extra.Allocate(fullBytes + 128));
    const std::vector<unsigned char> poison(8 * 1024 * 1024, 0xff);
    for (size_t pos = 0; pos < fullBytes + 128;) {
        const auto bytes = std::min(poison.size(), fullBytes + 128 - pos);
        Require(aclrtMemcpy(base + pos, bytes, poison.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE),
                "poison bounded diagnostic buffer");
        pos += bytes;
    }
    auto *full = t.Describe(plan.q, plan.padded, ACL_FLOAT, base + 64);
    auto prepared = Prepare(full, 1);
    std::vector<void *> workspace;
    uint64_t sum = 0;
    for (auto bytes : prepared.bytes)
        sum = Add(sum, bytes);
    MemoryGate(sum, "Diagnostic workspace budget exceeded");
    for (auto bytes : prepared.bytes)
        workspace.push_back(extra.Allocate(bytes));
    Upload();
    Execute(prepared, workspace);
    Download();
    Require(aclrtSynchronizeStream(resources.runtime.stream), "large full diagnostic completion");
    auto replay = Receive(0);
    Check(Same(normal.ids, replay.ids) && Same(normal.scores, replay.scores),
          "Normal graph and diagnostic outputs differ");
    std::vector<std::vector<float>> scores(lists.size());
    std::vector<std::vector<int32_t>> ids(lists.size());
    for (size_t i = 0; i < lists.size(); ++i) {
        const auto count = plan.q * plan.lists[i] * plan.k;
        scores[i].resize(count);
        ids[i].resize(count);
        Require(
            aclrtMemcpy(scores[i].data(), Bytes(count, 4), lists[i].scores, Bytes(count, 4), ACL_MEMCPY_DEVICE_TO_HOST),
            "read diagnostic level scores");
        Require(aclrtMemcpy(ids[i].data(), Bytes(count, 4), lists[i].ids, Bytes(count, 4), ACL_MEMCPY_DEVICE_TO_HOST),
                "read diagnostic level IDs");
        Write(path / ("level_" + std::to_string(i) + "_scores.bin"), scores[i]);
        Write(path / ("level_" + std::to_string(i) + "_ids.bin"), ids[i]);
    }
    Check(fs::file_size(reference) == Bytes(plan.q * plan.docs, 4),
          "Independent original-document reference size mismatch");
    std::ifstream expected(reference, std::ios::binary);
    std::ofstream actual(path / "scores.bin", std::ios::binary);
    double maxError = 0;
    auto compareList = [&](size_t level, uint64_t query, uint64_t list, const std::vector<Candidate> &wanted) {
        Check(wanted.size() == plan.k, "Insufficient independent candidates");
        for (size_t rank = 0; rank < plan.k; ++rank) {
            const size_t pos = (query * plan.lists[level] + list) * plan.k + rank;
            Check(ids[level][pos] == wanted[rank].id && std::memcmp(&scores[level][pos], &wanted[rank].score, 4) == 0,
                  "Leaf/tree exact IDs or score bits differ");
        }
    };
    for (uint64_t query = 0; query < plan.q; ++query) {
        std::array<Candidate, 32> gold{};
        int64_t goldCount = 0;
        for (uint64_t leaf = 0; leaf < plan.leaves; ++leaf) {
            std::vector<float> row(Leaf);
            std::vector<float> ref(Leaf);
            const auto first = leaf * Leaf;
            const auto valid = first < plan.docs ? std::min(Leaf, plan.docs - first) : 0;
            Require(aclrtMemcpy(row.data(), Bytes(Leaf, 4), base + 64 + Bytes(query * plan.padded + first, 4),
                                Bytes(Leaf, 4), ACL_MEMCPY_DEVICE_TO_HOST),
                    "read bounded score shard");
            if (valid) {
                Check(static_cast<bool>(expected.read(reinterpret_cast<char *>(ref.data()), Bytes(valid, 4))),
                      "Reference read failed");
                actual.write(reinterpret_cast<const char *>(row.data()), Bytes(valid, 4));
            }
            std::array<Candidate, 32> heap{};
            int64_t count = 0;
            for (uint64_t d = 0; d < Leaf; ++d) {
                Check(std::isfinite(row[d]) && row[d] >= 0, "Invalid actual full score");
                if (d < valid) {
                    const double error = std::abs(static_cast<double>(row[d]) - ref[d]);
                    Check(std::isfinite(ref[d]) && ref[d] >= 0 && error <= 1e-6 + 2e-6 * std::abs(ref[d]),
                          "Ordered FP32 formula mismatch");
                    maxError = std::max(maxError, error);
                    KeepCandidate(gold, goldCount, plan.k, {ref[d], static_cast<int32_t>(first + d)});
                } else
                    Check(row[d] == 0, "Padded articles must have zero score");
                KeepCandidate(heap, count, plan.k, {row[d], static_cast<int32_t>(first + d)});
            }
            compareList(0, query, leaf, Ordered(heap, count));
        }
        const auto ordered = Ordered(gold, goldCount);
        for (size_t rank = 0; rank < plan.k; ++rank)
            Check(normal.ids[query * plan.k + rank] == ordered[rank].id,
                  "Independent original-document strict final ranking differs");
    }
    for (size_t level = 1; level < lists.size(); ++level)
        for (uint64_t query = 0; query < plan.q; ++query)
            for (uint64_t group = 0; group < plan.lists[level]; ++group) {
                std::array<Candidate, 32> heap{};
                int64_t count = 0;
                for (uint64_t child = group * FanIn; child < std::min((group + 1) * FanIn, plan.lists[level - 1]);
                     ++child)
                    for (size_t rank = 0; rank < plan.k; ++rank) {
                        const size_t pos = (query * plan.lists[level - 1] + child) * plan.k + rank;
                        KeepCandidate(heap, count, plan.k, {scores[level - 1][pos], ids[level - 1][pos]});
                    }
                compareList(level, query, group, Ordered(heap, count));
            }
    actual.close();
    Check(bool(actual), "Cannot save diagnostic scores");
    std::array<unsigned char, 64> guard{};
    for (size_t pos : {size_t(0), fullBytes + 64}) {
        Require(aclrtMemcpy(guard.data(), 64, base + pos, 64, ACL_MEMCPY_DEVICE_TO_HOST), "read full diagnostic guard");
        Check(std::all_of(guard.begin(), guard.end(), [](unsigned char value) { return value == 0xff; }),
              "Full diagnostic guard overwritten");
    }
    resources.CheckGuards();
    Check(t.Close() && extra.Close(), "Diagnostic cleanup failed");
    std::ofstream proof(path / "tree_check.json");
    proof << "{\"status\":\"all_scores_all_leaf_lists_all_merge_levels_and_strict_gold_ids_passed\",\"max_abs_"
             "error\":"
          << maxError << ",\"levels\":" << lists.size() << "}\n";
    proof.close();
    Check(bool(proof), "Cannot save tree checks");
}

void Session::PoisonForDiagnostic()
{
    resources.CheckGuards();
    for (size_t i = 0; i < lists.size(); ++i) {
        std::vector<uint32_t> poison(plan.q * plan.lists[i] * plan.k, UINT32_MAX);
        for (void *p : {lists[i].scores, lists[i].ids})
            Require(aclrtMemcpy(p, Bytes(poison.size(), 4), poison.data(), Bytes(poison.size(), 4),
                                ACL_MEMCPY_HOST_TO_DEVICE),
                    "poison tree level");
    }
}
void Session::VerifyIndex()
{
    resources.runtime.RequireCurrentContext();
    std::vector<unsigned char> chunk(8 * 1024 * 1024);
    for (size_t i = 0; i < 5; ++i)
        for (size_t pos = 0; pos < indexBytes[i];) {
            const auto bytes = std::min(chunk.size(), indexBytes[i] - pos);
            Require(aclrtMemcpy(chunk.data(), bytes, static_cast<unsigned char *>(indexMemory[i]) + pos, bytes,
                                ACL_MEMCPY_DEVICE_TO_HOST),
                    "streaming resident index readback");
            Check(std::memcmp(chunk.data(), static_cast<const unsigned char *>(indexHost[i]) + pos, bytes) == 0,
                  "Index/directory changed");
            pos += bytes;
        }
    resources.CheckGuards();
}
} // namespace bm25large
