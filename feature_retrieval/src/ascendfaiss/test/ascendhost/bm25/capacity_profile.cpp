// Separate diagnostic executable/package. Device timestamps never enter normal Search timing.
#include "ascendhost/src/bm25/large_search.h"
#include "test/ascendhost/bm25/input_io.h"
#include "common/bm25/validation.h"
#include "acl/acl_base_rt.h"
#include "acl/acl_rt.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace bm25large {
// This executable links bm25_host without search_diagnostic.cpp or its friend definition.
class SessionDiagnostics {
public:
    static void Profile(Session &s, const std::array<Queries, 2> &queries, const fs::path &input,
                        const fs::path &output)
    {
        const size_t tasks = s.plan.q * s.plan.leaves;
        constexpr size_t traceWords = 8; // One independent 64-byte record per task.
        const size_t traceBytes = Bytes(tasks * traceWords, sizeof(int64_t));
        const size_t fullBytes = Bytes(s.plan.q * s.plan.padded, sizeof(float));
        Check(traceBytes <= fullBytes, "Trace shape too small");
        s.MemoryGate(fullBytes + 128, "Profile trace exceeds budget");
        Buffers buffers;
        Tensors tensors;
        buffers.stream = tensors.stream = s.resources.runtime.stream;
        auto *base = static_cast<unsigned char *>(buffers.Allocate(fullBytes + 128));
        Check(reinterpret_cast<uintptr_t>(base + 64) % 64 == 0, "Trace base must be 64-byte aligned");
        auto *trace = tensors.Describe(s.plan.q, s.plan.padded, ACL_FLOAT, base + 64);
        std::array<PlannedResult, 2> gold;
        for (size_t which = 0; which < 2; ++which) {
            gold[which] = s.Search(queries[which]);
            const auto wanted = ascend::bm25::Read<int32_t>(input / (which ? "B" : "A") / "expected_top_ids.bin");
            Check(Same(gold[which].ids, wanted), "Profile package normal Search oracle IDs differ");
            const auto prefix = output.string() + (which ? ".gold_B_" : ".gold_A_");
            ascend::bm25::Write(prefix + "scores.bin", gold[which].scores);
            ascend::bm25::Write(prefix + "ids.bin", gold[which].ids);
        }
        std::vector<void *> work;
        std::vector<uint64_t> workBytes;
        std::ofstream out(output);
        out << "{\"status\":\"device_stage_profile_traces_and_bit_checks_passed\",\"tasks\":" << tasks
            << ",\"queries\":" << s.plan.q
            << ",\"trace_contract\":\"simt_gm_cacheline64_nonce_v2\",\"trace_words\":8,\"samples\":[";
        bool comma = false;
        for (int cycle = -5; cycle < 30; ++cycle)
            for (int slot = 0; slot < 2; ++slot) {
                const int which = slot ^ ((cycle + 6) & 1);
                s.packet.Pack(queries[which], s.hostInput);
                const std::array<unsigned char, 64> guard = []() {
                    std::array<unsigned char, 64> result{};
                    result.fill(0xff);
                    return result;
                }();
                for (size_t pos : {size_t(0), fullBytes + 64})
                    Require(aclrtMemcpy(base + pos, 64, guard.data(), 64, ACL_MEMCPY_HOST_TO_DEVICE), "trace guard");
                const int64_t sequence = (cycle + 5) * 2 + slot + 1;
                std::vector<int64_t> values(tasks * traceWords, -1);
                for (size_t task = 0; task < tasks; ++task)
                    values[task * traceWords + 4] = sequence;
                Require(aclrtMemcpy(base + 64, traceBytes, values.data(), traceBytes, ACL_MEMCPY_HOST_TO_DEVICE),
                        "poison trace");
                auto prepared = s.Prepare(trace, 3);
                if (work.empty()) {
                    workBytes = prepared.bytes;
                    uint64_t total = 0;
                    for (auto bytes : workBytes) {
                        Check(bytes <= UINT64_MAX - total, "Profile workspace sum overflow");
                        total += bytes;
                    }
                    s.MemoryGate(total, "Profile workspace exceeds budget");
                    for (auto bytes : workBytes)
                        work.push_back(buffers.Allocate(bytes));
                }
                Check(prepared.bytes == workBytes, "Profile workspace changed");
                s.Upload();
                s.Execute(prepared, work);
                s.Download();
                Require(aclrtSynchronizeStream(s.resources.runtime.stream), "profile completion");
                const auto actual = s.Receive(0);
                Check(Same(actual.ids, gold[which].ids) && Same(actual.scores, gold[which].scores),
                      "Instrumented/normal score bits or IDs differ");
                Require(aclrtMemcpy(values.data(), traceBytes, base + 64, traceBytes, ACL_MEMCPY_DEVICE_TO_HOST),
                        "profile readback");
                for (size_t pos : {size_t(0), fullBytes + 64}) {
                    std::array<unsigned char, 64> actualGuard{};
                    Require(aclrtMemcpy(actualGuard.data(), 64, base + pos, 64, ACL_MEMCPY_DEVICE_TO_HOST),
                            "trace guard readback");
                    Check(actualGuard == guard, "Trace guard overwritten");
                }
                s.resources.CheckGuards();
                for (size_t task = 0; task < tasks; ++task) {
                    const auto *v = values.data() + task * traceWords;
                    const bool valid = v[0] >= 0 && v[1] > v[0] && v[2] > v[1] &&
                                       (v[3] == 1024 || v[3] == 2048 || v[3] == 4096) && v[4] == sequence &&
                                       v[5] == sequence && v[6] == static_cast<int64_t>(task) && v[7] == 2;
                    if (!valid) {
                        // Preserve the complete failing record before throwing; never relax the timing gate.
                        const auto context = "cycle=" + std::to_string(cycle) + " case=" + (which ? "B" : "A") +
                                             " task=" + std::to_string(task) + " sequence=" + std::to_string(sequence);
                        std::ofstream failure(output.string() + ".failure.json");
                        failure << "{\"context\":\"" << context << "\",\"record\":[";
                        for (size_t i = 0; i < traceWords; ++i)
                            failure << (i ? "," : "") << v[i];
                        failure << "]}\n";
                        failure.close();
                        Check(false, "Invalid device trace: " + context + "; see " + output.string() + ".failure.json");
                    }
                }
                if (cycle >= 0) {
                    out << (comma ? "," : "") << "{\"case\":\"" << (which ? "B" : "A") << "\",\"cycle\":" << cycle
                        << ",\"sequence\":" << sequence << ",\"task_timestamps\":[";
                    for (size_t i = 0; i < values.size(); ++i)
                        out << (i ? "," : "") << values[i];
                    out << "]}";
                    comma = true;
                }
            }
        out << "],\"completed\":true,\"normal_profile_outputs_bit_identical\":true,\"guards_passed\":true}\n";
        out.close();
        Check(bool(out), "Cannot save device profile");
        Check(tensors.Close() && buffers.Close(), "Profile cleanup failed");
    }
};
} // namespace bm25large

int main(int argc, char **argv)
{
    try {
        using ascend::bm25::Check;
        Check(argc == 5, "Usage: capacity_profile INPUT VENDORS DEVICE NEW_OUTPUT_JSON");
        const auto input = std::filesystem::absolute(argv[1]);
        Check(!std::filesystem::exists(argv[4]), "Profile output exists");
        std::array<bm25large::Queries, 2> queries{bm25search::LoadQueries((input / "A.txt").string()),
                                                  bm25search::LoadQueries((input / "B.txt").string())};
        Check(queries[0].size() == 8 && queries[1].size() == 8, "Fixed Q8 required");
        size_t words = 0;
        for (const auto &batch : queries) {
            size_t count = 0;
            for (const auto &q : batch)
                count += q.size();
            words = std::max(words, count);
        }
        const auto index = ascend::bm25::LoadIndex(input / "index");
        bm25large::Session session(index, std::filesystem::absolute(argv[2]),
                                   static_cast<int>(ascend::bm25::Number(argv[3], 0, INT32_MAX)), 8, 10, words);
        bm25large::SessionDiagnostics::Profile(session, queries, input, std::filesystem::absolute(argv[4]));
        session.Close();
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAILED: " << error.what() << std::endl;
        return 1;
    }
}
