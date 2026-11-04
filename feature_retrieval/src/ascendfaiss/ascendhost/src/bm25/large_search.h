#ifndef BM25_ASCENDHOST_SRC_BM25_LARGE_SEARCH_H
#define BM25_ASCENDHOST_SRC_BM25_LARGE_SEARCH_H
#include "ascendhost/src/bm25/planned_search.h"
#include "common/bm25/large_search_plan.h"
#include "common/bm25/topk_heap.h" // untimed independent bounded-heap checks only
namespace bm25large {
using ascend::bm25::Candidate;
using ascend::bm25::Better;
using ascend::bm25::KeepCandidate;
using ascend::bm25::Buffers;
using ascend::bm25::Bytes;
using ascend::bm25::Check;
using ascend::bm25::Describe3D;
using ascend::bm25::Index;
using ascend::bm25::Require;
using ascend::bm25::Runtime;
using ascend::bm25::Same;
using ascend::bm25::SearchFusedApi;
using ascend::bm25::Tensors;
namespace fs = std::filesystem;

using bm25search::Queries;
using bm25search::PlannedResult;
using bm25search::Clock;
struct Lists {
    void *scores{}, *ids{};
    aclTensor *s{}, *i{};
};
struct Prepared {
    std::vector<aclOpExecutor *> executors;
    std::vector<uint64_t> bytes;
};
class SessionDiagnostics;
class Session {
    friend class SessionDiagnostics;
    void *indexMemory[5]{};
    uint64_t indexUploadBytes = 0, replays = 0, normalSearchCalls = 0, initialFreeBytes = 0, totalDeviceBytes = 0;
    std::vector<uint64_t> workspaceBytes;
    const Index &index;
    Plan plan;
    SearchFusedApi api;
    bm25search::PlanResources resources;
    bm25plan::PacketLayout packet;
    std::vector<int64_t> directory;
    std::vector<float> paddedNorm;
    const void *indexHost[5]{};
    size_t indexBytes[5]{};
    aclTensor *indexTensors[5]{};
    aclTensor *offsets{}, *words{}, *placeholderTensor{};
    void *input{}, *output{}, *hostInput{}, *hostOutput{}, *placeholder{};
    std::vector<Lists> lists;
    std::vector<void *> work;
    bool failed = false, closed = false;
    Prepared Prepare(aclTensor *debug, int64_t mode);
    void Execute(Prepared &p, const std::vector<void *> &workspace);
    void Upload();
    void Download();
    PlannedResult Receive(size_t logical);
    void MemoryGate(uint64_t bytes, const char *why);
    static std::vector<Candidate> Ordered(std::array<Candidate, 32> &heap, int64_t count);

    void PoisonForDiagnostic();
    void SaveDiagnostic(const Queries &queries, const PlannedResult &normal, const fs::path &path,
                        const fs::path &reference);
    void VerifyIndex();

public:
    ~Session() = default;
    Session(const Session &) = delete;
    Session &operator=(const Session &) = delete;
    Session(Session &&) = delete;
    Session &operator=(Session &&) = delete;
    Session(const Index &source, const fs::path &vendors, int device, uint64_t q, uint64_t k, uint64_t wordsCapacity);
    Session(const Index &source, const fs::path &vendors, int device, uint64_t q, uint64_t k, uint64_t wordsCapacity,
            const ascend::bm25::RuntimeBinding &binding);
    PlannedResult Search(const Queries &queries);
    const Plan &GetPlan() const { return plan; }
    uint64_t GetIndexUploadBytes() const { return indexUploadBytes; }
    uint64_t GetReplays() const { return replays; }
    uint64_t GetNormalSearchCalls() const { return normalSearchCalls; }
    uint64_t GetInitialFreeBytes() const { return initialFreeBytes; }
    uint64_t GetTotalDeviceBytes() const { return totalDeviceBytes; }
    const std::vector<uint64_t> &GetWorkspaceBytes() const { return workspaceBytes; }
    void Close();
};
} // namespace bm25large

#endif // BM25_ASCENDHOST_SRC_BM25_LARGE_SEARCH_H
