#ifndef BM25_ASCENDHOST_SRC_BM25_PLANNED_SEARCH_H
#define BM25_ASCENDHOST_SRC_BM25_PLANNED_SEARCH_H
#include "acl/acl_rt.h"
#include "ascendhost/src/bm25/fused_runtime.h"
#include "ascendhost/src/bm25/fused_api.h"
#include "common/bm25/fixed_query_packet.h"
#include "common/bm25/shard_directory.h"
#include <array>
#include <chrono>

namespace bm25search {
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

using Clock = std::chrono::steady_clock;
struct PlannedResult {
    std::vector<float> scores;
    std::vector<int32_t> ids;
    double searchMs = 0;
    size_t logicalQueryBytes = 0, queryH2DBytes = 0, topkD2HBytes = 0;
};

// Coordinates graph lifetime with ALL referenced resources, including constructor-failure cleanup.
struct PlanResources {
    PlanResources() = default;
    PlanResources(const PlanResources &) = delete;
    PlanResources &operator=(const PlanResources &) = delete;
    PlanResources(PlanResources &&) = delete;
    PlanResources &operator=(PlanResources &&) = delete;
    Runtime runtime;
    Buffers buffers;
    Tensors tensors;
    std::vector<void *> pinned;
    struct Guard {
        unsigned char *allocation;
        size_t size;
    };
    std::vector<Guard> guards;
    aclmdlRI model{};
    bool capturing = false, modelValid = false, closed = false;
    void *Host(size_t bytes)
    {
        void *p = nullptr;
        Require(aclrtMallocHost(&p, bytes), "allocate pinned plan arena");
        Check(p != nullptr, "Null pinned plan arena");
        pinned.push_back(p);
        return p;
    }
    void *Device(size_t bytes)
    {
        auto *base = static_cast<unsigned char *>(buffers.Allocate(bytes + 128));
        std::array<unsigned char, 64> guard{};
        guard.fill(0xa7);
        Require(aclrtMemcpy(base, 64, guard.data(), 64, ACL_MEMCPY_HOST_TO_DEVICE), "initialize leading plan guard");
        Require(aclrtMemcpy(base + 64 + bytes, 64, guard.data(), 64, ACL_MEMCPY_HOST_TO_DEVICE),
                "initialize trailing plan guard");
        guards.push_back({base, bytes});
        return base + 64;
    }
    void CheckGuards() const;
    bool Close();
    ~PlanResources();
};

} // namespace bm25search
#endif // BM25_ASCENDHOST_SRC_BM25_PLANNED_SEARCH_H
