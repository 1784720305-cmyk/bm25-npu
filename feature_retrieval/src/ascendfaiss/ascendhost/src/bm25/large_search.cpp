#include "ascendhost/src/bm25/large_search.h"
#include "acl/acl_base_rt.h"
#include "acl/acl_rt.h"
#include "aclnn/acl_meta.h"
#include <cstdint>
#include <cstddef>
#include <vector>
#include <cstring>
#include <cmath>
#include <array>
#include <algorithm>
#include <filesystem>
#include "ascendhost/include/impl/Bm25RuntimeBinding.h"
#include "common/bm25/shard_directory.h"
#include "common/bm25/large_search_plan.h"
#include "common/bm25/error.h"
#include <chrono>
#include <ratio>
namespace bm25large {
Prepared Session::Prepare(aclTensor *debug, int64_t mode)
{
    Prepared p;
    p.executors.resize(lists.size());
    p.bytes.resize(lists.size());
    Require(api.prepareFused(indexTensors[0], indexTensors[1], indexTensors[2], indexTensors[3], indexTensors[4],
                             offsets, words, plan.k, plan.leaves, mode, lists[0].s, lists[0].i, debug, &p.bytes[0],
                             &p.executors[0]),
            "prepare bounded leaves");
    for (size_t i = 1; i < lists.size(); ++i)
        Require(api.prepareMerge(lists[i - 1].s, lists[i - 1].i, plan.k, plan.padded, lists[i].s, lists[i].i,
                                 &p.bytes[i], &p.executors[i]),
                "prepare merge tree level");
    for (size_t i = 0; i < p.bytes.size(); ++i)
        Check(p.executors[i] && p.bytes[i] <= SIZE_MAX, "Invalid tree executor/workspace");
    return p;
}

void Session::Execute(Prepared &p, const std::vector<void *> &workspace)
{
    Require(api.executeFused(workspace[0], p.bytes[0], p.executors[0], resources.runtime.stream),
            "execute bounded leaves");
    p.executors[0] = nullptr;
    for (size_t i = 1; i < lists.size(); ++i) {
        Require(api.executeMerge(workspace[i], p.bytes[i], p.executors[i], resources.runtime.stream),
                "execute merge level");
        p.executors[i] = nullptr;
    }
}

void Session::Upload()
{
    Require(aclrtMemcpyAsync(input, packet.inputBytes, hostInput, packet.inputBytes, ACL_MEMCPY_HOST_TO_DEVICE,
                             resources.runtime.stream),
            "large query H2D");
}

void Session::Download()
{
    Require(aclrtMemcpyAsync(hostOutput, packet.outputBytes, output, packet.outputBytes, ACL_MEMCPY_DEVICE_TO_HOST,
                             resources.runtime.stream),
            "large topK D2H");
}

PlannedResult Session::Receive(size_t logical)
{
    PlannedResult result;
    result.scores.resize(plan.q * plan.k);
    result.ids.resize(result.scores.size());
    std::memcpy(result.scores.data(), hostOutput, Bytes(result.scores.size(), 4));
    std::memcpy(result.ids.data(), static_cast<unsigned char *>(hostOutput) + packet.idOffset,
                Bytes(result.ids.size(), 4));
    for (size_t i = 0; i < result.ids.size(); ++i) {
        Check(result.ids[i] >= 0 && static_cast<uint64_t>(result.ids[i]) < plan.docs &&
                  std::isfinite(result.scores[i]) && result.scores[i] >= 0,
              "Invalid real-document result");
        if (i % plan.k)
            Check(result.scores[i - 1] > result.scores[i] ||
                      (result.scores[i - 1] == result.scores[i] && result.ids[i - 1] < result.ids[i]),
                  "Strict global ranking failure");
    }
    result.logicalQueryBytes = logical;
    result.queryH2DBytes = packet.inputBytes;
    result.topkD2HBytes = packet.outputBytes;
    return result;
}

void Session::MemoryGate(uint64_t bytes, const char *why)
{
    size_t free = 0;
    size_t total = 0;
    Require(aclrtGetMemInfo(ACL_HBM_MEM, &free, &total), "query actual free device memory");
    Check(total > 0 && bytes <= static_cast<uint64_t>(free) * 4 / 5, why);
}

std::vector<Candidate> Session::Ordered(std::array<Candidate, 32> &heap, int64_t count)
{
    std::sort(heap.begin(), heap.begin() + count, Better{});
    return {heap.begin(), heap.begin() + count};
}

Session::Session(const Index &source, const fs::path &vendors, int device, uint64_t q, uint64_t k,
                 uint64_t wordsCapacity)
    : Session(source, vendors, device, q, k, wordsCapacity, {})
{
}

Session::Session(const Index &source, const fs::path &vendors, int device, uint64_t q, uint64_t k,
                 uint64_t wordsCapacity, const ascend::bm25::RuntimeBinding &binding)
    : index(source), plan(source.norm.size(), source.idf.size(), source.docs.size(), q, k, wordsCapacity), api(vendors),
      packet(q, wordsCapacity, k)
{
    auto &rt = resources.runtime;
    rt.device = device;
    rt.Initialize(device, binding);
    resources.buffers.stream = resources.tensors.stream = rt.stream;
    size_t free = 0;
    size_t total = 0;
    Require(aclrtGetMemInfo(ACL_HBM_MEM, &free, &total), "initial actual memory");
    initialFreeBytes = free;
    totalDeviceBytes = total;
    MemoryGate(plan.knownDeviceBytes, "Index/tree/arenas exceed80% of actual free memory");
    paddedNorm = index.norm;
    paddedNorm.resize(plan.padded, 1.0F);
    directory = bm25plan::BuildDirectory(index.offsets, index.docs, plan.padded, plan.leaves);
    indexHost[0] = directory.data();
    indexHost[1] = index.docs.data();
    indexHost[2] = index.tf.data();
    indexHost[3] = paddedNorm.data();
    indexHost[4] = index.idf.data();
    const uint64_t counts[5] = {directory.size(), index.docs.size(), index.tf.size(), paddedNorm.size(),
                                index.idf.size()};
    const aclDataType types[5] = {ACL_INT64, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT};
    auto &t = resources.tensors;
    for (size_t i = 0; i < 5; ++i) {
        indexBytes[i] = Bytes(counts[i], i ? 4 : 8);
        indexUploadBytes += indexBytes[i];
        auto *memory = resources.Device(indexBytes[i]);
        indexTensors[i] = t.Describe1D(counts[i], types[i], memory);
        Require(aclrtMemcpy(memory, indexBytes[i], indexHost[i], indexBytes[i], ACL_MEMCPY_HOST_TO_DEVICE),
                "one-time large index upload");
        indexMemory[i] = memory;
    }
    hostInput = resources.Host(packet.inputBytes);
    hostOutput = resources.Host(packet.outputBytes);
    input = resources.Device(packet.inputBytes);
    output = resources.Device(packet.outputBytes);
    placeholder = resources.Device(4);
    offsets = t.Describe1D(q + 1, ACL_INT64, input);
    words = t.Describe1D(wordsCapacity ? wordsCapacity : 1, ACL_INT32,
                         static_cast<unsigned char *>(input) + packet.wordOffset);
    placeholderTensor = t.Describe1D(1, ACL_FLOAT, placeholder);
    for (size_t i = 0; i < plan.lists.size(); ++i) {
        Lists x;
        const auto count = plan.q * plan.lists[i] * plan.k;
        if (i + 1 == plan.lists.size()) {
            x.scores = output;
            x.ids = static_cast<unsigned char *>(output) + packet.idOffset;
        } else {
            x.scores = resources.Device(Bytes(count, 4));
            x.ids = resources.Device(Bytes(count, 4));
        }
        x.s = Describe3D(t, q, plan.lists[i], k, ACL_FLOAT, x.scores);
        x.i = Describe3D(t, q, plan.lists[i], k, ACL_INT32, x.ids);
        lists.push_back(x);
    }
    const Queries empty(q);
    packet.Pack(empty, hostInput);
    auto prepared = Prepare(placeholderTensor, 0);
    workspaceBytes = prepared.bytes;
    uint64_t sum = 0;
    for (auto bytes : prepared.bytes)
        sum = Add(sum, bytes);
    MemoryGate(sum, "Actual tree workspace exceeds memory budget");
    for (auto bytes : prepared.bytes)
        work.push_back(resources.buffers.Allocate(bytes));
    Require(aclmdlRICaptureBegin(rt.stream, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL), "begin large graph capture");
    resources.capturing = true;
    Upload();
    Execute(prepared, work);
    Download();
    Require(aclmdlRICaptureEnd(rt.stream, &resources.model), "end large graph capture");
    resources.capturing = false;
    resources.modelValid = true;
    auto bootstrap = Search(empty);
    for (size_t i = 0; i < bootstrap.ids.size(); ++i)
        Check(bootstrap.scores[i] == 0 && bootstrap.ids[i] == static_cast<int32_t>(i % k),
              "Padded merge empty bootstrap failed");
}

PlannedResult Session::Search(const Queries &queries)
{
    const auto start = Clock::now();
    if (closed || failed) {
        throw ascend::bm25::InvalidState("Large session closed/failed");
    }
    resources.runtime.RequireCurrentContext();
    // Packing validates the complete request before writing the reusable input arena.
    // A rejected request has submitted no work and must not invalidate the session.
    const auto logical = packet.Pack(queries, hostInput);
    try {
        ++normalSearchCalls;
        Require(aclmdlRIExecuteAsync(resources.model, resources.runtime.stream), "replay bounded merge tree");
        ++replays;
        Require(aclrtSynchronizeStream(resources.runtime.stream), "complete Search through result D2H");
        auto result = Receive(logical);
        result.searchMs = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        return result;
    } catch (...) {
        failed = true;
        throw;
    }
}

void Session::Close()
{
    if (!closed) {
        resources.runtime.RequireCurrentContext();
        Check(resources.Close(), "Large plan cleanup failed");
        closed = true;
    }
}

} // namespace bm25large
