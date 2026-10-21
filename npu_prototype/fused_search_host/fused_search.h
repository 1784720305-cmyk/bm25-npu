#pragma once
#include "../fused_chain_host/fused_runtime.h"
#include "fused_api.h"
#include "../search_host/search_inputs.h"
#include <chrono>

namespace bm25search {
using Clock = std::chrono::steady_clock;
struct Result {
    std::vector<float> scores;
    std::vector<int32_t> ids;
    double searchMs = 0;
    size_t logicalQueryBytes = 0, queryH2DBytes = 0;
};

// Fused Search adapter; frozen operator/128 threads/S32. Normal output[0] is ONE FP32.
// Fixed-capacity in-process session. Constructor owns cold init, index upload and bootstrap.
// Every successful Search receives raw grouped IDs, performs its own packing, and returns
// ONLY CPU-available final Top-K. Diagnostics are separate methods called after Search.
class NpuSearchSession {
    const Index &index;
    SearchFusedApi api;
    Runtime runtime;
    Buffers buffers;
    Tensors tensors;
    size_t maxQ, maxWords;
    int64_t k, shards, docs;

    void *indexMemory[5]{}; aclTensor *indexTensors[5]{};
    const void *indexHost[5]{}; size_t indexBytes[5]{};
    void *offsetMemory = nullptr, *wordMemory = nullptr, *output[5]{};
    size_t capacityBytes[5]{};
    void *workspace[2][2]{}; uint64_t capacityWorkspace[2][2]{}; bool prepared[2]{};
    bool failed = false, closed = false;

    Result SearchImpl(const Queries &queries,Clock::time_point start,void *debugMemory=nullptr)
    {
        Check(!closed && !failed,"Search session is closed/failed");
        for (const auto &row:queries) Check(row.size()<=64,"Fused Search exceeds 64 words per query");
        auto packed = PackQueries(queries,maxQ,maxWords);
        const size_t mode=debugMemory?1:0;
        const int64_t q = static_cast<int64_t>(queries.size());
        Tensors current; current.stream = runtime.stream;
        auto *offsets = current.Describe1D(packed.offsets.size(),ACL_INT64,offsetMemory);
        auto *words = current.Describe1D(std::max<size_t>(1,packed.words.size()),ACL_INT32,wordMemory);
        auto *debugScores = debugMemory ? current.Describe(q,docs,ACL_FLOAT,debugMemory) : current.Describe1D(1,ACL_FLOAT,output[0]);
        auto *mergeScores = Describe3D(current,q,shards,k,ACL_FLOAT,output[1]);
        auto *mergeIds = Describe3D(current,q,shards,k,ACL_INT32,output[2]);
        auto *topScores = current.Describe(q,k,ACL_FLOAT,output[3]);
        auto *topIds = current.Describe(q,k,ACL_INT32,output[4]);
        const int32_t dummy = 0;
        const size_t offsetBytes = Bytes(packed.offsets.size(),8);
        const size_t wordBytes = Bytes(std::max<size_t>(1,packed.words.size()),4);
        Require(aclrtMemcpy(offsetMemory,Bytes(maxQ+1,8),packed.offsets.data(),offsetBytes,ACL_MEMCPY_HOST_TO_DEVICE),"search upload offsets");
        const void *wordHost = packed.words.empty()?static_cast<const void *>(&dummy):packed.words.data();
        Require(aclrtMemcpy(wordMemory,Bytes(std::max<size_t>(1,maxWords),4),wordHost,wordBytes,ACL_MEMCPY_HOST_TO_DEVICE),"search upload words");
        aclOpExecutor *executors[2]{}; uint64_t sizes[2]{};
        Require(api.prepareFused(indexTensors[0],indexTensors[1],indexTensors[2],indexTensors[3],indexTensors[4],offsets,words,
            k,shards,static_cast<int64_t>(mode),mergeScores,mergeIds,debugScores,&sizes[0],&executors[0]),"search fused prepare");
        Require(api.prepareMerge(mergeScores,mergeIds,k,docs,topScores,topIds,&sizes[1],&executors[1]),"search merge prepare");
        for (size_t i=0;i<2;++i) {
            Check(executors[i] && sizes[i]<=std::numeric_limits<size_t>::max(),"Invalid fused executor/workspace");
            if (!prepared[mode]) { capacityWorkspace[mode][i]=sizes[i]; workspace[mode][i]=buffers.Allocate(static_cast<size_t>(sizes[i])); }
            else Check(sizes[i]<=capacityWorkspace[mode][i],"Workspace exceeds cold allocation; no hidden resize");
        }
        prepared[mode]=true;
        Require(api.executeFused(workspace[mode][0],sizes[0],executors[0],runtime.stream),"search fused execute"); executors[0]=nullptr;
        Require(api.executeMerge(workspace[mode][1],sizes[1],executors[1],runtime.stream),"search merge execute"); executors[1]=nullptr;
        Require(aclrtSynchronizeStream(runtime.stream),"search complete chain synchronize");
        // Successful chain sync proves no Tensor-dependent work remains. Avoid a redundant
        // cleanup sync; on earlier exceptions current.stream retains the safe RAII fallback.
        current.stream = nullptr;
        Result result;
        result.scores.resize(static_cast<size_t>(q)*k); result.ids.resize(result.scores.size());
        const size_t bytes = Bytes(result.scores.size(),4);
        Require(aclrtMemcpy(result.scores.data(),bytes,output[3],bytes,ACL_MEMCPY_DEVICE_TO_HOST),"search download final scores");
        Require(aclrtMemcpy(result.ids.data(),bytes,output[4],bytes,ACL_MEMCPY_DEVICE_TO_HOST),"search download final IDs");
        for (size_t i = 0; i < result.ids.size(); ++i) {
            Check(result.ids[i] >= 0 && result.ids[i] < docs && std::isfinite(result.scores[i]) && result.scores[i] >= 0,
                  "Invalid CPU-available Top-K result");
            if (i%static_cast<size_t>(k)) Check(result.scores[i-1] > result.scores[i] ||
                (result.scores[i-1] == result.scores[i] && result.ids[i-1] < result.ids[i]),"Returned ranking order/tie rule invalid");
        }
        Check(current.Close(),"Search tensor cleanup failed");
        { std::vector<aclTensor *> release; release.swap(current.items); }
        result.logicalQueryBytes = offsetBytes+Bytes(packed.words.size(),4);
        result.queryH2DBytes = offsetBytes+wordBytes;
        // Include destruction of request packing buffers before the stop timestamp.
        { PackedQueries release; release.offsets.swap(packed.offsets); release.words.swap(packed.words); }
        result.searchMs = std::chrono::duration<double,std::milli>(Clock::now()-start).count();
        return result; // Vectors already hold the final IDs and original FP32 scores on CPU.
    }
public:
    uint64_t indexUploadBytes = 0, debugPoisonBytesPerCall = 0;
    NpuSearchSession(const Index &source,const fs::path &vendors,int device,int64_t selectedK,int64_t selectedShards,
                     size_t queryCapacity,size_t wordCapacity)
        : index(source),api(vendors),maxQ(queryCapacity),maxWords(wordCapacity),k(selectedK),shards(selectedShards),
          docs(static_cast<int64_t>(index.norm.size()))
    {
        Check(maxQ >= 1 && maxQ <= 10000 && maxWords <= 16777216 &&
              static_cast<uint64_t>(maxQ) <= static_cast<uint64_t>(INT32_MAX)/docs &&
              k >= 1 && k <= 32 && shards >= 1 && shards <= 32 && docs%shards == 0 && docs/shards >= k && docs/shards<=8192,
              "Invalid fixed search capacity/dimensions");
        runtime.device = device;
        Require(aclInit(nullptr),"aclInit"); runtime.initialized = true;
        Require(aclrtSetDevice(device),"aclrtSetDevice"); runtime.selected = true;
        Require(aclrtCreateContext(&runtime.context,device),"aclrtCreateContext");
        aclrtRunMode mode; Require(aclrtGetRunMode(&mode),"aclrtGetRunMode"); Check(mode == ACL_HOST,"Requires ACL_HOST");
        Require(aclrtCreateStream(&runtime.stream),"aclrtCreateStream");
        buffers.stream = runtime.stream; tensors.stream = runtime.stream;
        indexHost[0]=index.offsets.data(); indexHost[1]=index.docs.data(); indexHost[2]=index.tf.data();
        indexHost[3]=index.norm.data(); indexHost[4]=index.idf.data();
        const size_t counts[5]={index.offsets.size(),index.docs.size(),index.tf.size(),index.norm.size(),index.idf.size()};
        const size_t widths[5]={8,4,4,4,4}; const aclDataType types[5]={ACL_INT64,ACL_INT32,ACL_FLOAT,ACL_FLOAT,ACL_FLOAT};
        Stage("search_cold_upload_index_once");
        for (size_t i=0;i<5;++i) {
            indexBytes[i]=Bytes(counts[i],widths[i]); indexUploadBytes+=indexBytes[i];
            indexMemory[i]=buffers.Allocate(indexBytes[i]); indexTensors[i]=tensors.Describe1D(counts[i],types[i],indexMemory[i]);
            Require(aclrtMemcpy(indexMemory[i],indexBytes[i],indexHost[i],indexBytes[i],ACL_MEMCPY_HOST_TO_DEVICE),"search cold index upload");
        }
        offsetMemory=buffers.Allocate(Bytes(maxQ+1,8)); wordMemory=buffers.Allocate(Bytes(std::max<size_t>(1,maxWords),4));
        const size_t elements[5]={1,maxQ*shards*k,maxQ*shards*k,maxQ*k,maxQ*k};
        for (size_t i=0;i<5;++i) {
            capacityBytes[i]=Bytes(elements[i],4); output[i]=buffers.Allocate(capacityBytes[i]);
            debugPoisonBytesPerCall+=capacityBytes[i];
        }
        Stage("search_cold_empty_capacity_bootstrap");
        // One synthetic all-empty capacity batch, no real query results cached. Its query
        // H2D and Top-K D2H are cold-only and explicitly recorded by the caller report.
        const auto bootstrap=SearchImpl(Queries(maxQ),Clock::now());
        for (size_t i=0;i<bootstrap.ids.size();++i)
            Check(bootstrap.scores[i]==0 && bootstrap.ids[i]==static_cast<int32_t>(i%k),"Cold empty bootstrap invalid");

    }
    Result Search(const Queries &queries)
    {
        const auto start=Clock::now(); // First public-call statement, before validation or packing.
        try { return SearchImpl(queries,start); }
        catch (...) { failed=true; throw; } // No retry with cached/CPU results after any failure.
    }
    void PoisonForDiagnostic()
    {
        Check(!closed && !failed,"Diagnostic on closed/failed session");
        std::vector<uint32_t> poison(*std::max_element(capacityBytes,capacityBytes+5)/4,UINT32_MAX);
        for (size_t i=0;i<5;++i)
            Require(aclrtMemcpy(output[i],capacityBytes[i],poison.data(),capacityBytes[i],ACL_MEMCPY_HOST_TO_DEVICE),"debug poison outside Search");
    }
    // Runs only at untimed checkpoints, never called from public Search or measured samples.
    void SaveDiagnostic(const Queries &queries,const Result &normal,const fs::path &directory,const fs::path &reference)
    {
        Check(!closed && !failed,"Diagnostic on failed session");
        try {
            const size_t q=queries.size(), count=q*static_cast<size_t>(docs), bytes=Bytes(count,4);
            Buffers debug; debug.stream=runtime.stream; void *full=debug.Allocate(bytes);
            std::vector<uint32_t> poison(count,UINT32_MAX);
            Require(aclrtMemcpy(full,bytes,poison.data(),bytes,ACL_MEMCPY_HOST_TO_DEVICE),"debug full score poison");
            const auto checked=SearchImpl(queries,Clock::now(),full);
            Check(Same(normal.scores,checked.scores) && Same(normal.ids,checked.ids),"Public debug0 and diagnostic debug1 Top-K differ");
            Request request; request.label="untimed_search_checkpoint"; request.offsets.resize(q+1);
            request.expected=Read<float>(reference); request.actual.resize(count);
            request.partialScores.resize(q*shards*k); request.partialIds.resize(q*shards*k);
            request.topScores=checked.scores; request.topIds=checked.ids;
            Require(aclrtMemcpy(request.actual.data(),bytes,full,bytes,ACL_MEMCPY_DEVICE_TO_HOST),"debug download full");
            const size_t partBytes=Bytes(request.partialScores.size(),4);
            Require(aclrtMemcpy(request.partialScores.data(),partBytes,output[1],partBytes,ACL_MEMCPY_DEVICE_TO_HOST),"debug candidates");
            Require(aclrtMemcpy(request.partialIds.data(),partBytes,output[2],partBytes,ACL_MEMCPY_DEVICE_TO_HOST),"debug candidate IDs");
            CompareReference(request); CheckRanking(request,docs,k,shards);
            uint32_t placeholder=UINT32_MAX;
            Require(aclrtMemcpy(&placeholder,4,output[0],4,ACL_MEMCPY_DEVICE_TO_HOST),"debug check normal placeholder");
            Check(placeholder==0,"Public Search failed to overwrite placeholder");
            Write(directory/"scores.bin",request.actual); Write(directory/"partial_scores.bin",request.partialScores);
            Write(directory/"partial_ids.bin",request.partialIds);
            Check(debug.Close(),"Debug memory release failed");
        } catch (...) { failed=true; throw; }
    }
    void VerifyIndex() const
    {
        Check(!closed && !failed,"Index check on closed/failed session");
        for (size_t i=0;i<5;++i) {
            std::vector<unsigned char> raw(indexBytes[i]);
            Require(aclrtMemcpy(raw.data(),raw.size(),indexMemory[i],raw.size(),ACL_MEMCPY_DEVICE_TO_HOST),"debug static index readback");
            Check(std::memcmp(raw.data(),indexHost[i],raw.size())==0,"Resident index changed");
        }
    }
    const uint64_t *WorkspaceCapacities() const { return capacityWorkspace[0]; }
    void Close()
    {
        if (closed) return;
        Check(tensors.Close(),"Static tensor release failed"); Check(buffers.Close(),"Buffer release failed");
        Check(runtime.Close(),"Search runtime release failed"); closed=true;
    }
    // Members destroy in reverse order: tensors, buffers, runtime. API libraries stay loaded.
};
}
