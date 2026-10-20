#pragma once
#include "../resident_chain_host/resident_chain_common.h"
#include "search_inputs.h"
#include <chrono>

namespace bm25search {
using Clock = std::chrono::steady_clock;
struct Result {
    std::vector<float> scores;
    std::vector<int32_t> ids;
    double searchMs = 0;
    size_t logicalQueryBytes = 0, queryH2DBytes = 0;
};

// Fixed-capacity in-process session. Constructor owns cold init, index upload and bootstrap.
// Every successful Search receives raw grouped IDs, performs its own packing, and returns
// ONLY CPU-available final Top-K. Diagnostics are separate methods called after Search.
class NpuSearchSession {
    const Index &index;
    ChainApi api;
    Runtime runtime;
    Buffers buffers;
    Tensors tensors;
    size_t maxQ, maxWords;
    int64_t k, shards, docs;
    uint64_t acceptedTopWorkspace;
    void *indexMemory[5]{}; aclTensor *indexTensors[5]{};
    const void *indexHost[5]{}; size_t indexBytes[5]{};
    void *offsetMemory = nullptr, *wordMemory = nullptr, *output[5]{};
    size_t capacityBytes[5]{};
    void *workspace[3]{}; uint64_t capacityWorkspace[3]{};
    bool bootstrapping = true, failed = false, closed = false;

    Result SearchImpl(const Queries &queries,Clock::time_point start)
    {
        Check(!closed && !failed,"Search session is closed/failed");
        auto packed = PackQueries(queries,maxQ,maxWords);
        const int64_t q = static_cast<int64_t>(queries.size());
        Tensors current; current.stream = runtime.stream;
        auto *offsets = current.Describe1D(packed.offsets.size(),ACL_INT64,offsetMemory);
        auto *words = current.Describe1D(std::max<size_t>(1,packed.words.size()),ACL_INT32,wordMemory);
        auto *scores = current.Describe(q,docs,ACL_FLOAT,output[0]);
        auto *localIn = current.Describe(q*shards,docs/shards,ACL_FLOAT,output[0]);
        auto *partialScores = current.Describe(q*shards,k,ACL_FLOAT,output[1]);
        auto *partialIds = current.Describe(q*shards,k,ACL_INT32,output[2]);
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
        aclOpExecutor *executors[3]{}; uint64_t sizes[3]{};
        Require(api.score.prepare(indexTensors[0],indexTensors[1],indexTensors[2],indexTensors[3],indexTensors[4],offsets,words,scores,
                                  &sizes[0],&executors[0]),"search score prepare");
        Require(api.prepareTop(localIn,k,partialScores,partialIds,&sizes[1],&executors[1]),"search local Top-K prepare");
        Require(api.prepareMerge(mergeScores,mergeIds,k,docs,topScores,topIds,&sizes[2],&executors[2]),"search merge prepare");
        Check(sizes[1] == acceptedTopWorkspace,"Search Top-K workspace differs from accepted radix_collect");
        for (size_t i = 0; i < 3; ++i) {
            Check(executors[i] && sizes[i] <= std::numeric_limits<size_t>::max(),"Invalid search executor/workspace");
            if (bootstrapping) { capacityWorkspace[i] = sizes[i]; workspace[i] = buffers.Allocate(static_cast<size_t>(sizes[i])); }
            else Check(sizes[i] <= capacityWorkspace[i],"Search workspace exceeds declared cold allocation; stop without hidden resize");
        }
        Require(api.score.execute(workspace[0],sizes[0],executors[0],runtime.stream),"search score execute"); executors[0] = nullptr;
        Require(api.executeTop(workspace[1],sizes[1],executors[1],runtime.stream),"search local Top-K execute"); executors[1] = nullptr;
        Require(api.executeMerge(workspace[2],sizes[2],executors[2],runtime.stream),"search merge execute"); executors[2] = nullptr;
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
                     size_t queryCapacity,size_t wordCapacity,uint64_t acceptedWorkspace)
        : index(source),api(vendors),maxQ(queryCapacity),maxWords(wordCapacity),k(selectedK),shards(selectedShards),
          docs(static_cast<int64_t>(index.norm.size())),acceptedTopWorkspace(acceptedWorkspace)
    {
        Check(maxQ >= 1 && maxQ <= 10000 && maxWords <= 16777216 &&
              static_cast<uint64_t>(maxQ) <= static_cast<uint64_t>(INT32_MAX)/docs &&
              k >= 1 && k <= 32 && shards >= 1 && shards <= 32 && docs%shards == 0 && docs/shards >= k,
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
        const size_t elements[5]={maxQ*static_cast<size_t>(docs),maxQ*shards*k,maxQ*shards*k,maxQ*k,maxQ*k};
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
        bootstrapping=false;
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
        std::vector<uint32_t> poison(capacityBytes[0]/4,UINT32_MAX);
        for (size_t i=0;i<5;++i)
            Require(aclrtMemcpy(output[i],capacityBytes[i],poison.data(),capacityBytes[i],ACL_MEMCPY_HOST_TO_DEVICE),"debug poison outside Search");
    }
    void DownloadDebug(size_t q,const fs::path &directory) const
    {
        Check(!closed && !failed && q>=1 && q<=maxQ,"Invalid debug snapshot");
        std::vector<float> scores(q*docs),partial(q*shards*k); std::vector<int32_t> ids(partial.size());
        Require(aclrtMemcpy(scores.data(),Bytes(scores.size(),4),output[0],Bytes(scores.size(),4),ACL_MEMCPY_DEVICE_TO_HOST),"debug full scores outside Search");
        Require(aclrtMemcpy(partial.data(),Bytes(partial.size(),4),output[1],Bytes(partial.size(),4),ACL_MEMCPY_DEVICE_TO_HOST),"debug candidates outside Search");
        Require(aclrtMemcpy(ids.data(),Bytes(ids.size(),4),output[2],Bytes(ids.size(),4),ACL_MEMCPY_DEVICE_TO_HOST),"debug candidate IDs outside Search");
        Write(directory/"scores.bin",scores); Write(directory/"partial_scores.bin",partial); Write(directory/"partial_ids.bin",ids);
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
    const uint64_t *WorkspaceCapacities() const { return capacityWorkspace; }
    void Close()
    {
        if (closed) return;
        Check(tensors.Close(),"Static tensor release failed"); Check(buffers.Close(),"Buffer release failed");
        Check(runtime.Close(),"Search runtime release failed"); closed=true;
    }
    // Members destroy in reverse order: tensors, buffers, runtime. API libraries stay loaded.
};
}
