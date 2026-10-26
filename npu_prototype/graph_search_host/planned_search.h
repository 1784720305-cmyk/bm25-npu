#pragma once
#include "../fused_chain_host/fused_runtime.h"
#include "../fused_search_host/fused_api.h"
#include "../../common/fixed_query_packet.h"
#include "../../common/shard_directory.h"
#include <array>
#include <chrono>

namespace bm25search {
using Clock=std::chrono::steady_clock;
struct PlannedResult {
    std::vector<float> scores;
    std::vector<int32_t> ids;
    double searchMs=0;
    size_t logicalQueryBytes=0,queryH2DBytes=0,topkD2HBytes=0;
};

// Coordinates graph lifetime with ALL referenced resources, including constructor-failure cleanup.
struct PlanResources {
    Runtime runtime;
    Buffers buffers;
    Tensors tensors;
    std::vector<void *> pinned;
    struct Guard { unsigned char *allocation; size_t size; };
    std::vector<Guard> guards;
    aclmdlRI model{};
    bool capturing=false,modelValid=false,closed=false;
    void *Host(size_t bytes)
    {
        void *p=nullptr; Require(aclrtMallocHost(&p,bytes),"allocate pinned plan arena");
        Check(p!=nullptr,"Null pinned plan arena"); pinned.push_back(p); return p;
    }
    void *Device(size_t bytes)
    {
        auto *base=static_cast<unsigned char *>(buffers.Allocate(bytes+128));
        std::array<unsigned char,64> guard{}; guard.fill(0xa7);
        Require(aclrtMemcpy(base,64,guard.data(),64,ACL_MEMCPY_HOST_TO_DEVICE),"initialize leading plan guard");
        Require(aclrtMemcpy(base+64+bytes,64,guard.data(),64,ACL_MEMCPY_HOST_TO_DEVICE),"initialize trailing plan guard");
        guards.push_back({base,bytes}); return base+64;
    }
    void CheckGuards() const
    {
        std::array<unsigned char,64> actual{};
        for (const auto &guard:guards) for (size_t offset:{size_t(0),size_t(64)+guard.size}) {
            Require(aclrtMemcpy(actual.data(),64,guard.allocation+offset,64,ACL_MEMCPY_DEVICE_TO_HOST),"read plan guard outside timing");
            Check(std::all_of(actual.begin(),actual.end(),[](unsigned char value){return value==0xa7;}),"Plan arena guard overwritten");
        }
    }
    bool Close()
    {
        if (closed) return true;
        if (capturing) {
            // CaptureEnd is also required on exceptions; never synchronize an active capture.
            if (aclmdlRICaptureEnd(runtime.stream,&model)!=ACL_ERROR_NONE) return false;
            capturing=false; modelValid=true;
        }
        if (runtime.stream && aclrtSynchronizeStream(runtime.stream)!=ACL_ERROR_NONE) return false;
        if (modelValid) {
            if (aclmdlRIDestroy(model)!=ACL_ERROR_NONE) return false;
            modelValid=false;
        }
        bool ok=true;
        for (void *p:pinned) if (aclrtFreeHost(p)!=ACL_ERROR_NONE) ok=false;
        pinned.clear();
        tensors.stream=nullptr; buffers.stream=nullptr;
        ok=tensors.Close() && ok; ok=buffers.Close() && ok; ok=runtime.Close() && ok;
        closed=true; return ok;
    }
    ~PlanResources()
    {
        if (!Close()) {
            // A failed capture/sync/destroy cannot prove the graph has stopped using addresses.
            // Retain ALL ownership until process exit instead of freeing a subset underneath it.
            std::cerr << "cleanup_failed=plan resources retained until process exit\n";
            tensors.items.clear(); buffers.items.clear(); pinned.clear();
            tensors.stream=nullptr; buffers.stream=nullptr;
            runtime.stream=nullptr; runtime.context=nullptr; runtime.initialized=false; runtime.selected=false;
        }
    }
};

class PlannedNpuSearchSession {
    const Index &index;
    SearchFusedApi api; // Loaded libraries outlive the plan and all its descriptors.
    std::vector<int64_t> directory;
    PlanResources resources;
    bm25plan::PacketLayout layout;
    size_t maxQ,maxWords;
    int64_t k,shards,docs;
    bool graph,directoryMode,failed=false,closed=false;
    void *indexMemory[5]{},*input=nullptr,*output=nullptr,*hostInput=nullptr,*hostOutput=nullptr;
    const void *indexHost[5]{};
    size_t indexBytes[5]{};
    aclTensor *indexTensors[5]{},*offsets=nullptr,*words=nullptr,*debugPlaceholder=nullptr;
    aclTensor *partialScores=nullptr,*partialIds=nullptr,*topScores=nullptr,*topIds=nullptr;
    void *partial[2]{},*placeholder=nullptr,*workspace[2]{};
    uint64_t capacity[2]{};
    struct Prepared { aclOpExecutor *executors[2]{}; uint64_t sizes[2]{}; };
    Prepared Prepare(aclTensor *debug,int64_t mode)
    {
        Prepared p;
        Require(api.prepareFused(indexTensors[0],indexTensors[1],indexTensors[2],indexTensors[3],indexTensors[4],
            offsets,words,k,shards,mode,partialScores,partialIds,debug,&p.sizes[0],&p.executors[0]),"plan prepare fused");
        Require(api.prepareMerge(partialScores,partialIds,k,docs,topScores,topIds,&p.sizes[1],&p.executors[1]),"plan prepare merge");
        for (size_t i=0;i<2;++i) Check(p.executors[i] && p.sizes[i]<=SIZE_MAX,"Invalid plan executor/workspace");
        return p;
    }
    void Upload()
    {
        Require(aclrtMemcpyAsync(input,layout.inputBytes,hostInput,layout.inputBytes,ACL_MEMCPY_HOST_TO_DEVICE,
            resources.runtime.stream),"plan asynchronous query H2D");
    }
    void Execute(Prepared &p,void *const (&work)[2])
    {
        Require(api.executeFused(work[0],p.sizes[0],p.executors[0],resources.runtime.stream),"plan execute fused");
        p.executors[0]=nullptr;
        Require(api.executeMerge(work[1],p.sizes[1],p.executors[1],resources.runtime.stream),"plan execute merge");
        p.executors[1]=nullptr;
    }
    void Download()
    {
        Require(aclrtMemcpyAsync(hostOutput,layout.outputBytes,output,layout.outputBytes,ACL_MEMCPY_DEVICE_TO_HOST,
            resources.runtime.stream),"plan asynchronous final Top-K D2H");
    }
    PlannedResult Receive(size_t logical)
    {
        PlannedResult result; result.scores.resize(maxQ*k); result.ids.resize(maxQ*k);
        std::memcpy(result.scores.data(),hostOutput,result.scores.size()*4);
        std::memcpy(result.ids.data(),static_cast<unsigned char *>(hostOutput)+layout.idOffset,result.ids.size()*4);
        for (size_t i=0;i<result.ids.size();++i) {
            Check(result.ids[i]>=0 && result.ids[i]<docs && std::isfinite(result.scores[i]) && result.scores[i]>=0,
                "Invalid CPU-available planned Top-K");
            if (i%static_cast<size_t>(k)) Check(result.scores[i-1]>result.scores[i] ||
                (result.scores[i-1]==result.scores[i] && result.ids[i-1]<result.ids[i]),"Planned Top-K order/tie mismatch");
        }
        result.logicalQueryBytes=logical; result.queryH2DBytes=layout.inputBytes; result.topkD2HBytes=layout.outputBytes;
        return result;
    }
public:
    uint64_t indexUploadBytes=0,replays=0,normalSearchCalls=0,normalPrepareCalls=0;
    PlannedNpuSearchSession(const Index &source,const fs::path &vendors,int device,int64_t selectedK,
        int64_t selectedShards,size_t queryCapacity,size_t wordCapacity,bool capture,bool useDirectory)
        :index(source),api(vendors),layout(queryCapacity,wordCapacity,selectedK),maxQ(queryCapacity),maxWords(wordCapacity),
         k(selectedK),shards(selectedShards),docs(static_cast<int64_t>(source.norm.size())),graph(capture),directoryMode(useDirectory)
    {
        Check(docs>=1 && maxQ<=static_cast<uint64_t>(INT32_MAX)/docs && shards>=1 && shards<=32 &&
            docs%shards==0 && docs/shards>=k && docs/shards<=8192,"Invalid planned search dimensions");
        auto &rt=resources.runtime; rt.device=device;
        Require(aclInit(nullptr),"aclInit"); rt.initialized=true;
        Require(aclrtSetDevice(device),"aclrtSetDevice"); rt.selected=true;
        Require(aclrtCreateContext(&rt.context,device),"aclrtCreateContext");
        aclrtRunMode mode; Require(aclrtGetRunMode(&mode),"aclrtGetRunMode"); Check(mode==ACL_HOST,"Requires ACL_HOST");
        Require(aclrtCreateStream(&rt.stream),"aclrtCreateStream");
        resources.buffers.stream=rt.stream; resources.tensors.stream=rt.stream;
        if (directoryMode) directory=bm25plan::BuildDirectory(index.offsets,index.docs,index.norm.size(),shards);
        indexHost[0]=directoryMode?directory.data():index.offsets.data();
        indexHost[1]=index.docs.data(); indexHost[2]=index.tf.data(); indexHost[3]=index.norm.data(); indexHost[4]=index.idf.data();
        const size_t counts[5]={directoryMode?directory.size():index.offsets.size(),index.docs.size(),index.tf.size(),index.norm.size(),index.idf.size()};
        const size_t widths[5]={8,4,4,4,4}; const aclDataType types[5]={ACL_INT64,ACL_INT32,ACL_FLOAT,ACL_FLOAT,ACL_FLOAT};
        for (size_t i=0;i<5;++i) {
            indexBytes[i]=Bytes(counts[i],widths[i]); indexUploadBytes+=indexBytes[i];
            indexMemory[i]=resources.Device(indexBytes[i]); indexTensors[i]=resources.tensors.Describe1D(counts[i],types[i],indexMemory[i]);
            Require(aclrtMemcpy(indexMemory[i],indexBytes[i],indexHost[i],indexBytes[i],ACL_MEMCPY_HOST_TO_DEVICE),"plan cold index upload");
        }
        hostInput=resources.Host(layout.inputBytes); hostOutput=resources.Host(layout.outputBytes);
        input=resources.Device(layout.inputBytes); output=resources.Device(layout.outputBytes);
        placeholder=resources.Device(4);
        for (void *&p:partial) p=resources.Device(Bytes(maxQ*shards*k,4));
        auto &t=resources.tensors; auto *in=static_cast<unsigned char *>(input),*out=static_cast<unsigned char *>(output);
        offsets=t.Describe1D(maxQ+1,ACL_INT64,input);
        words=t.Describe1D(std::max<size_t>(1,maxWords),ACL_INT32,in+layout.wordOffset);
        debugPlaceholder=t.Describe1D(1,ACL_FLOAT,placeholder);
        partialScores=Describe3D(t,maxQ,shards,k,ACL_FLOAT,partial[0]); partialIds=Describe3D(t,maxQ,shards,k,ACL_INT32,partial[1]);
        topScores=t.Describe(maxQ,k,ACL_FLOAT,out); topIds=t.Describe(maxQ,k,ACL_INT32,out+layout.idOffset);
        const Queries empty(maxQ); layout.Pack(empty,hostInput);
        auto prepared=Prepare(debugPlaceholder,0);
        for (size_t i=0;i<2;++i) { capacity[i]=prepared.sizes[i]; workspace[i]=resources.buffers.Allocate(static_cast<size_t>(capacity[i])); }
        if (graph) {
            Require(aclmdlRICaptureBegin(rt.stream,ACL_MODEL_RI_CAPTURE_MODE_GLOBAL),"begin planned capture"); resources.capturing=true;
            Upload(); Execute(prepared,workspace); Download();
            Require(aclmdlRICaptureEnd(rt.stream,&resources.model),"end planned capture");
            resources.capturing=false; resources.modelValid=true;
            const auto bootstrap=Search(empty);
            for (size_t i=0;i<bootstrap.ids.size();++i) Check(bootstrap.scores[i]==0 && bootstrap.ids[i]==static_cast<int32_t>(i%k),"Graph empty bootstrap failed");
        } else {
            Upload(); Execute(prepared,workspace); Download();
            Require(aclrtSynchronizeStream(rt.stream),"async empty bootstrap complete");
            const auto bootstrap=Receive((maxQ+1)*8);
            for (size_t i=0;i<bootstrap.ids.size();++i) Check(bootstrap.scores[i]==0 && bootstrap.ids[i]==static_cast<int32_t>(i%k),"Async empty bootstrap failed");
        }
    }
    PlannedResult Search(const Queries &queries)
    {
        const auto start=Clock::now();
        try {
            Check(!closed && !failed,"Planned Search session closed/failed");
            const size_t logical=layout.Pack(queries,hostInput); ++normalSearchCalls;
            if (graph) { Require(aclmdlRIExecuteAsync(resources.model,resources.runtime.stream),"replay planned search"); ++replays; }
            else {
                auto p=Prepare(debugPlaceholder,0); normalPrepareCalls+=2;
                for (size_t i=0;i<2;++i) Check(p.sizes[i]<=capacity[i],"Fixed plan workspace exceeded");
                Upload(); Execute(p,workspace); Download();
            }
            Require(aclrtSynchronizeStream(resources.runtime.stream),"planned Search wait through final D2H");
            auto result=Receive(logical);
            result.searchMs=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
            return result;
        } catch (...) { failed=true; throw; }
    }
    void PoisonForDiagnostic()
    {
        Check(!closed && !failed,"Poison on failed plan"); resources.CheckGuards();
        std::vector<unsigned char> poison(std::max(layout.outputBytes,Bytes(maxQ*shards*k,4)),0xff);
        Require(aclrtMemcpy(output,layout.outputBytes,poison.data(),layout.outputBytes,ACL_MEMCPY_HOST_TO_DEVICE),"poison plan output");
        for (void *p:partial) Require(aclrtMemcpy(p,Bytes(maxQ*shards*k,4),poison.data(),Bytes(maxQ*shards*k,4),ACL_MEMCPY_HOST_TO_DEVICE),"poison plan candidates");
        std::memset(hostOutput,0xff,layout.outputBytes);
    }
    void SaveDiagnostic(const Queries &queries,const PlannedResult &normal,const fs::path &path,const fs::path &reference)
    {
        Check(!closed && !failed,"Diagnostic on failed plan");
        try {
            layout.Pack(queries,hostInput);
            Buffers extra; extra.stream=resources.runtime.stream;
            const size_t count=maxQ*static_cast<size_t>(docs),bytes=Bytes(count,4);
            auto *base=static_cast<unsigned char *>(extra.Allocate(bytes+128));
            std::vector<unsigned char> poison(bytes+128,0xff);
            Require(aclrtMemcpy(base,bytes+128,poison.data(),bytes+128,ACL_MEMCPY_HOST_TO_DEVICE),"poison debug score buffer and guards");
            Tensors t; t.stream=resources.runtime.stream; auto *full=t.Describe(maxQ,docs,ACL_FLOAT,base+64);
            auto p=Prepare(full,1); void *work[2]={extra.Allocate(p.sizes[0]),extra.Allocate(p.sizes[1])};
            Upload(); Execute(p,work); Download(); Require(aclrtSynchronizeStream(resources.runtime.stream),"debug plan complete");
            auto checked=Receive((maxQ+1)*8);
            Check(Same(normal.scores,checked.scores) && Same(normal.ids,checked.ids),"Replay and full-score diagnostic differ");
            Request request; request.label="planned_checkpoint"; request.offsets.resize(maxQ+1);
            request.expected=Read<float>(reference); request.actual.resize(count);
            request.partialScores.resize(maxQ*shards*k); request.partialIds.resize(maxQ*shards*k);
            request.topScores=normal.scores; request.topIds=normal.ids;
            Require(aclrtMemcpy(request.actual.data(),bytes,base+64,bytes,ACL_MEMCPY_DEVICE_TO_HOST),"debug download full plan scores");
            const size_t partBytes=Bytes(maxQ*shards*k,4);
            Require(aclrtMemcpy(request.partialScores.data(),partBytes,partial[0],partBytes,ACL_MEMCPY_DEVICE_TO_HOST),"debug download plan candidates");
            Require(aclrtMemcpy(request.partialIds.data(),partBytes,partial[1],partBytes,ACL_MEMCPY_DEVICE_TO_HOST),"debug download plan candidate IDs");
            CompareReference(request); CheckRanking(request,docs,k,shards);
            std::array<unsigned char,64> guard{};
            for (size_t offset:{size_t(0),bytes+64}) {
                Require(aclrtMemcpy(guard.data(),64,base+offset,64,ACL_MEMCPY_DEVICE_TO_HOST),"read debug score guard");
                Check(std::all_of(guard.begin(),guard.end(),[](unsigned char value){return value==0xff;}),"Debug full-score guard overwritten");
            }
            resources.CheckGuards();
            Write(path/"scores.bin",request.actual); Write(path/"partial_scores.bin",request.partialScores); Write(path/"partial_ids.bin",request.partialIds);
        } catch (...) { failed=true; throw; }
    }
    void VerifyIndex() const
    {
        Check(!closed && !failed,"Index check on failed plan");
        for (size_t i=0;i<5;++i) {
            std::vector<unsigned char> actual(indexBytes[i]);
            Require(aclrtMemcpy(actual.data(),actual.size(),indexMemory[i],actual.size(),ACL_MEMCPY_DEVICE_TO_HOST),"verify plan index readback");
            Check(std::memcmp(actual.data(),indexHost[i],actual.size())==0,"Resident plan index/directory changed");
        }
        resources.CheckGuards();
    }
    const uint64_t *WorkspaceCapacities() const { return capacity; }
    size_t InputBytes() const { return layout.inputBytes; }
    size_t OutputBytes() const { return layout.outputBytes; }
    size_t DirectoryBytes() const { return directory.size()*8; }
    void Close() { if (!closed) { Check(resources.Close(),"Plan cleanup failed"); closed=true; } }
};
}
