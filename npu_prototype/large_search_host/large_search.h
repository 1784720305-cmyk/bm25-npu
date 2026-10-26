#pragma once
#include "../graph_search_host/planned_search.h"
#include "../../common/large_search_plan.h"
#include "../../cpu_reference/topk_document_shards.h" // untimed independent bounded-heap checks only
namespace bm25large {
using bm25search::Queries;using bm25search::PlannedResult;using bm25search::Clock;
struct Lists {void *scores{},*ids{};aclTensor *s{},*i{};};
struct Prepared {std::vector<aclOpExecutor *> executors;std::vector<uint64_t> bytes;};
class Session {
    const Index &index;
    Plan plan;
    SearchFusedApi api;
    bm25search::PlanResources resources;
    bm25plan::PacketLayout packet;
    std::vector<int64_t> directory;
    std::vector<float> paddedNorm;
    const void *indexHost[5]{};size_t indexBytes[5]{};aclTensor *indexTensors[5]{};
    aclTensor *offsets{},*words{},*placeholderTensor{};
    void *input{},*output{},*hostInput{},*hostOutput{},*placeholder{};
    std::vector<Lists> lists;
    std::vector<void *> work;
    bool failed=false,closed=false;
    Prepared Prepare(aclTensor *debug,int64_t mode) {
        Prepared p;p.executors.resize(lists.size());p.bytes.resize(lists.size());
        Require(api.prepareFused(indexTensors[0],indexTensors[1],indexTensors[2],indexTensors[3],indexTensors[4],offsets,words,
            plan.k,plan.leaves,mode,lists[0].s,lists[0].i,debug,&p.bytes[0],&p.executors[0]),"prepare bounded leaves");
        for(size_t i=1;i<lists.size();++i)
            Require(api.prepareMerge(lists[i-1].s,lists[i-1].i,plan.k,plan.padded,lists[i].s,lists[i].i,
                &p.bytes[i],&p.executors[i]),"prepare merge tree level");
        for(size_t i=0;i<p.bytes.size();++i)Check(p.executors[i] && p.bytes[i]<=SIZE_MAX,"Invalid tree executor/workspace");
        return p;
    }
    void Execute(Prepared &p,const std::vector<void *> &workspace) {
        Require(api.executeFused(workspace[0],p.bytes[0],p.executors[0],resources.runtime.stream),"execute bounded leaves");p.executors[0]=nullptr;
        for(size_t i=1;i<lists.size();++i) {
            Require(api.executeMerge(workspace[i],p.bytes[i],p.executors[i],resources.runtime.stream),"execute merge level");p.executors[i]=nullptr;
        }
    }
    void Upload() {Require(aclrtMemcpyAsync(input,packet.inputBytes,hostInput,packet.inputBytes,ACL_MEMCPY_HOST_TO_DEVICE,resources.runtime.stream),"large query H2D");}
    void Download() {Require(aclrtMemcpyAsync(hostOutput,packet.outputBytes,output,packet.outputBytes,ACL_MEMCPY_DEVICE_TO_HOST,resources.runtime.stream),"large topK D2H");}
    PlannedResult Receive(size_t logical) {
        PlannedResult result;result.scores.resize(plan.q*plan.k);result.ids.resize(result.scores.size());
        std::memcpy(result.scores.data(),hostOutput,Bytes(result.scores.size(),4));
        std::memcpy(result.ids.data(),static_cast<unsigned char *>(hostOutput)+packet.idOffset,Bytes(result.ids.size(),4));
        for(size_t i=0;i<result.ids.size();++i) {
            Check(result.ids[i]>=0 && static_cast<uint64_t>(result.ids[i])<plan.docs && std::isfinite(result.scores[i]) && result.scores[i]>=0,"Invalid real-document result");
            if(i%plan.k)Check(result.scores[i-1]>result.scores[i] || (result.scores[i-1]==result.scores[i] && result.ids[i-1]<result.ids[i]),"Strict global ranking failure");
        }
        result.logicalQueryBytes=logical;result.queryH2DBytes=packet.inputBytes;result.topkD2HBytes=packet.outputBytes;return result;
    }
    void MemoryGate(uint64_t bytes,const char *why) {
        size_t free=0,total=0;Require(aclrtGetMemInfo(ACL_HBM_MEM,&free,&total),"query actual free device memory");
        Check(total>0 && bytes<=static_cast<uint64_t>(free)*4/5,why);
    }
    static std::vector<Candidate> Ordered(std::array<Candidate,32> &heap,int64_t count) {
        std::sort(heap.begin(),heap.begin()+count,Better{});return {heap.begin(),heap.begin()+count};
    }
public:
    uint64_t indexUploadBytes=0,replays=0,normalSearchCalls=0,initialFreeBytes=0,totalDeviceBytes=0;
    std::vector<uint64_t> workspaceBytes;
    Session(const Index &source,const fs::path &vendors,int device,uint64_t q,uint64_t k,uint64_t wordsCapacity)
        :index(source),plan(source.norm.size(),source.idf.size(),source.docs.size(),q,k,wordsCapacity),api(vendors),packet(q,wordsCapacity,k) {
        auto &rt=resources.runtime;rt.device=device;
        Require(aclInit(nullptr),"aclInit");rt.initialized=true;Require(aclrtSetDevice(device),"set device");rt.selected=true;
        Require(aclrtCreateContext(&rt.context,device),"create context");Require(aclrtCreateStream(&rt.stream),"create stream");
        resources.buffers.stream=resources.tensors.stream=rt.stream;
        size_t free=0,total=0;Require(aclrtGetMemInfo(ACL_HBM_MEM,&free,&total),"initial actual memory");
        initialFreeBytes=free;totalDeviceBytes=total;MemoryGate(plan.knownDeviceBytes,"Index/tree/arenas exceed80% of actual free memory");
        paddedNorm=index.norm;paddedNorm.resize(plan.padded,1.0F);
        directory=bm25plan::BuildDirectory(index.offsets,index.docs,plan.padded,plan.leaves);
        indexHost[0]=directory.data();indexHost[1]=index.docs.data();indexHost[2]=index.tf.data();indexHost[3]=paddedNorm.data();indexHost[4]=index.idf.data();
        const uint64_t counts[5]={directory.size(),index.docs.size(),index.tf.size(),paddedNorm.size(),index.idf.size()};
        const aclDataType types[5]={ACL_INT64,ACL_INT32,ACL_FLOAT,ACL_FLOAT,ACL_FLOAT};
        auto &t=resources.tensors;
        for(size_t i=0;i<5;++i) {
            indexBytes[i]=Bytes(counts[i],i?4:8);indexUploadBytes+=indexBytes[i];auto *memory=resources.Device(indexBytes[i]);
            indexTensors[i]=t.Describe1D(counts[i],types[i],memory);
            Require(aclrtMemcpy(memory,indexBytes[i],indexHost[i],indexBytes[i],ACL_MEMCPY_HOST_TO_DEVICE),"one-time large index upload");
            indexMemory[i]=memory;
        }
        hostInput=resources.Host(packet.inputBytes);hostOutput=resources.Host(packet.outputBytes);
        input=resources.Device(packet.inputBytes);output=resources.Device(packet.outputBytes);placeholder=resources.Device(4);
        offsets=t.Describe1D(q+1,ACL_INT64,input);words=t.Describe1D(wordsCapacity?wordsCapacity:1,ACL_INT32,static_cast<unsigned char *>(input)+packet.wordOffset);
        placeholderTensor=t.Describe1D(1,ACL_FLOAT,placeholder);
        for(size_t i=0;i<plan.lists.size();++i) {
            Lists x;const auto count=plan.q*plan.lists[i]*plan.k;
            if(i+1==plan.lists.size()) {x.scores=output;x.ids=static_cast<unsigned char *>(output)+packet.idOffset;}
            else {x.scores=resources.Device(Bytes(count,4));x.ids=resources.Device(Bytes(count,4));}
            x.s=Describe3D(t,q,plan.lists[i],k,ACL_FLOAT,x.scores);x.i=Describe3D(t,q,plan.lists[i],k,ACL_INT32,x.ids);lists.push_back(x);
        }
        const Queries empty(q);packet.Pack(empty,hostInput);auto prepared=Prepare(placeholderTensor,0);workspaceBytes=prepared.bytes;
        uint64_t sum=0;for(auto bytes:prepared.bytes)sum=Add(sum,bytes);MemoryGate(sum,"Actual tree workspace exceeds memory budget");
        for(auto bytes:prepared.bytes)work.push_back(resources.buffers.Allocate(bytes));
        Require(aclmdlRICaptureBegin(rt.stream,ACL_MODEL_RI_CAPTURE_MODE_GLOBAL),"begin large graph capture");resources.capturing=true;
        Upload();Execute(prepared,work);Download();Require(aclmdlRICaptureEnd(rt.stream,&resources.model),"end large graph capture");
        resources.capturing=false;resources.modelValid=true;
        auto bootstrap=Search(empty);
        for(size_t i=0;i<bootstrap.ids.size();++i)Check(bootstrap.scores[i]==0 && bootstrap.ids[i]==static_cast<int32_t>(i%k),"Padded merge empty bootstrap failed");
    }
    void *indexMemory[5]{};
    PlannedResult Search(const Queries &queries) {
        const auto start=Clock::now();
        try {
            Check(!closed && !failed,"Large session closed/failed");const auto logical=packet.Pack(queries,hostInput);++normalSearchCalls;
            Require(aclmdlRIExecuteAsync(resources.model,resources.runtime.stream),"replay bounded merge tree");++replays;
            Require(aclrtSynchronizeStream(resources.runtime.stream),"complete Search through result D2H");
            auto result=Receive(logical);result.searchMs=std::chrono::duration<double,std::milli>(Clock::now()-start).count();return result;
        } catch(...) {failed=true;throw;}
    }
    void PoisonForDiagnostic() {
        resources.CheckGuards();
        for(size_t i=0;i<lists.size();++i) {
            std::vector<uint32_t> poison(plan.q*plan.lists[i]*plan.k,UINT32_MAX);
            for(void *p:{lists[i].scores,lists[i].ids})Require(aclrtMemcpy(p,Bytes(poison.size(),4),poison.data(),Bytes(poison.size(),4),ACL_MEMCPY_HOST_TO_DEVICE),"poison tree level");
        }
    }
    void SaveDiagnostic(const Queries &queries,const PlannedResult &normal,const fs::path &path,const fs::path &reference) {
        Check(!closed && !failed,"Diagnostic on invalid large session");
        packet.Pack(queries,hostInput);PoisonForDiagnostic();
        const auto fullBytes=Bytes(plan.q*plan.padded,4);MemoryGate(fullBytes+128,"Full-score diagnostic exceeds device memory budget");
        Buffers extra;extra.stream=resources.runtime.stream;Tensors t;t.stream=resources.runtime.stream;
        auto *base=static_cast<unsigned char *>(extra.Allocate(fullBytes+128));
        const std::vector<unsigned char> poison(8*1024*1024,0xff);
        for(size_t pos=0;pos<fullBytes+128;) {const auto bytes=std::min(poison.size(),fullBytes+128-pos);
            Require(aclrtMemcpy(base+pos,bytes,poison.data(),bytes,ACL_MEMCPY_HOST_TO_DEVICE),"poison bounded diagnostic buffer");pos+=bytes;}
        auto *full=t.Describe(plan.q,plan.padded,ACL_FLOAT,base+64);auto prepared=Prepare(full,1);std::vector<void *> workspace;
        uint64_t sum=0;for(auto bytes:prepared.bytes)sum=Add(sum,bytes);MemoryGate(sum,"Diagnostic workspace budget exceeded");
        for(auto bytes:prepared.bytes)workspace.push_back(extra.Allocate(bytes));
        Upload();Execute(prepared,workspace);Download();Require(aclrtSynchronizeStream(resources.runtime.stream),"large full diagnostic completion");
        auto replay=Receive(0);Check(Same(normal.ids,replay.ids) && Same(normal.scores,replay.scores),"Normal graph and diagnostic outputs differ");
        std::vector<std::vector<float>> scores(lists.size());std::vector<std::vector<int32_t>> ids(lists.size());
        for(size_t i=0;i<lists.size();++i) {
            const auto count=plan.q*plan.lists[i]*plan.k;scores[i].resize(count);ids[i].resize(count);
            Require(aclrtMemcpy(scores[i].data(),Bytes(count,4),lists[i].scores,Bytes(count,4),ACL_MEMCPY_DEVICE_TO_HOST),"read diagnostic level scores");
            Require(aclrtMemcpy(ids[i].data(),Bytes(count,4),lists[i].ids,Bytes(count,4),ACL_MEMCPY_DEVICE_TO_HOST),"read diagnostic level IDs");
            Write(path/("level_"+std::to_string(i)+"_scores.bin"),scores[i]);
            Write(path/("level_"+std::to_string(i)+"_ids.bin"),ids[i]);
        }
        Check(fs::file_size(reference)==Bytes(plan.q*plan.docs,4),"Independent original-document reference size mismatch");
        std::ifstream expected(reference,std::ios::binary);std::ofstream actual(path/"scores.bin",std::ios::binary);double maxError=0;
        auto compareList=[&](size_t level,uint64_t query,uint64_t list,const std::vector<Candidate> &wanted) {
            Check(wanted.size()==plan.k,"Insufficient independent candidates");
            for(size_t rank=0;rank<plan.k;++rank) {const size_t pos=(query*plan.lists[level]+list)*plan.k+rank;
                Check(ids[level][pos]==wanted[rank].id && std::memcmp(&scores[level][pos],&wanted[rank].score,4)==0,"Leaf/tree exact IDs or score bits differ");}
        };
        for(uint64_t query=0;query<plan.q;++query) {
            std::array<Candidate,32> gold{};int64_t goldCount=0;
            for(uint64_t leaf=0;leaf<plan.leaves;++leaf) {
                std::vector<float> row(Leaf),ref(Leaf);const auto first=leaf*Leaf,valid=first<plan.docs?std::min(Leaf,plan.docs-first):0;
                Require(aclrtMemcpy(row.data(),Bytes(Leaf,4),base+64+Bytes(query*plan.padded+first,4),Bytes(Leaf,4),ACL_MEMCPY_DEVICE_TO_HOST),"read bounded score shard");
                if(valid) {Check(static_cast<bool>(expected.read(reinterpret_cast<char *>(ref.data()),Bytes(valid,4))),"Reference read failed");
                    actual.write(reinterpret_cast<const char *>(row.data()),Bytes(valid,4));}
                std::array<Candidate,32> heap{};int64_t count=0;
                for(uint64_t d=0;d<Leaf;++d) {
                    Check(std::isfinite(row[d]) && row[d]>=0,"Invalid actual full score");
                    if(d<valid) {const double error=std::abs(static_cast<double>(row[d])-ref[d]);
                        Check(std::isfinite(ref[d]) && ref[d]>=0 && error<=1e-6+2e-6*std::abs(ref[d]),"Ordered FP32 formula mismatch");maxError=std::max(maxError,error);
                        KeepCandidate(gold,goldCount,plan.k,{ref[d],static_cast<int32_t>(first+d)});
                    } else Check(row[d]==0,"Padded articles must have zero score");
                    KeepCandidate(heap,count,plan.k,{row[d],static_cast<int32_t>(first+d)});
                }
                compareList(0,query,leaf,Ordered(heap,count));
            }
            const auto ordered=Ordered(gold,goldCount);
            for(size_t rank=0;rank<plan.k;++rank)Check(normal.ids[query*plan.k+rank]==ordered[rank].id,"Independent original-document strict final ranking differs");
        }
        for(size_t level=1;level<lists.size();++level)for(uint64_t query=0;query<plan.q;++query)for(uint64_t group=0;group<plan.lists[level];++group) {
            std::array<Candidate,32> heap{};int64_t count=0;
            for(uint64_t child=group*FanIn;child<std::min((group+1)*FanIn,plan.lists[level-1]);++child)
                for(size_t rank=0;rank<plan.k;++rank) {const size_t pos=(query*plan.lists[level-1]+child)*plan.k+rank;
                    KeepCandidate(heap,count,plan.k,{scores[level-1][pos],ids[level-1][pos]});}
            compareList(level,query,group,Ordered(heap,count));
        }
        actual.close();Check(bool(actual),"Cannot save diagnostic scores");
        std::array<unsigned char,64> guard{};
        for(size_t pos:{size_t(0),fullBytes+64}) {Require(aclrtMemcpy(guard.data(),64,base+pos,64,ACL_MEMCPY_DEVICE_TO_HOST),"read full diagnostic guard");
            Check(std::all_of(guard.begin(),guard.end(),[](unsigned char value){return value==0xff;}),"Full diagnostic guard overwritten");}
        resources.CheckGuards();Check(t.Close() && extra.Close(),"Diagnostic cleanup failed");
        std::ofstream proof(path/"tree_check.json");proof<<"{\"status\":\"all_scores_all_leaf_lists_all_merge_levels_and_strict_gold_ids_passed\",\"max_abs_error\":"<<maxError<<",\"levels\":"<<lists.size()<<"}\n";
        proof.close();Check(bool(proof),"Cannot save tree checks");
    }
    void VerifyIndex() {
        std::vector<unsigned char> chunk(8*1024*1024);
        for(size_t i=0;i<5;++i)for(size_t pos=0;pos<indexBytes[i];) {const auto bytes=std::min(chunk.size(),indexBytes[i]-pos);
            Require(aclrtMemcpy(chunk.data(),bytes,static_cast<unsigned char *>(indexMemory[i])+pos,bytes,ACL_MEMCPY_DEVICE_TO_HOST),"streaming resident index readback");
            Check(std::memcmp(chunk.data(),static_cast<const unsigned char *>(indexHost[i])+pos,bytes)==0,"Index/directory changed");pos+=bytes;}
        resources.CheckGuards();
    }
    const Plan &GetPlan() const {return plan;}
    void Close() {if(!closed){Check(resources.Close(),"Large plan cleanup failed");closed=true;}}
};
}
