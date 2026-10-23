// Isolated clock diagnostic. Uses actual generated ABI; never replaces the Search executable.
#include "fused_runtime.h"
#if !defined(BM25_PHASE_CMAKE_SHA) || !defined(BM25_PHASE_RUNNER_SHA) || !defined(BM25_PHASE_RUNTIME_SHA)
#error Build using the phase diagnostic CMake project.
#endif

namespace {
constexpr int64_t k = 10, shards = 32;
constexpr int warmup = 5, repeats = 10;
constexpr size_t guard = 16;

struct Api {
    ApiLibrary fused, merge;
    FusedPrepare prepare;
    MergePrepare prepareMerge;
    Execute execute, executeMerge;
    explicit Api(const fs::path &vendors) : fused(vendors/"bm25_fused_score_topk"), merge(vendors/"bm25_topk_merge")
    {
        prepare=fused.Symbol<FusedPrepare>("aclnnBm25FusedScoreTopKGetWorkspaceSize");
        execute=fused.Symbol<Execute>("aclnnBm25FusedScoreTopK");
        prepareMerge=merge.Symbol<MergePrepare>("aclnnBm25TopKMergeGetWorkspaceSize");
        executeMerge=merge.Symbol<Execute>("aclnnBm25TopKMerge");
    }
};

template<class T> std::vector<T> Download(void *memory,size_t words,size_t count)
{
    static_assert(sizeof(T)==4 || sizeof(T)==8);
    Check(Bytes(count,sizeof(T))<=Bytes(words,4),"Download exceeds output capacity");
    std::vector<uint32_t> raw(words);
    Require(aclrtMemcpy(raw.data(),Bytes(words,4),memory,Bytes(words,4),ACL_MEMCPY_DEVICE_TO_HOST),"download output and guard");
    for(size_t i=Bytes(count,sizeof(T))/4;i<words;++i) Check(raw[i]==UINT32_MAX,"Output tail guard changed");
    std::vector<T> result(count);
    if(count) std::memcpy(result.data(),raw.data(),Bytes(count,sizeof(T)));
    return result;
}

struct Trace { int cycle; std::string label; std::vector<uint64_t> ticks; };

void SaveRanking(const fs::path &dir,const Request &r)
{
    Check(fs::create_directories(dir),"Output directory already exists");
    Write(dir/"scores.bin",r.actual);
    Write(dir/"partial_scores.bin",r.partialScores); Write(dir/"partial_ids.bin",r.partialIds);
    Write(dir/"top_scores.bin",r.topScores); Write(dir/"top_ids.bin",r.topIds);
}
}

int main(int argc,char **argv)
{
    try {
        if(argc!=6) { std::cerr<<"Usage: bm25_fused_phase_runner INDEX PLAN OUTPUT DEVICE VENDORS\n"; return 2; }
        static_assert(sizeof(float)==4 && sizeof(int32_t)==4 && sizeof(uint64_t)==8 && std::numeric_limits<float>::is_iec559);
        const uint16_t endian=1; Check(*reinterpret_cast<const unsigned char *>(&endian)==1,"Little-endian required");
        const auto output=fs::absolute(argv[3]); Check(!fs::exists(output),"Output directory already exists");
        const int device=static_cast<int>(Number(argv[4],0,INT32_MAX));
        Stage("validate_current_workload_and_plan");
        const Index index(fs::absolute(argv[1])); const size_t docs=index.norm.size();
        auto requests=LoadRequests(fs::absolute(argv[2]),docs);
        Check(docs==262144 && requests.size()==2 && requests[0].label=="A" && requests[1].label=="B" &&
            requests[0].Queries()==8 && requests[1].Queries()==8,"Only current 256K/Q8/K10/S32 A/B diagnostic supported");
        Api api(fs::absolute(argv[5])); Runtime runtime; runtime.device=device;
        Stage("runtime_init"); Require(aclInit(nullptr),"aclInit"); runtime.initialized=true;
        Require(aclrtSetDevice(device),"aclrtSetDevice"); runtime.selected=true;
        Require(aclrtCreateContext(&runtime.context,device),"aclrtCreateContext");
        aclrtRunMode mode; Require(aclrtGetRunMode(&mode),"aclrtGetRunMode"); Check(mode==ACL_HOST,"ACL_HOST required");
        Require(aclrtCreateStream(&runtime.stream),"aclrtCreateStream");
        Buffers buffers; buffers.stream=runtime.stream; Tensors tensors; tensors.stream=runtime.stream;
        const void *host[5]={index.offsets.data(),index.docs.data(),index.tf.data(),index.norm.data(),index.idf.data()};
        const size_t counts[5]={index.offsets.size(),index.docs.size(),index.tf.size(),index.norm.size(),index.idf.size()};
        const size_t widths[5]={8,4,4,4,4};
        const aclDataType types[5]={ACL_INT64,ACL_INT32,ACL_FLOAT,ACL_FLOAT,ACL_FLOAT};
        void *resident[5]; aclTensor *inputs[5]; uint64_t indexBytes=0;
        Stage("allocate_and_upload_index_once");
        for(size_t i=0;i<5;++i) {
            const size_t bytes=Bytes(counts[i],widths[i]); indexBytes+=bytes;
            resident[i]=buffers.Allocate(bytes); inputs[i]=tensors.Describe1D(counts[i],types[i],resident[i]);
            Require(aclrtMemcpy(resident[i],bytes,host[i],bytes,ACL_MEMCPY_HOST_TO_DEVICE),"index upload once");
        }
        const size_t q=8, tasks=q*shards;
        // Separate full-score and clock outputs: no aliasing or writes beyond a described tensor.
        const size_t capacity[6]={q*docs+guard,tasks*k+guard,tasks*k+guard,q*k+guard,q*k+guard,tasks*8+guard};
        void *out[6]; size_t maxWords=0;
        for(size_t i=0;i<6;++i) { out[i]=buffers.Allocate(Bytes(capacity[i],4)); maxWords=std::max(maxWords,capacity[i]); }
        std::vector<uint32_t> poison(maxWords,UINT32_MAX);
        const size_t maxTerms=std::max<size_t>({1,requests[0].terms.size(),requests[1].terms.size()});
        void *offsets=buffers.Allocate(Bytes(q+1,8)), *words=buffers.Allocate(Bytes(maxTerms,4));
        void *workspace[2][2]={}; uint64_t workspaceCapacity[2][2]={};
        std::vector<Trace> traces;
        auto call=[&](Request &r,int debug,int cycle,bool saveTrace) {
            Check(debug==1 || debug==2,"Diagnostic mode must be 1 or 2");
            // Keep the large disabled full-score buffer untouched throughout warmup/recording.
            // It is poisoned/checked in preflight and checked again after ALL recorded calls;
            // no 8 MiB debug-score DMA is inserted between the measured kernels.
            for(size_t i=0;i<6;++i) if(i!=0 || debug==1 || cycle<0)
                Require(aclrtMemcpy(out[i],Bytes(capacity[i],4),poison.data(),Bytes(capacity[i],4),ACL_MEMCPY_HOST_TO_DEVICE),"poison outputs");
            Tensors current; current.stream=runtime.stream;
            auto *qo=current.Describe1D(q+1,ACL_INT64,offsets);
            auto *qt=current.Describe1D(std::max<size_t>(1,r.terms.size()),ACL_INT32,words);
            auto *ps=Describe3D(current,q,shards,k,ACL_FLOAT,out[1]);
            auto *pi=Describe3D(current,q,shards,k,ACL_INT32,out[2]);
            auto *ts=current.Describe(q,k,ACL_FLOAT,out[3]);
            auto *ti=current.Describe(q,k,ACL_INT32,out[4]);
            auto *extra=debug==1?current.Describe(q,docs,ACL_FLOAT,out[0]):current.Describe1D(tasks*8,ACL_FLOAT,out[5]);
            const int32_t dummy=0;
            Require(aclrtMemcpy(offsets,Bytes(q+1,8),r.offsets.data(),Bytes(q+1,8),ACL_MEMCPY_HOST_TO_DEVICE),"upload query offsets");
            Require(aclrtMemcpy(words,Bytes(maxTerms,4),r.terms.empty()?static_cast<const void *>(&dummy):r.terms.data(),
                Bytes(std::max<size_t>(1,r.terms.size()),4),ACL_MEMCPY_HOST_TO_DEVICE),"upload query IDs");
            uint64_t need[2]={}; aclOpExecutor *executor[2]={};
            Require(api.prepare(inputs[0],inputs[1],inputs[2],inputs[3],inputs[4],qo,qt,k,shards,debug,
                ps,pi,extra,&need[0],&executor[0]),"prepare fused diagnostic");
            Require(api.prepareMerge(ps,pi,k,docs,ts,ti,&need[1],&executor[1]),"prepare merge");
            for(size_t i=0;i<2;++i) {
                Check(executor[i] && need[i]<=std::numeric_limits<size_t>::max(),"Invalid executor/workspace");
                const size_t slot=debug-1;
                if(!workspace[slot][i] && !workspaceCapacity[slot][i]) {
                    workspaceCapacity[slot][i]=need[i]; workspace[slot][i]=buffers.Allocate(need[i]);
                }
                Check(need[i]<=workspaceCapacity[slot][i],"Workspace grew beyond initial capacity");
            }
            Require(api.execute(workspace[debug-1][0],need[0],executor[0],runtime.stream),"execute fused diagnostic");
            Require(api.executeMerge(workspace[debug-1][1],need[1],executor[1],runtime.stream),"execute merge");
            Require(aclrtSynchronizeStream(runtime.stream),"synchronize diagnostic chain");
            auto partial=Download<float>(out[1],capacity[1],tasks*k);
            auto local=Download<int32_t>(out[2],capacity[2],tasks*k);
            auto top=Download<float>(out[3],capacity[3],q*k);
            auto ids=Download<int32_t>(out[4],capacity[4],q*k);
            if(debug==1) {
                r.actual=Download<float>(out[0],capacity[0],q*docs);
                Download<uint32_t>(out[5],capacity[5],0);
                r.partialScores=std::move(partial); r.partialIds=std::move(local);
                r.topScores=std::move(top); r.topIds=std::move(ids);
                CompareReference(r); CheckRanking(r,docs,k,shards);
            } else {
                Check(Same(partial,r.partialScores) && Same(local,r.partialIds) && Same(top,r.topScores) && Same(ids,r.topIds),
                    "Clock mode changed local/final IDs or FP32 score bits");
                if(cycle<0) Download<uint32_t>(out[0],capacity[0],0);
                auto ticks=Download<uint64_t>(out[5],capacity[5],tasks*4);
                for(size_t task=0;task<tasks;++task) {
                    const auto *v=ticks.data()+task*4;
                    Check(v[0]!=UINT64_MAX && v[0]<=v[1] && v[1]<v[2] && v[2]<v[3] && v[3]!=UINT64_MAX,
                        "Missing, wrapped or unordered per-task SIMT clock record");
                }
                if(saveTrace) traces.push_back({cycle,r.label,std::move(ticks)});
            }
            Check(current.Close(),"Diagnostic descriptor cleanup failed");
            std::cout<<"phase_call_checked="<<r.label<<" debug="<<debug<<" cycle="<<cycle<<std::endl;
        };
        Stage("debug1_full_scores_and_strict_rankings");
        for(auto &r:requests) call(r,1,-2,false);
        Stage("clock_mode_preflight_and_warmup");
        for(auto &r:requests) call(r,2,-1,false);
        for(int cycle=0;cycle<warmup;++cycle) for(int pos=0;pos<2;++pos)
            call(requests[cycle%2?1-pos:pos],2,cycle,false);
        Stage("record_phase_cycles_alternating_AB_BA");
        for(int cycle=0;cycle<repeats;++cycle) for(int pos=0;pos<2;++pos)
            call(requests[cycle%2?1-pos:pos],2,cycle,true);
        Stage("check_disabled_full_score_buffer_after_all_clock_calls");
        Download<uint32_t>(out[0],capacity[0],0);
        Stage("verify_resident_index_readback_unchanged");
        for(size_t i=0;i<5;++i) {
            const size_t bytes=Bytes(counts[i],widths[i]); std::vector<unsigned char> raw(bytes);
            Require(aclrtMemcpy(raw.data(),bytes,resident[i],bytes,ACL_MEMCPY_DEVICE_TO_HOST),"index readback");
            Check(std::memcmp(raw.data(),host[i],bytes)==0,"Resident index changed");
        }
        Stage("release_resources"); Check(tensors.Close(),"Static tensor cleanup failed");
        Check(buffers.Close(),"Buffer cleanup failed"); Check(runtime.Close(),"Runtime cleanup failed");
        Stage("save_checked_diagnostic_outputs"); Check(fs::create_directories(output),"Cannot create fresh output");
        for(const auto &r:requests) SaveRanking(output/r.label,r);
        for(const auto &trace:traces) Write(output/("cycle"+std::to_string(trace.cycle)+"_"+trace.label+".bin"),trace.ticks);
        std::ofstream report(output/"device_session.json");
        report<<"{\"status\":\"phase_cycles_outputs_and_guards_checked\",\"device\":"<<device
            <<",\"queries\":8,\"docs\":262144,\"k\":10,\"shards\":32,\"warmup\":5,\"repeats\":10"
            <<",\"index_upload_batches\":1,\"index_tensor_h2d_calls\":5,\"index_upload_bytes\":"<<indexBytes
            <<",\"index_readback_byte_identical\":true,\"debug1_calls\":2,\"clock_mode_calls\":32,\"saved_trace_calls\":20"
            <<",\"every_clock_call_rank_bits_match_debug1\":true,\"all_output_guards_passed\":true"
            <<",\"full_score_guard_check_policy\":\"preflight_and_after_all_clock_calls\""
            <<",\"measured_calls_full_score_poison_or_download\":false"
            <<",\"host_source_sha256\":{\"npu_prototype/fused_phase_host/CMakeLists.txt\":\""<<BM25_PHASE_CMAKE_SHA
            <<"\",\"npu_prototype/fused_phase_host/phase_runner.cpp\":\""<<BM25_PHASE_RUNNER_SHA
            <<"\",\"npu_prototype/fused_chain_host/fused_runtime.h\":\""<<BM25_PHASE_RUNTIME_SHA<<"\"}}\n";
        report.close(); Check(static_cast<bool>(report),"Cannot save diagnostic session");
        std::cout<<"Clock diagnostic completed; cycle counts only, not uninstrumented kernel/Search latency or speedup.\n";
        return 0;
    } catch(const std::exception &e) { std::cerr<<"FAILED: "<<e.what()<<"; no CPU output substituted.\n"; return 1; }
}
