// Actual generated fused ABI reviewed 2026-10-21. Device diagnostic only; no CPU scoring fallback.
#include "fused_runtime.h"
#if !defined(BM25_FUSED_CMAKE_SHA) || !defined(BM25_FUSED_RUNTIME_SHA) || !defined(BM25_FUSED_RUNNER_SHA)
#error Use this directory's CMake project to record compiled source fingerprints.
#endif

struct FusedApi {
    ApiLibrary fused, merge;
    FusedPrepare prepareFused;
    MergePrepare prepareMerge;
    Execute executeFused, executeMerge;
    explicit FusedApi(const fs::path &vendors)
        : fused(vendors/"bm25_fused_score_topk"), merge(vendors/"bm25_topk_merge")
    {
        prepareFused = fused.Symbol<FusedPrepare>("aclnnBm25FusedScoreTopKGetWorkspaceSize");
        executeFused = fused.Symbol<Execute>("aclnnBm25FusedScoreTopK");
        prepareMerge = merge.Symbol<MergePrepare>("aclnnBm25TopKMergeGetWorkspaceSize");
        executeMerge = merge.Symbol<Execute>("aclnnBm25TopKMerge");
    }
};

template<class T> std::vector<T> Download(void *memory,size_t capacity,size_t count,const char *name)
{
    static_assert(sizeof(T) == 4);
    Check(Bytes(count,4) <= capacity,"Download exceeds capacity");
    std::vector<uint32_t> raw(capacity/4);
    Require(aclrtMemcpy(raw.data(),capacity,memory,capacity,ACL_MEMCPY_DEVICE_TO_HOST),name);
    for (size_t i=count;i<raw.size();++i) Check(raw[i] == UINT32_MAX,"Output wrote beyond its current logical shape");
    std::vector<T> result(count);
    if (count) std::memcpy(result.data(),raw.data(),Bytes(count,4));
    return result;
}
uint32_t CheckPoison(void *memory,size_t capacity,bool zeroFirst)
{
    const auto raw = Download<uint32_t>(memory,capacity,capacity/4,"download diagnostic guard");
    if (zeroFirst) Check(raw[0] == 0U,"debug=0 placeholder was not overwritten with positive zero");
    for (size_t i=zeroFirst?1:0;i<raw.size();++i)
        Check(raw[i] == UINT32_MAX,"Disabled debug changed full-score output or placeholder guard");
    return raw[0];
}
void CheckModes(const Request &r)
{
    Check(Same(r.partialScores,r.disabled.partialScores) && Same(r.partialIds,r.disabled.partialIds) &&
          Same(r.topScores,r.disabled.topScores) && Same(r.topIds,r.disabled.topIds) &&
          Same(r.groupScores,r.disabled.groupScores) && Same(r.groupIds,r.disabled.groupIds),
          "debug1/debug0 candidate/group/final result bytes differ");
}
void SaveArrays(const fs::path &dir,const std::vector<float> &partial,const std::vector<int32_t> &ids,
                const std::vector<float> &top,const std::vector<int32_t> &topIds,
                const std::vector<float> &group,const std::vector<int32_t> &groupIds)
{
    Check(fs::create_directories(dir),"Cannot create fresh mode output directory");
    Write(dir/"partial_scores.bin",partial); Write(dir/"partial_ids.bin",ids);
    Write(dir/"top_scores.bin",top); Write(dir/"top_ids.bin",topIds);
    if (!group.empty()) { Write(dir/"group_scores.bin",group); Write(dir/"group_ids.bin",groupIds); }
}
void SaveReport(const fs::path &out,const std::vector<Request> &requests,uint64_t indexBytes,int device,
                int64_t docs,int64_t k,int64_t shards,size_t poisonBytes,size_t guardBytes)
{
    const int stages = shards == 64 ? 3 : 2;
    std::ofstream file(out/"device_session.json");
    file << "{\n\"status\":\"device_fused_chain_debug_modes_and_rankings_checked\",\n"
         << "\"device\":" << device << ",\"docs\":" << docs << ",\"k\":" << k << ",\"shards\":" << shards
         << ",\"pipeline_stages_per_mode\":" << stages << ",\"intermediate_host_roundtrips\":0,\n"
         << "\"index_upload_batches\":1,\"index_tensor_h2d_calls\":5,\"index_upload_bytes\":" << indexBytes
         << ",\"index_readback_byte_identical\":true,\"index_device_allocations_reused\":true,\n"
         << "\"debug_index_d2h_calls\":5,\"debug_index_d2h_bytes\":" << indexBytes << ",\n"
         << "\"scope\":\"correctness only: poison, full-score and guard downloads; no Search timing or speedup\",\n"
         << "\"transfer_counter_scope\":\"explicit application copies only; no claim about framework internals\",\n"
         << "\"host_source_sha256\":{\n"
         << "\"npu_prototype/fused_chain_host/CMakeLists.txt\":\"" << BM25_FUSED_CMAKE_SHA << "\",\n"
         << "\"npu_prototype/fused_chain_host/fused_runtime.h\":\"" << BM25_FUSED_RUNTIME_SHA << "\",\n"
         << "\"npu_prototype/fused_chain_host/fused_chain_runner.cpp\":\"" << BM25_FUSED_RUNNER_SHA << "\"},\n"
         << "\"requests\":[\n";
    for (size_t i=0;i<requests.size();++i) {
        const auto &r = requests[i];
        const size_t logical = Bytes(r.offsets.size(),8)+Bytes(r.terms.size(),4);
        file << (i?",\n":"") << "{\"label\":\"" << r.label << "\",\"queries\":" << r.Queries()
             << ",\"logical_word_count\":" << r.terms.size() << ",\"mode_order\":[1,0]"
             << ",\"query_h2d_bytes_per_mode\":" << logical+(r.terms.empty()?4:0)
             << ",\"query_tensor_h2d_calls_per_mode\":2,\"fresh_executors_per_mode\":" << stages
             << ",\"chain_synchronizations_per_mode\":1,\"tensor_cleanup_synchronizations_per_mode\":1"
             << ",\"debug_poison_h2d_calls_per_mode\":" << (shards==64?8:6)
             << ",\"debug_poison_h2d_bytes_per_mode\":" << poisonBytes << ",\"placeholder_guard_bytes\":" << guardBytes
             << ",\"debug_enabled_score_elements\":" << r.actual.size() << ",\"debug_disabled_score_elements\":1"
             << ",\"outputs_poisoned\":true,\"logical_tail_guards_unchanged\":true"
             << ",\"disabled_full_buffer_unchanged\":true,\"disabled_placeholder_guard_unchanged\":true"
             << ",\"debug_modes_all_candidate_group_topk_bits_equal\":true,\"full_reference_checked\":true"
             << ",\"exact_device_ranking_checked\":true,\"repeated_all_output_bits_checked\":" << (r.repeated?"true":"false")
             << ",\"workspace_bytes_by_mode\":[";
        for (int mode=0;mode<2;++mode) {
            file << (mode?",[":"[");
            for (int stage=0;stage<stages;++stage) file << (stage?",":"") << r.workspace[mode][stage];
            file << "]";
        }
        file << "]}";
    }
    file << "\n]}\n"; file.close(); Check(static_cast<bool>(file),"Cannot save fused session report");
}

int main(int argc,char **argv)
{
    try {
        if (argc == 3 && std::string(argv[1]) == "--check-libraries") {
            FusedApi api(fs::absolute(argv[2]));
            std::cout << "Both libraries and all four reviewed API symbol owners verified. No device work submitted.\n";
            return 0;
        }
        if (argc != 8) {
            std::cerr << "Usage: bm25_fused_chain_runner INDEX_DIR PLAN_TSV OUTPUT_DIR DEVICE VENDORS K SHARDS\n"; return 2;
        }
        static_assert(sizeof(float)==4 && sizeof(int32_t)==4 && sizeof(int64_t)==8 && std::numeric_limits<float>::is_iec559);
        const uint16_t endian=1; Check(*reinterpret_cast<const unsigned char *>(&endian)==1,"Little-endian required");
        const auto output=fs::absolute(argv[3]); Check(!fs::exists(output),"Output already exists");
        const int device=static_cast<int>(Number(argv[4],0,INT32_MAX));
        const int64_t k=Number(argv[6],1,32), shards=Number(argv[7],1,64);
        Check(shards<=32 || shards==64,"Only shards1..32 or64 supported");
        Stage("validate_static_index_and_request_plan");
        const Index index(fs::absolute(argv[1])); const int64_t docs=static_cast<int64_t>(index.norm.size());
        Check(docs%shards==0 && docs/shards>=k && docs/shards<=8192,"Unsupported fused shard width");
        auto requests=LoadRequests(fs::absolute(argv[2]),index.norm.size());
        size_t maxQueries=0,maxWords=1;
        for (const auto &r:requests) { maxQueries=std::max(maxQueries,static_cast<size_t>(r.Queries())); maxWords=std::max(maxWords,r.terms.size()); }
        FusedApi api(fs::absolute(argv[5])); Runtime runtime; runtime.device=device;
        Stage("runtime_init"); Require(aclInit(nullptr),"aclInit"); runtime.initialized=true;
        Require(aclrtSetDevice(device),"aclrtSetDevice"); runtime.selected=true;
        Require(aclrtCreateContext(&runtime.context,device),"aclrtCreateContext");
        aclrtRunMode mode; Require(aclrtGetRunMode(&mode),"aclrtGetRunMode"); Check(mode==ACL_HOST,"Requires ACL_HOST");
        Require(aclrtCreateStream(&runtime.stream),"aclrtCreateStream");
        Buffers buffers; buffers.stream=runtime.stream; Tensors tensors; tensors.stream=runtime.stream;
        const void *host[5]={index.offsets.data(),index.docs.data(),index.tf.data(),index.norm.data(),index.idf.data()};
        const size_t counts[5]={index.offsets.size(),index.docs.size(),index.tf.size(),index.norm.size(),index.idf.size()};
        const size_t widths[5]={8,4,4,4,4}; const aclDataType types[5]={ACL_INT64,ACL_INT32,ACL_FLOAT,ACL_FLOAT,ACL_FLOAT};
        void *indexMemory[5]; aclTensor *inputs[5]; uint64_t indexBytes=0;
        Stage("allocate_and_upload_index_once");
        for (size_t i=0;i<5;++i) {
            const size_t bytes=Bytes(counts[i],widths[i]); indexBytes+=bytes;
            indexMemory[i]=buffers.Allocate(bytes); inputs[i]=tensors.Describe1D(counts[i],types[i],indexMemory[i]);
            Require(aclrtMemcpy(indexMemory[i],bytes,host[i],bytes,ACL_MEMCPY_HOST_TO_DEVICE),"index upload once");
        }
        const size_t capacities[8]={Bytes(maxQueries*docs,4),Bytes(maxQueries*shards*k,4),Bytes(maxQueries*shards*k,4),
            Bytes(maxQueries*k,4),Bytes(maxQueries*k,4),shards==64?Bytes(maxQueries*2*k,4):0,
            shards==64?Bytes(maxQueries*2*k,4):0,Bytes(maxQueries*docs+16,4)};
        void *outputs[8]; size_t poisonBytes=0,maxCapacity=0;
        for (size_t i=0;i<8;++i) { outputs[i]=buffers.Allocate(capacities[i]); poisonBytes+=capacities[i]; maxCapacity=std::max(maxCapacity,capacities[i]); }
        std::vector<uint32_t> poison(maxCapacity/4,UINT32_MAX);
        void *offsetMemory=buffers.Allocate(Bytes(maxQueries+1,8)), *wordMemory=buffers.Allocate(Bytes(maxWords,4));
        void *workspace[2][3]={}; uint64_t capacity[2][3]={}; const int stages=shards==64?3:2;
        std::map<std::string,size_t> previous;
        for (size_t call=0;call<requests.size();++call) {
            auto &r=requests[call]; const int64_t q=r.Queries();
            for (int debug:{1,0}) {
                std::cout << "request=" << r.label << " queries=" << q << " words=" << r.terms.size() << " debug=" << debug << std::endl;
                Stage("poison_output_capacities_and_guards");
                for (size_t i=0;i<8;++i) if (capacities[i])
                    Require(aclrtMemcpy(outputs[i],capacities[i],poison.data(),capacities[i],ACL_MEMCPY_HOST_TO_DEVICE),"poison output");
                Tensors current; current.stream=runtime.stream;
                auto *offset=current.Describe1D(r.offsets.size(),ACL_INT64,offsetMemory);
                auto *terms=current.Describe1D(std::max<size_t>(1,r.terms.size()),ACL_INT32,wordMemory);
                auto *partialScores=Describe3D(current,q,shards,k,ACL_FLOAT,outputs[1]);
                auto *partialIds=Describe3D(current,q,shards,k,ACL_INT32,outputs[2]);
                auto *debugScores=debug?current.Describe(q,docs,ACL_FLOAT,outputs[0]):current.Describe1D(1,ACL_FLOAT,outputs[7]);
                auto *topScores=current.Describe(q,k,ACL_FLOAT,outputs[3]); auto *topIds=current.Describe(q,k,ACL_INT32,outputs[4]);
                const int32_t dummy=0; const void *words=r.terms.empty()?static_cast<const void *>(&dummy):r.terms.data();
                Stage("upload_query_only");
                Require(aclrtMemcpy(offsetMemory,Bytes(maxQueries+1,8),r.offsets.data(),Bytes(r.offsets.size(),8),ACL_MEMCPY_HOST_TO_DEVICE),"query offsets upload");
                Require(aclrtMemcpy(wordMemory,Bytes(maxWords,4),words,Bytes(std::max<size_t>(1,r.terms.size()),4),ACL_MEMCPY_HOST_TO_DEVICE),"query words upload");
                aclOpExecutor *executors[3]={}; Stage("prepare_fused_and_merge_stages");
                Require(api.prepareFused(inputs[0],inputs[1],inputs[2],inputs[3],inputs[4],offset,terms,k,shards,debug,
                    partialScores,partialIds,debugScores,&r.workspace[debug][0],&executors[0]),"fused prepare");
                if (shards==64) {
                    auto *groupInScores=Describe3D(current,q*2,32,k,ACL_FLOAT,outputs[1]);
                    auto *groupInIds=Describe3D(current,q*2,32,k,ACL_INT32,outputs[2]);
                    auto *groupOutScores=current.Describe(q*2,k,ACL_FLOAT,outputs[5]);
                    auto *groupOutIds=current.Describe(q*2,k,ACL_INT32,outputs[6]);
                    Require(api.prepareMerge(groupInScores,groupInIds,k,docs/2,groupOutScores,groupOutIds,
                        &r.workspace[debug][1],&executors[1]),"group merge prepare");
                    auto *lastInScores=Describe3D(current,q,2,k,ACL_FLOAT,outputs[5]);
                    auto *lastInIds=Describe3D(current,q,2,k,ACL_INT32,outputs[6]);
                    Require(api.prepareMerge(lastInScores,lastInIds,k,docs,topScores,topIds,
                        &r.workspace[debug][2],&executors[2]),"final merge prepare");
                } else Require(api.prepareMerge(partialScores,partialIds,k,docs,topScores,topIds,
                    &r.workspace[debug][1],&executors[1]),"merge prepare");
                for (int stage=0;stage<stages;++stage) {
                    Check(executors[stage] && r.workspace[debug][stage]<=std::numeric_limits<size_t>::max(),"Invalid executor/workspace");
                    if (call==0) { capacity[debug][stage]=r.workspace[debug][stage]; workspace[debug][stage]=buffers.Allocate(capacity[debug][stage]); }
                    else Check(r.workspace[debug][stage]<=capacity[debug][stage],"Workspace grew beyond per-mode capacity");
                }
                Stage("execute_fused_then_device_merges");
                Require(api.executeFused(workspace[debug][0],r.workspace[debug][0],executors[0],runtime.stream),"fused execute"); executors[0]=nullptr;
                for (int stage=1;stage<stages;++stage) {
                    Require(api.executeMerge(workspace[debug][stage],r.workspace[debug][stage],executors[stage],runtime.stream),"merge execute"); executors[stage]=nullptr;
                }
                Stage("synchronize_then_download_and_check_guards"); Require(aclrtSynchronizeStream(runtime.stream),"chain synchronize");
                auto partial=Download<float>(outputs[1],capacities[1],q*shards*k,"download partial scores");
                auto local=Download<int32_t>(outputs[2],capacities[2],q*shards*k,"download partial IDs");
                auto top=Download<float>(outputs[3],capacities[3],q*k,"download top scores");
                auto ids=Download<int32_t>(outputs[4],capacities[4],q*k,"download top IDs");
                std::vector<float> group; std::vector<int32_t> groupIds;
                if (shards==64) { group=Download<float>(outputs[5],capacities[5],q*2*k,"download groups"); groupIds=Download<int32_t>(outputs[6],capacities[6],q*2*k,"download group IDs"); }
                if (debug) {
                    r.actual=Download<float>(outputs[0],capacities[0],q*docs,"download full debug scores");
                    r.partialScores=std::move(partial); r.partialIds=std::move(local); r.topScores=std::move(top); r.topIds=std::move(ids);
                    r.groupScores=std::move(group); r.groupIds=std::move(groupIds); CompareReference(r); CheckRanking(r,docs,k,shards);
                    CheckPoison(outputs[7],capacities[7],false);
                } else {
                    r.disabled.partialScores=std::move(partial); r.disabled.partialIds=std::move(local);
                    r.disabled.topScores=std::move(top); r.disabled.topIds=std::move(ids);
                    r.disabled.groupScores=std::move(group); r.disabled.groupIds=std::move(groupIds);
                    CheckPoison(outputs[0],capacities[0],false);
                    r.disabledPlaceholder=CheckPoison(outputs[7],capacities[7],true); CheckModes(r);
                }
                Check(current.Close(),"Request descriptor cleanup failed");
                std::cout << "mode_passed=" << r.label << " debug=" << debug << " pipeline_stages=" << stages << std::endl;
            }
            const auto key=r.directory.string(); const auto prior=previous.find(key);
            if (prior!=previous.end()) {
                const auto &old=requests[prior->second];
                Check(Same(r.actual,old.actual) && Same(r.partialScores,old.partialScores) && Same(r.partialIds,old.partialIds) &&
                    Same(r.topScores,old.topScores) && Same(r.topIds,old.topIds) && Same(r.groupScores,old.groupScores) && Same(r.groupIds,old.groupIds),
                    "Repeated device output bytes changed"); r.repeated=true;
            } else previous.emplace(key,call);
        }
        Stage("verify_resident_index_readback_unchanged");
        for (size_t i=0;i<5;++i) {
            const size_t bytes=Bytes(counts[i],widths[i]); std::vector<unsigned char> actual(bytes);
            Require(aclrtMemcpy(actual.data(),bytes,indexMemory[i],bytes,ACL_MEMCPY_DEVICE_TO_HOST),"index readback");
            Check(std::memcmp(actual.data(),host[i],bytes)==0,"Resident index changed");
        }
        Stage("release_resources"); Check(tensors.Close(),"Static tensor cleanup failed");
        Check(buffers.Close(),"Buffer cleanup failed"); Check(runtime.Close(),"Runtime cleanup failed");
        Stage("save_checked_outputs"); Check(fs::create_directories(output),"Cannot create output directory");
        for (const auto &r:requests) {
            const auto dir=output/r.label;
            SaveArrays(dir/"debug1",r.partialScores,r.partialIds,r.topScores,r.topIds,r.groupScores,r.groupIds);
            Write(dir/"debug1/scores.bin",r.actual);
            SaveArrays(dir/"debug0",r.disabled.partialScores,r.disabled.partialIds,r.disabled.topScores,r.disabled.topIds,
                r.disabled.groupScores,r.disabled.groupIds);
            Write(dir/"debug0/debug_placeholder.bin",std::vector<uint32_t>{r.disabledPlaceholder});
        }
        SaveReport(output,requests,indexBytes,device,docs,k,shards,poisonBytes,capacities[7]);
        std::cout << "index_upload_batches=1 logical_requests=" << requests.size() << " actual_mode_chains=" << requests.size()*2
                  << " intermediate_host_roundtrips=0\nFused chain correctness diagnostic completed. No Search timing or speedup claim.\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << "FAILED: " << error.what() << "; no CPU output substituted.\n"; return 1; }
}
