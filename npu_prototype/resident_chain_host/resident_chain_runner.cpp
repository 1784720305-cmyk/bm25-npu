// Three existing self-developed kernels, one stream, no intermediate host roundtrip.
#include "resident_chain_common.h"
#if !defined(BM25_CHAIN_CMAKE_SHA) || !defined(BM25_CHAIN_COMMON_SHA) || !defined(BM25_CHAIN_RUNNER_SHA) || !defined(BM25_CHAIN_SHARED_SHA)
#error Use this directory's CMake project to record all compiled source fingerprints.
#endif

void SaveReport(const fs::path &out,const std::vector<Request> &requests,uint64_t indexBytes,
                int device,int64_t docs,int64_t k,int64_t shards,size_t poisonBytes)
{
    std::ofstream file(out/"device_session.json");
    file << "{\n  \"status\": \"device_resident_chain_completed_reference_and_ranking_checked\",\n"
         << "  \"device\": " << device << ", \"docs\": " << docs << ", \"k\": " << k << ", \"shards\": " << shards << ",\n"
         << "  \"pipeline_stages\": 3, \"intermediate_host_roundtrips\": 0,\n"
         << "  \"index_upload_batches\": 1, \"index_tensor_h2d_calls\": 5, \"index_upload_bytes\": " << indexBytes << ",\n"
         << "  \"index_readback_byte_identical\": true, \"index_device_allocations_reused\": true,\n"
         << "  \"debug_index_d2h_calls\": 5, \"debug_index_d2h_bytes\": " << indexBytes << ",\n"
         << "  \"scope\": \"correctness diagnostic with poison/full score/candidate downloads; no search timing\",\n"
         << "  \"transfer_counter_scope\": \"explicit application aclrtMemcpy only; framework internals not profiled\",\n"
         << "  \"host_source_sha256\": {\n"
         << "    \"npu_prototype/resident_chain_host/CMakeLists.txt\": \"" << BM25_CHAIN_CMAKE_SHA << "\",\n"
         << "    \"npu_prototype/resident_chain_host/resident_chain_common.h\": \"" << BM25_CHAIN_COMMON_SHA << "\",\n"
         << "    \"npu_prototype/resident_chain_host/resident_chain_runner.cpp\": \"" << BM25_CHAIN_RUNNER_SHA << "\",\n"
         << "    \"npu_prototype/query_score_host/query_score_runtime.h\": \"" << BM25_CHAIN_SHARED_SHA << "\"\n  },\n  \"requests\": [\n";
    for (size_t i = 0; i < requests.size(); ++i) {
        const auto &r = requests[i];
        const size_t logical = Bytes(r.offsets.size(),8)+Bytes(r.terms.size(),4);
        file << (i?",\n":"") << "    {\"label\": \"" << r.label << "\", \"queries\": " << r.Queries()
             << ", \"logical_word_count\": " << r.terms.size() << ", \"logical_query_bytes\": " << logical
             << ", \"query_h2d_bytes\": " << logical+(r.terms.empty()?4:0) << ", \"query_tensor_h2d_calls\": 2"
             << ", \"topk_d2h_bytes\": " << Bytes(r.topScores.size(),8) << ", \"topk_d2h_calls\": 2"
             << ", \"debug_score_and_candidate_d2h_bytes\": " << Bytes(r.actual.size(),4)+Bytes(r.partialScores.size(),8)
             << ", \"debug_score_and_candidate_d2h_calls\": 3"
             << ", \"debug_poison_h2d_calls\": 5, \"debug_poison_h2d_bytes\": " << poisonBytes
             << ", \"workspace_bytes\": [" << r.workspace[0] << "," << r.workspace[1] << "," << r.workspace[2] << "]"
             << ", \"outputs_poisoned\": true, \"fresh_executors\": 3, \"chain_synchronizations\": 1"
             << ", \"debug_tensor_cleanup_synchronizations\": 1"
             << ", \"full_reference_checked\": true, \"exact_device_ranking_checked\": true"
             << ", \"repeated_all_output_bits_checked\": " << (r.repeated?"true":"false") << "}";
    }
    file << "\n  ]\n}\n"; file.close(); Check(static_cast<bool>(file),"Cannot save chain report");
}

int main(int argc,char **argv)
{
    try {
        if (argc == 3 && std::string(argv[1]) == "--check-libraries") {
            ChainApi api(fs::absolute(argv[2]));
            std::cout << "All six actual generated API symbol owners checked. No device work submitted.\n";
            return 0;
        }
        if (argc != 9) {
            std::cerr << "Usage: bm25_resident_chain_runner INDEX_DIR PLAN_TSV OUTPUT_DIR DEVICE VENDORS K SHARDS ACCEPTED_TOPK_WORKSPACE\n";
            return 2;
        }
        static_assert(sizeof(float) == 4 && sizeof(int32_t) == 4 && sizeof(int64_t) == 8);
        static_assert(std::numeric_limits<float>::is_iec559);
        const uint16_t endian = 1;
        Check(*reinterpret_cast<const unsigned char *>(&endian) == 1,"Requires little-endian host");
        const auto output = fs::absolute(argv[3]);
        Check(!fs::exists(output),"Output exists; choose a new directory");
        const int device = static_cast<int>(Number(argv[4],0,INT32_MAX));
        const int64_t k = Number(argv[6],1,32), shards = Number(argv[7],1,32);
        const auto acceptedTopWorkspace = static_cast<uint64_t>(Number(argv[8],0,INT64_MAX));
        Stage("validate_static_index_and_request_plan");
        const Index index(fs::absolute(argv[1]));
        const int64_t docs = static_cast<int64_t>(index.norm.size());
        Check(docs%shards == 0 && docs/shards >= k,"Existing merge requires equal shards, each at least K articles");
        auto requests = LoadRequests(fs::absolute(argv[2]),index.norm.size());
        size_t maxQueries = 0, maxWords = 1;
        for (const auto &r : requests) {
            maxQueries = std::max(maxQueries,static_cast<size_t>(r.Queries()));
            maxWords = std::max(maxWords,r.terms.size());
        }
        ChainApi api(fs::absolute(argv[5]));
        Runtime runtime; runtime.device = device;
        Stage("runtime_init");
        Require(aclInit(nullptr),"aclInit"); runtime.initialized = true;
        Require(aclrtSetDevice(device),"aclrtSetDevice"); runtime.selected = true;
        Require(aclrtCreateContext(&runtime.context,device),"aclrtCreateContext");
        aclrtRunMode mode; Require(aclrtGetRunMode(&mode),"aclrtGetRunMode");
        Check(mode == ACL_HOST,"Diagnostic requires ACL_HOST");
        Require(aclrtCreateStream(&runtime.stream),"aclrtCreateStream");
        Buffers buffers; buffers.stream = runtime.stream;
        Tensors tensors; tensors.stream = runtime.stream;
        const void *host[5] = {index.offsets.data(),index.docs.data(),index.tf.data(),index.norm.data(),index.idf.data()};
        const size_t counts[5] = {index.offsets.size(),index.docs.size(),index.tf.size(),index.norm.size(),index.idf.size()};
        const size_t widths[5] = {8,4,4,4,4};
        const aclDataType types[5] = {ACL_INT64,ACL_INT32,ACL_FLOAT,ACL_FLOAT,ACL_FLOAT};
        void *memory[5]; aclTensor *inputs[5]; uint64_t indexBytes = 0;
        Stage("allocate_and_upload_index_once");
        for (size_t i = 0; i < 5; ++i) {
            const size_t bytes = Bytes(counts[i],widths[i]); indexBytes += bytes;
            memory[i] = buffers.Allocate(bytes);
            inputs[i] = tensors.Describe1D(counts[i],types[i],memory[i]);
            Require(aclrtMemcpy(memory[i],bytes,host[i],bytes,ACL_MEMCPY_HOST_TO_DEVICE),"index upload once");
        }
        const size_t capacities[5] = {Bytes(maxQueries*docs,4),Bytes(maxQueries*shards*k,4),
                                     Bytes(maxQueries*shards*k,4),Bytes(maxQueries*k,4),Bytes(maxQueries*k,4)};
        void *outputs[5]; size_t poisonBytes = 0;
        for (size_t i = 0; i < 5; ++i) { outputs[i] = buffers.Allocate(capacities[i]); poisonBytes += capacities[i]; }
        void *offsetMemory = buffers.Allocate(Bytes(maxQueries+1,8));
        void *wordMemory = buffers.Allocate(Bytes(maxWords,4));
        std::vector<uint32_t> poison(maxQueries*docs,UINT32_MAX);
        void *workspace[3] = {nullptr,nullptr,nullptr}; uint64_t capacity[3] = {0,0,0};
        std::map<std::string,size_t> previous;
        for (size_t call = 0; call < requests.size(); ++call) {
            auto &r = requests[call]; const int64_t q = r.Queries();
            std::cout << "request=" << r.label << " queries=" << q << " words=" << r.terms.size()
                      << " k=" << k << " shards=" << shards << std::endl;
            Stage("poison_all_five_output_capacities");
            for (size_t i = 0; i < 5; ++i)
                Require(aclrtMemcpy(outputs[i],capacities[i],poison.data(),capacities[i],ACL_MEMCPY_HOST_TO_DEVICE),"poison output capacity");
            Tensors current; current.stream = runtime.stream;
            auto *offset = current.Describe1D(r.offsets.size(),ACL_INT64,offsetMemory);
            auto *terms = current.Describe1D(std::max<size_t>(1,r.terms.size()),ACL_INT32,wordMemory);
            auto *scores = current.Describe(q,docs,ACL_FLOAT,outputs[0]);
            auto *localInput = current.Describe(q*shards,docs/shards,ACL_FLOAT,outputs[0]);
            auto *partialScores = current.Describe(q*shards,k,ACL_FLOAT,outputs[1]);
            auto *partialIds = current.Describe(q*shards,k,ACL_INT32,outputs[2]);
            auto *mergeScores = Describe3D(current,q,shards,k,ACL_FLOAT,outputs[1]);
            auto *mergeIds = Describe3D(current,q,shards,k,ACL_INT32,outputs[2]);
            auto *topScores = current.Describe(q,k,ACL_FLOAT,outputs[3]);
            auto *topIds = current.Describe(q,k,ACL_INT32,outputs[4]);
            const int32_t dummy = 0;
            Stage("upload_query_only");
            Require(aclrtMemcpy(offsetMemory,Bytes(maxQueries+1,8),r.offsets.data(),Bytes(r.offsets.size(),8),ACL_MEMCPY_HOST_TO_DEVICE),"query offsets upload");
            const void *wordHost = r.terms.empty()?static_cast<const void *>(&dummy):r.terms.data();
            Require(aclrtMemcpy(wordMemory,Bytes(maxWords,4),wordHost,Bytes(std::max<size_t>(1,r.terms.size()),4),ACL_MEMCPY_HOST_TO_DEVICE),"query words upload");
            aclOpExecutor *executors[3] = {nullptr,nullptr,nullptr};
            Stage("prepare_all_three_stages");
            Require(api.score.prepare(inputs[0],inputs[1],inputs[2],inputs[3],inputs[4],offset,terms,scores,&r.workspace[0],&executors[0]),"score prepare");
            Require(api.prepareTop(localInput,k,partialScores,partialIds,&r.workspace[1],&executors[1]),"local Top-K prepare");
            Require(api.prepareMerge(mergeScores,mergeIds,k,docs,topScores,topIds,&r.workspace[2],&executors[2]),"merge prepare");
            Check(r.workspace[1] == acceptedTopWorkspace,"Top-K workspace differs from accepted radix_collect unit cases");
            for (size_t i = 0; i < 3; ++i) {
                Check(executors[i] && r.workspace[i] <= std::numeric_limits<size_t>::max(),"Invalid executor/workspace");
                if (call == 0) { capacity[i] = r.workspace[i]; workspace[i] = buffers.Allocate(static_cast<size_t>(capacity[i])); }
                else Check(r.workspace[i] <= capacity[i],"Workspace grew; stop fixed-capacity diagnostic");
            }
            Stage("execute_score");
            Require(api.score.execute(workspace[0],r.workspace[0],executors[0],runtime.stream),"score execute"); executors[0] = nullptr;
            Stage("execute_local_topk");
            Require(api.executeTop(workspace[1],r.workspace[1],executors[1],runtime.stream),"local Top-K execute"); executors[1] = nullptr;
            Stage("execute_merge");
            Require(api.executeMerge(workspace[2],r.workspace[2],executors[2],runtime.stream),"merge execute"); executors[2] = nullptr;
            Stage("synchronize_whole_chain_once"); Require(aclrtSynchronizeStream(runtime.stream),"chain synchronize");
            r.actual.resize(static_cast<size_t>(q)*docs); r.partialScores.resize(static_cast<size_t>(q)*shards*k);
            r.partialIds.resize(r.partialScores.size()); r.topScores.resize(static_cast<size_t>(q)*k); r.topIds.resize(r.topScores.size());
            Stage("download_topk_then_debug_intermediates");
            auto download = [&](void *dest,size_t bytes,size_t i) { Require(aclrtMemcpy(dest,bytes,outputs[i],bytes,ACL_MEMCPY_DEVICE_TO_HOST),"chain output download"); };
            download(r.topScores.data(),Bytes(r.topScores.size(),4),3); download(r.topIds.data(),Bytes(r.topIds.size(),4),4);
            download(r.actual.data(),Bytes(r.actual.size(),4),0); download(r.partialScores.data(),Bytes(r.partialScores.size(),4),1);
            download(r.partialIds.data(),Bytes(r.partialIds.size(),4),2);
            CompareReference(r); CheckRanking(r,docs,k,shards);
            const auto key = r.directory.string(); const auto prior = previous.find(key);
            if (prior != previous.end()) {
                const auto &old = requests[prior->second];
                Check(Same(r.actual,old.actual) && Same(r.partialScores,old.partialScores) && Same(r.partialIds,old.partialIds) &&
                      Same(r.topScores,old.topScores) && Same(r.topIds,old.topIds),"Repeated device output bits changed");
                r.repeated = true;
            } else previous.emplace(key,call);
            Check(current.Close(),"Request tensor cleanup failed");
            std::cout << "request_passed=" << r.label << " full_scores=" << r.actual.size() << " exact_topk=" << r.topIds.size() << std::endl;
        }
        Stage("verify_resident_index_readback_unchanged");
        for (size_t i = 0; i < 5; ++i) {
            const size_t bytes = Bytes(counts[i],widths[i]); std::vector<unsigned char> actual(bytes);
            Require(aclrtMemcpy(actual.data(),bytes,memory[i],bytes,ACL_MEMCPY_DEVICE_TO_HOST),"index debug readback");
            Check(std::memcmp(actual.data(),host[i],bytes) == 0,"Resident index changed");
        }
        Stage("release_resources"); Check(tensors.Close(),"Static tensor cleanup failed");
        Check(buffers.Close(),"Buffer cleanup failed"); Check(runtime.Close(),"Runtime cleanup failed");
        Stage("save_checked_outputs"); Check(fs::create_directories(output),"Cannot create output directory");
        for (const auto &r : requests) {
            const auto dir = output/r.label; Check(fs::create_directory(dir),"Cannot create request output directory");
            Write(dir/"scores.bin",r.actual); Write(dir/"partial_scores.bin",r.partialScores); Write(dir/"partial_ids.bin",r.partialIds);
            Write(dir/"top_scores.bin",r.topScores); Write(dir/"top_ids.bin",r.topIds);
        }
        SaveReport(output,requests,indexBytes,device,docs,k,shards,poisonBytes);
        std::cout << "index_upload_batches=1 requests=" << requests.size() << " pipeline_stages=3 intermediate_host_roundtrips=0\n"
                  << "Resident-index full-ranking correctness diagnostic completed. No search timing or speedup claim.\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << "FAILED: " << error.what() << "; no CPU output substituted.\n"; return 1; }
}
