// First query-ID Search correctness/measurement-boundary diagnostic. Not a steady-state benchmark.
#include "npu_search.h"
#include "compiled_search_sources.h"
#include <array>
#include <iomanip>

struct RawRequest { std::string label; fs::path source; bm25search::Queries queries; };
std::vector<RawRequest> LoadPlan(const fs::path &plan)
{
    std::ifstream file(plan); Check(file.is_open(),"Cannot read raw query plan");
    std::vector<RawRequest> result; std::map<std::string,bool> labels; std::string line;
    while (std::getline(file,line)) {
        if (!line.empty() && line.back()=='\r') line.pop_back();
        const auto pos=line.find('\t');
        Check(pos!=std::string::npos && pos>0 && line.find('\t',pos+1)==std::string::npos,"Plan requires label<TAB>raw-query-file");
        RawRequest request; request.label=line.substr(0,pos);
        Check(std::all_of(request.label.begin(),request.label.end(),[](unsigned char c){return std::isalnum(c)||c=='_';}) &&
              labels.emplace(request.label,true).second,"Unsafe/duplicate request label");
        request.source=fs::canonical(line.substr(pos+1)); request.queries=bm25search::LoadQueries(request.source.string());
        result.push_back(std::move(request)); Check(result.size()<=32,"Diagnostic exceeds 32 requests");
    }
    Check(file.eof() && !result.empty(),"Empty/unreadable query plan"); return result;
}

int main(int argc,char **argv)
{
    try {
        if (argc==3 && std::string(argv[1])=="--check-libraries") {
            ChainApi api(fs::absolute(argv[2]));
            std::cout << "All six actual generated API symbol owners checked. No device work submitted.\n"; return 0;
        }
        if (argc!=11) {
            std::cerr << "Usage: bm25_search_runner INDEX PLAN OUTPUT DEVICE VENDORS K SHARDS MAX_QUERIES MAX_WORDS TOPK_WORKSPACE\n"; return 2;
        }
        const auto out=fs::absolute(argv[3]); Check(!fs::exists(out),"Output exists; choose a fresh directory");
        const int device=static_cast<int>(Number(argv[4],0,INT32_MAX));
        const auto k=Number(argv[6],1,32), shards=Number(argv[7],1,32);
        const auto capQ=Number(argv[8],1,10000), capWords=Number(argv[9],0,16777216);
        const auto topWorkspace=Number(argv[10],0,INT64_MAX);
        static_assert(sizeof(float)==4 && sizeof(int32_t)==4 && sizeof(int64_t)==8 && std::numeric_limits<float>::is_iec559);
        const uint16_t endian=1; Check(*reinterpret_cast<const unsigned char *>(&endian)==1,"Requires little-endian host");
        Stage("load_public_raw_query_ids_before_session");
        const auto requests=LoadPlan(fs::absolute(argv[2])); // IDs grouped in CPU memory, no backend offsets/flat words.
        const auto coldStart=bm25search::Clock::now();
        const Index index(fs::absolute(argv[1]));
        bm25search::NpuSearchSession session(index,fs::absolute(argv[5]),device,k,shards,capQ,capWords,topWorkspace);
        const double coldMs=std::chrono::duration<double,std::milli>(bm25search::Clock::now()-coldStart).count();
        std::array<uint64_t,3> workspaces{};
        for (size_t i=0;i<3;++i) workspaces[i]=session.WorkspaceCapacities()[i];
        Check(fs::create_directories(out),"Cannot create new output");
        std::vector<bm25search::Result> results;
        std::map<std::string,size_t> previous;
        for (size_t i=0;i<requests.size();++i) {
            const auto &r=requests[i];
            Stage("diagnostic_poison_outside_Search"); session.PoisonForDiagnostic();
            auto result=session.Search(r.queries); // Entire timed public search, no expected results passed in.
            const auto directory=out/r.label; Check(fs::create_directory(directory),"Cannot create request output");
            Stage("diagnostic_download_and_save_outside_Search");
            Write(directory/"top_scores.bin",result.scores); Write(directory/"top_ids.bin",result.ids);
            session.DownloadDebug(r.queries.size(),directory);
            const auto prior=previous.find(r.source.string());
            if (prior!=previous.end()) {
                const auto &old=results[prior->second];
                Check(Same(result.scores,old.scores) && Same(result.ids,old.ids),"Repeated Search Top-K bits changed");
            } else previous.emplace(r.source.string(),i);
            std::cout << "search_completed=" << r.label << " queries=" << r.queries.size()
                      << " search_call_ms=" << std::setprecision(17) << result.searchMs << std::endl;
            results.push_back(std::move(result));
        }
        Stage("diagnostic_index_readback_outside_Search"); session.VerifyIndex();
        session.Close();
        std::ofstream file(out/"search_session.json"); file << std::setprecision(17);
        file << "{\n\"status\":\"device_search_calls_completed_independent_check_pending\",\n"
             << "\"device\":" << device << ",\"docs\":" << index.norm.size() << ",\"k\":" << k << ",\"shards\":" << shards << ",\n"
             << "\"capacity_queries\":" << capQ << ",\"capacity_words\":" << capWords << ",\n"
             << "\"index_upload_batches\":1,\"index_tensor_h2d_calls\":5,\"index_upload_bytes\":" << session.indexUploadBytes << ",\n"
             << "\"index_readback_byte_identical\":true,\"pipeline_stages\":3,\"intermediate_host_roundtrips\":0,\n"
             << "\"input_contract\":\"raw_grouped_INT32_word_IDs_in_CPU_memory\",\n"
             << "\"timer_policy\":\"Search entry before validation/packing through CPU Top-K download/required checks/request cleanup\",\n"
             << "\"cold_index_load_runtime_upload_capacity_bootstrap_ms\":" << coldMs << ",\n"
             << "\"cold_bootstrap\":{\"synthetic_empty_queries\":" << capQ << ",\"query_h2d_calls\":2,\"query_h2d_bytes\":" << (capQ+1)*8+4
             << ",\"topk_d2h_calls\":2,\"topk_d2h_bytes\":" << 8*capQ*k << "},\n"
             << "\"workspace_capacity_bytes\":[" << workspaces[0] << "," << workspaces[1] << "," << workspaces[2] << "],\n"
             << "\"debug_index_d2h_calls\":5,\"debug_index_d2h_bytes\":" << session.indexUploadBytes << ",\n"
             << "\"debug_snapshot_policy\":\"after Search returns; full scores and partial candidates; no timed reference IO or debug download\",\n"
             << "\"transfer_counter_scope\":\"explicit application aclrtMemcpy only; framework internals not profiled\",\n"
             << "\"host_source_sha256\":{";
        for (size_t i=0;i<BM25_SEARCH_SOURCE_COUNT;++i)
            file << (i?",":"") << "\"" << BM25_SEARCH_SOURCES[i][0] << "\":\"" << BM25_SEARCH_SOURCES[i][1] << "\"";
        file << "},\n\"requests\":[\n";
        for (size_t i=0;i<requests.size();++i) {
            const auto &r=requests[i]; const auto &result=results[i];
            size_t words=0; for (const auto &row:r.queries) words+=row.size();
            file << (i?",\n":"") << "{\"label\":\"" << r.label << "\",\"queries\":" << r.queries.size() << ",\"words\":" << words
                 << ",\"search_call_ms\":" << result.searchMs << ",\"logical_query_bytes\":" << result.logicalQueryBytes
                 << ",\"query_h2d_calls\":2,\"query_h2d_bytes\":" << result.queryH2DBytes
                 << ",\"topk_d2h_calls\":2,\"topk_d2h_bytes\":" << Bytes(result.ids.size(),8)
                 << ",\"debug_poison_h2d_calls\":5,\"debug_poison_h2d_bytes\":" << session.debugPoisonBytesPerCall
                 << ",\"debug_score_candidate_d2h_calls\":3,\"debug_score_candidate_d2h_bytes\":" << 4*r.queries.size()*index.norm.size()+8*r.queries.size()*shards*k
                 << ",\"packing_inside_Search\":true,\"topk_received_before_stop\":true,\"debug_outside_Search\":true} ";
        }
        file << "\n]}\n"; file.close(); Check(static_cast<bool>(file),"Cannot save Search diagnostic report");
        std::cout << "Search-boundary device calls complete; independent ranking/formula checks still required. Not steady-state performance or a speedup claim.\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << "FAILED: " << error.what() << "; no CPU output substituted.\n"; return 1; }
}
