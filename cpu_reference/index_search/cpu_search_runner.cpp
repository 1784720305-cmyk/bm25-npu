// Raw grouped query IDs -> resident host index -> CPU Top-K. First-call diagnostic only.
#include "cpu_index_search.h"
#include "compiled_cpu_search_sources.h"
#include <cctype>
#include <iomanip>
#include <iostream>
#include <map>

using namespace bm25cpu;
struct Request { std::string label; fs::path source; Queries queries; };
int64_t Number(const char *text,int64_t low,int64_t high)
{
    size_t used=0; const auto value=std::stoll(text,&used);
    Check(used==std::strlen(text) && value>=low && value<=high,"Invalid numeric argument"); return value;
}
std::vector<Request> LoadPlan(const fs::path &path)
{
    std::ifstream file(path); Check(file.is_open(),"Cannot open CPU raw query plan");
    std::vector<Request> requests; std::map<std::string,bool> names; std::string line;
    while (std::getline(file,line)) {
        if (!line.empty() && line.back()=='\r') line.pop_back();
        const auto pos=line.find('\t');
        Check(pos!=std::string::npos && pos>0 && line.find('\t',pos+1)==std::string::npos,"Invalid raw plan row");
        Request r; r.label=line.substr(0,pos);
        Check(std::all_of(r.label.begin(),r.label.end(),[](unsigned char c){return std::isalnum(c)||c=='_';}) &&
              names.emplace(r.label,true).second,"Unsafe/duplicate CPU request label");
        r.source=fs::canonical(line.substr(pos+1)); r.queries=bm25search::LoadQueries(r.source.string());
        requests.push_back(std::move(r)); Check(requests.size()<=32,"CPU diagnostic exceeds 32 requests");
    }
    Check(file.eof() && !requests.empty(),"Empty/unreadable raw plan"); return requests;
}

int main(int argc,char **argv)
{
    try {
        Check(argc==10,"Usage: bm25_cpu_index_search_runner INDEX PLAN OUTPUT THREADS K DOCUMENT_SHARDS MAX_QUERIES MAX_WORDS raw_query_ids_v1");
        // argv[9] is an explicit protocol tag, preventing accidental invocation as an old prepared-array runner.
        Check(std::string(argv[9])=="raw_query_ids_v1","CPU Search protocol tag mismatch");
        const auto output=fs::absolute(argv[3]); Check(!fs::exists(output),"Output exists; choose a fresh directory");
        const int threads=static_cast<int>(Number(argv[4],1,65536));
        const auto k=Number(argv[5],1,32),shards=Number(argv[6],1,65536),capQ=Number(argv[7],1,10000),capWords=Number(argv[8],0,16777216);
        static_assert(sizeof(float)==4 && sizeof(int32_t)==4 && sizeof(int64_t)==8 && std::numeric_limits<float>::is_iec559);
        const uint16_t endian=1; Check(*reinterpret_cast<const unsigned char *>(&endian)==1,"Requires little-endian host");
        std::cout << "stage=load_raw_grouped_query_ids\n";
        const auto requests=LoadPlan(fs::absolute(argv[2]));
        const auto start=Clock::now(); const Index index(fs::absolute(argv[1]));
        CpuSearchSession session(index,threads,k,shards,capQ,capWords);
        const double coldMs=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
        Check(fs::create_directories(output),"Cannot create fresh CPU diagnostic output");
        std::vector<Result> results; std::map<std::string,size_t> previous;
        for (size_t i=0;i<requests.size();++i) {
            const auto &r=requests[i];
            session.PoisonForDiagnostic(); // Outside public Search; no artificial clear cost removal inside Search.
            auto result=session.Search(r.queries);
            const auto dir=output/r.label; Check(fs::create_directory(dir),"Cannot create CPU request output");
            Write(dir/"top_scores.bin",result.scores); Write(dir/"top_ids.bin",result.ids);
            Write(dir/"scores.bin",session.DebugScores(r.queries.size()));
            const auto old=previous.find(r.source.string());
            if (old!=previous.end()) Check(Same(result.scores,results[old->second].scores) && Same(result.ids,results[old->second].ids),"Repeated CPU Top-K bits changed");
            else previous.emplace(r.source.string(),i);
            std::cout << std::setprecision(17) << "cpu_search_completed=" << r.label << " queries=" << r.queries.size()
                      << " search_call_ms=" << result.searchMs << '\n';
            results.push_back(std::move(result));
        }
        const Index original(fs::absolute(argv[1])); // Debug reread outside Search; Python also checks input hashes unchanged.
        Check(index.Equal(original),"Resident host index differs from original source bytes");
        std::ofstream file(output/"cpu_search_session.json"); file << std::setprecision(17);
        file << "{\n\"status\":\"cpu_index_search_calls_completed_independent_check_pending\",\n"
             << "\"docs\":" << index.norm.size() << ",\"k\":" << k << ",\"document_shards\":" << shards
             << ",\"requested_threads\":" << threads << ",\"actual_threads\":" << session.actualThreads << ",\n"
             << "\"capacity_queries\":" << capQ << ",\"capacity_words\":" << capWords << ",\n"
             << "\"input_contract\":\"raw_grouped_INT32_word_IDs_in_CPU_memory\",\n"
             << "\"timer_policy\":\"CPU Search entry before request validation through index lookup/FP32 scoring/Top-K/required checks/CPU result\",\n"
             << "\"index_storage\":\"host_memory\",\"query_index_loads\":1,\"index_bytes\":" << index.ByteCount() << ",\n"
             << "\"host_index_byte_identical_to_source\":true,\"debug_source_reread_batches\":1,\n"
             << "\"cold_index_load_threads_capacity_bootstrap_ms\":" << coldMs << ",\"cold_bootstrap_empty_queries\":" << capQ << ",\n"
             << "\"fp32_flags\":\"-O3 -march=native -ffp-contract=off -fno-fast-math\",\n"
             << "\"algorithm\":\"ordered term posting-range lookup per article partition; bounded heaps and final merge\",\n"
             << "\"cpu_needs_backend_word_packing\":false,\"diagnostic_poison_before_calls\":true,\"diagnostic_files_outside_Search\":true,\n"
             << "\"host_source_sha256\":{";
        for (size_t i=0;i<BM25_CPU_SEARCH_SOURCE_COUNT;++i)
            file << (i?",":"") << "\"" << BM25_CPU_SEARCH_SOURCES[i][0] << "\":\"" << BM25_CPU_SEARCH_SOURCES[i][1] << "\"";
        file << "},\n\"requests\":[\n";
        for (size_t i=0;i<requests.size();++i) {
            const auto &r=requests[i]; const auto &result=results[i];
            file << (i?",\n":"") << "{\"label\":\"" << r.label << "\",\"queries\":" << r.queries.size() << ",\"words\":" << result.words
                 << ",\"search_call_ms\":" << result.searchMs << ",\"topk_output_bytes\":" << Bytes(result.ids.size(),8)
                 << ",\"debug_full_scores_bytes\":" << 4*r.queries.size()*index.norm.size()
                 << ",\"index_lookup_inside_Search\":true,\"topk_ready_before_stop\":true,\"debug_outside_Search\":true}";
        }
        file << "\n]}\n"; file.close(); Check(static_cast<bool>(file),"Cannot save CPU Search session report");
        std::cout << "CPU indexed Search diagnostic completed. Validate independent formula and ranking; no steady-state or speedup claim.\n";
        return 0;
    } catch (const std::exception &e) { std::cerr << "FAILED: " << e.what() << "; stop, no cached/alternate result substituted.\n"; return 1; }
}
