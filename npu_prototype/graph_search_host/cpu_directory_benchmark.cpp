#include "../../cpu_reference/directory_search/cpu_directory_search.h"
#include "../search_benchmark_host/benchmark_common.h"

int main(int argc,char **argv)
{
    try {
        bm25bench::Check(argc==12,"Usage: CPU_BENCH INDEX A_RAW B_RAW OUTPUT THREADS K SHARDS CAP_Q CAP_WORDS WARMUP REPEATS");
        const auto output=std::filesystem::absolute(argv[4]); bm25bench::Check(!std::filesystem::exists(output),"Output exists");
        const int threads=static_cast<int>(bm25bench::Number(argv[5],1,65536));
        const auto k=bm25bench::Number(argv[6],1,32),shards=bm25bench::Number(argv[7],1,65536);
        const auto capq=bm25bench::Number(argv[8],1,10000),capwords=bm25bench::Number(argv[9],0,16777216);
        const int warmup=static_cast<int>(bm25bench::Number(argv[10],1,10000)),repeats=static_cast<int>(bm25bench::Number(argv[11],10,100000));
        const uint16_t endian=1; bm25bench::Check(*reinterpret_cast<const unsigned char *>(&endian)==1,"Requires little-endian");
        const auto a=bm25search::LoadQueries(argv[2]),b=bm25search::LoadQueries(argv[3]);
        const auto cold=bm25bench::Clock::now(); const bm25cpu::Index index(std::filesystem::absolute(argv[1]));
        bm25cpu::DirectoryCpuSearchSession session(index,threads,k,shards,capq,capwords);
        const double coldMs=std::chrono::duration<double,std::milli>(bm25bench::Clock::now()-cold).count();
        bm25bench::Check(std::filesystem::create_directories(output),"Cannot create fresh output");
        auto save=[&](const char *name,const auto &queries,const auto &result) {
            const auto path=output/name; bm25bench::Check(std::filesystem::create_directory(path),"Cannot create checkpoint");
            bm25cpu::Write(path/"top_scores.bin",result.scores); bm25cpu::Write(path/"top_ids.bin",result.ids);
            bm25cpu::Write(path/"scores.bin",session.DebugScores(queries.size()));
        };
        const auto data=bm25bench::Measure(session,a,b,warmup,repeats,[&](){session.PoisonForDiagnostic();},save,
            [](const auto &){return std::array<uint64_t,2>{0,0};});
        const bm25cpu::Index source(std::filesystem::absolute(argv[1]));
        bm25bench::Check(index.Equal(source),"Resident CPU index changed");
        std::ofstream record(output/"benchmark_session.json");
        bm25bench::WriteCommon(record,"cpu",data,index.norm.size(),a.size(),k,shards,capq,capwords,index.ByteCount(),coldMs);
        record << "\"requested_threads\":" << threads << ",\"actual_threads\":" << session.actualThreads
               << ",\"index_readback_byte_identical\":true,\"debug_source_reread_batches\":1,\n"
               << "\"fp32_flags\":\"-O3 -march=native -ffp-contract=off -fno-fast-math\"\n}\n";
        record.close(); bm25bench::Check(static_cast<bool>(record),"Cannot save CPU benchmark record");
        std::cout << "CPU Search measurement complete; independent checkpoints still required. No speedup claim.\n";
    } catch (const std::exception &error) { std::cerr << "FAILED: " << error.what() << '\n'; return 1; }
}
