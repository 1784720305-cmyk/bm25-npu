#include "fused_search.h"
#include "../search_benchmark_host/benchmark_common.h"

int main(int argc,char **argv)
{
    try {
        bm25bench::Check(argc==15,"Usage: NPU_BENCH INDEX A_RAW B_RAW OUTPUT DEVICE VENDORS K SHARDS CAP_Q CAP_WORDS WARMUP REPEATS REF_A REF_B");
        const auto output=fs::absolute(argv[4]); bm25bench::Check(!fs::exists(output),"Output exists");
        const int device=static_cast<int>(bm25bench::Number(argv[5],0,INT32_MAX));
        const auto k=bm25bench::Number(argv[7],1,32),shards=bm25bench::Number(argv[8],1,32);
        const auto capq=bm25bench::Number(argv[9],1,10000),capwords=bm25bench::Number(argv[10],0,16777216);
        const int warmup=static_cast<int>(bm25bench::Number(argv[11],1,10000)),repeats=static_cast<int>(bm25bench::Number(argv[12],10,100000));
        const uint16_t endian=1; bm25bench::Check(*reinterpret_cast<const unsigned char *>(&endian)==1,"Requires little-endian");
        const auto a=bm25search::LoadQueries(argv[2]),b=bm25search::LoadQueries(argv[3]);
        const auto cold=bm25bench::Clock::now(); const Index index(fs::absolute(argv[1]));
        bm25search::NpuSearchSession session(index,fs::absolute(argv[6]),device,k,shards,capq,capwords);
        const double coldMs=std::chrono::duration<double,std::milli>(bm25bench::Clock::now()-cold).count();
        std::array<uint64_t,2> capacities{};
        for (size_t i=0;i<2;++i) capacities[i]=session.WorkspaceCapacities()[i];
        bm25bench::Check(fs::create_directories(output),"Cannot create fresh output");
        auto save=[&](const char *name,const auto &queries,const auto &result) {
            const auto path=output/name; bm25bench::Check(fs::create_directory(path),"Cannot create checkpoint");
            Write(path/"top_scores.bin",result.scores); Write(path/"top_ids.bin",result.ids); session.SaveDiagnostic(queries,result,path,fs::absolute(queries==a?argv[13]:argv[14]));
        };
        const auto data=bm25bench::Measure(session,a,b,warmup,repeats,[&](){session.PoisonForDiagnostic();},save,
            [](const auto &result){return std::array<uint64_t,2>{result.queryH2DBytes,8*result.ids.size()};});
        session.VerifyIndex(); session.Close();
        std::ofstream record(output/"benchmark_session.json");
        bm25bench::WriteCommon(record,"npu",data,index.norm.size(),a.size(),k,shards,capq,capwords,session.indexUploadBytes,coldMs);
        record << "\"normal_full_score_elements\":0,\"debug_placeholder_elements\":1,\"diagnostic_extra_chains\":4,\n"
               << "\"device\":" << device << ",\"index_tensor_h2d_calls\":5,\"pipeline_stages\":2,\n"
               << "\"index_readback_byte_identical\":true,\"debug_index_d2h_calls\":5,\"intermediate_host_roundtrips\":0,\n"
               << "\"workspace_capacity_bytes\":[" << capacities[0] << "," << capacities[1] << "]\n}\n";
        record.close(); bm25bench::Check(static_cast<bool>(record),"Cannot save NPU benchmark record");
        std::cout << "NPU Search measurement complete; independent checkpoints still required. No CPU fallback or speedup claim.\n";
    } catch (const std::exception &error) { std::cerr << "FAILED: " << error.what() << '\n'; return 1; }
}
