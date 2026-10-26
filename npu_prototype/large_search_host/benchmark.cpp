#include "large_search.h"
#include "merge_check.h"
#include "../search_benchmark_host/benchmark_common.h"
int main(int argc,char **argv) {
    try {
        if(argc==5 && std::string(argv[1])=="--merge-check") {
            CheckLargeMerge(fs::absolute(argv[2]),static_cast<int>(Number(argv[3],0,INT32_MAX)),fs::absolute(argv[4]));return 0;
        }
        if(argc==4 && std::string(argv[1])=="--memory") {
            Runtime rt;rt.device=static_cast<int>(Number(argv[2],0,INT32_MAX));Require(aclInit(nullptr),"memory init");rt.initialized=true;
            Require(aclrtSetDevice(rt.device),"memory device");rt.selected=true;
            size_t free=0,total=0;Require(aclrtGetMemInfo(ACL_HBM_MEM,&free,&total),"actual memory probe");
            Check(total>0,"Device memory unavailable");Check(!fs::exists(argv[3]),"Memory output exists");
            std::ofstream report(argv[3]);report<<"{\"free\":"<<free<<",\"total\":"<<total<<"}\n";report.close();
            Check(bool(report) && rt.Close(),"Memory probe cleanup/output failed");return 0;
        }
        bm25bench::Check(argc==14,"Usage: LARGE INDEX A_RAW B_RAW OUTPUT DEVICE VENDORS K Q WORDS WARMUP REPEATS REF_A REF_B");
        const auto output=fs::absolute(argv[4]);Check(!fs::exists(output),"New output required");
        const auto a=bm25search::LoadQueries(argv[2]),b=bm25search::LoadQueries(argv[3]);
        const int device=static_cast<int>(Number(argv[5],0,INT32_MAX));
        const auto k=Number(argv[7],1,32),q=Number(argv[8],1,10000),words=Number(argv[9],0,16777216);
        const int warm=static_cast<int>(Number(argv[10],1,10000)),repeats=static_cast<int>(Number(argv[11],10,100000));
        const auto cold=bm25bench::Clock::now();const Index index(fs::absolute(argv[1]));
        bm25large::Session session(index,fs::absolute(argv[6]),device,q,k,words);
        const auto coldMs=std::chrono::duration<double,std::milli>(bm25bench::Clock::now()-cold).count();
        Check(fs::create_directories(output),"Cannot create output");
        auto save=[&](const char *name,const auto &queries,const auto &result) {
            const auto path=output/name;Check(fs::create_directory(path),"Cannot create checkpoint");
            Write(path/"top_scores.bin",result.scores);Write(path/"top_ids.bin",result.ids);
            session.SaveDiagnostic(queries,result,path,fs::absolute(queries==a?argv[12]:argv[13]));
        };
        const auto measured=bm25bench::Measure(session,a,b,warm,repeats,[&](){session.PoisonForDiagnostic();},save,
            [](const auto &result){return std::array<uint64_t,2>{result.queryH2DBytes,result.topkD2HBytes};});
        session.VerifyIndex();session.Close();const auto &plan=session.GetPlan();
        std::ofstream record(output/"benchmark_session.json");
        bm25bench::WriteCommon(record,"npu",measured,plan.docs,q,k,plan.leaves,q,words,session.indexUploadBytes,coldMs);
        record<<"\"large_search_contract\":\"bounded8192_global_ids_tree32_v1\",\"device\":"<<device
            <<",\"padded_docs\":"<<plan.padded<<",\"leaf_docs\":8192,\"merge_fan_in\":32,\"list_counts\":[";
        for(size_t i=0;i<plan.lists.size();++i)record<<(i?",":"")<<plan.lists[i];
        record<<"],\"workspace_capacity_bytes\":[";
        for(size_t i=0;i<session.workspaceBytes.size();++i)record<<(i?",":"")<<session.workspaceBytes[i];
        record<<"],\"normal_full_score_elements\":0,\"diagnostic_extra_chains\":4,\"intermediate_host_roundtrips\":0,"
            <<"\"graph_capture\":true,\"index_readback_byte_identical\":true,\"normal_prepare_calls\":0,"
            <<"\"normal_search_calls\":"<<session.normalSearchCalls<<",\"replays\":"<<session.replays
            <<",\"pipeline_stages\":"<<plan.lists.size()<<",\"directory_bytes\":"<<plan.vocab*(plan.leaves+1)*8
            <<",\"packet_input_bytes\":"<<plan.inputBytes<<",\"packet_output_bytes\":"<<plan.outputBytes
            <<",\"known_device_bytes_upper_bound\":"<<plan.knownDeviceBytes<<",\"initial_free_device_bytes\":"<<session.initialFreeBytes
            <<",\"total_device_bytes\":"<<session.totalDeviceBytes<<"\n}\n";
        record.close();Check(bool(record),"Cannot save benchmark session");
        return 0;
    } catch(const std::exception &error) {std::cerr<<"FAILED: "<<error.what()<<std::endl;return 1;}
}
