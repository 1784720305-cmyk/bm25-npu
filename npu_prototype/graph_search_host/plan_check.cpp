#include "planned_search.h"

int main(int argc,char **argv)
{
    try {
        Check(argc==12,"Usage: PLAN_CHECK INDEX REQUESTS OUTPUT DEVICE VENDORS K SHARDS QCAP WCAP GRAPH DIRECTORY");
        const Index index(fs::absolute(argv[1]));
        const auto requests=LoadRequests(fs::absolute(argv[2]),index.norm.size());
        const auto output=fs::absolute(argv[3]); Check(!fs::exists(output),"Check output exists");
        const int device=static_cast<int>(Number(argv[4],0,INT32_MAX));
        const auto k=Number(argv[6],1,32),shards=Number(argv[7],1,32),qcap=Number(argv[8],1,10000),wcap=Number(argv[9],0,16777216);
        const bool graph=Number(argv[10],0,1)!=0,directory=Number(argv[11],0,1)!=0;
        bm25search::PlannedNpuSearchSession session(index,fs::absolute(argv[5]),device,k,shards,qcap,wcap,graph,directory);
        Check(fs::create_directories(output),"Cannot create plan check output");
        std::map<bm25search::Queries,bm25search::PlannedResult> previous;
        for (const auto &request:requests) {
            Check(request.Queries()==qcap,"Diagnostic Q differs from fixed session");
            bm25search::Queries queries(static_cast<size_t>(qcap));
            for (size_t q=0;q<queries.size();++q) queries[q].assign(request.terms.begin()+request.offsets[q],request.terms.begin()+request.offsets[q+1]);
            session.PoisonForDiagnostic();
            const auto result=session.Search(queries);
            if (previous.count(queries)) Check(Same(previous.at(queries).scores,result.scores) &&
                Same(previous.at(queries).ids,result.ids),"Changed bits after intervening queries");
            previous[queries]=result;
            const auto destination=output/request.label; Check(fs::create_directory(destination),"Duplicate plan check destination");
            Write(destination/"top_scores.bin",result.scores); Write(destination/"top_ids.bin",result.ids);
            session.SaveDiagnostic(queries,result,destination,request.directory/"expected_scores.bin");
            std::cout << "planned_request_passed=" << request.label << " graph=" << graph << " directory=" << directory << std::endl;
        }
        // Invalid input cannot be packed into a live plan. A new session is not needed to
        // check portable packing rejection: the exact same routine runs inside Search.
        bm25plan::PacketLayout layout(qcap,wcap,k); std::vector<unsigned char> packet(layout.inputBytes,0xa7);
        auto reject=[&](const bm25search::Queries &q) {
            const auto before=packet; bool rejected=false;
            try { layout.Pack(q,packet.data()); } catch (const std::exception &) { rejected=true; }
            Check(rejected && packet==before,"Invalid packet changed plan input before rejection");
        };
        reject(bm25search::Queries());
        auto invalid=bm25search::Queries(qcap); invalid[0]={-1}; reject(invalid);
        invalid[0]=std::vector<int32_t>(65,0); reject(invalid);
        session.VerifyIndex(); session.Close();
        std::ofstream report(output/"report.json");
        report << "{\"status\":\"actual_planned_chain_full_scores_rankings_guards_and_new_inputs_passed\",\"calls\":"
            << requests.size() << ",\"graph\":" << (graph?"true":"false") << ",\"directory\":" << (directory?"true":"false") << "}\n";
        report.close(); Check(static_cast<bool>(report),"Cannot save plan check report");
    } catch (const std::exception &e) { std::cerr << "FAILED: " << e.what() << '\n'; return 1; }
}
