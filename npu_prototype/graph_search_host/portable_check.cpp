#include "../../common/fixed_query_packet.h"
#include "../../cpu_reference/directory_search/cpu_directory_search.h"
#include <iostream>

int main()
{
    using bm25cpu::Check;
    try {
        bm25plan::PacketLayout layout(2,128,10);
        std::vector<unsigned char> data(layout.inputBytes,0xff);
        bm25search::Queries q{{0,0,4},{INT32_MAX-1}};
        Check(layout.Pack(q,data.data())==40,"Logical packet bytes differ");
        int64_t offsets[3]{}; std::memcpy(offsets,data.data(),24);
        Check(offsets[0]==0 && offsets[1]==3 && offsets[2]==4,"Packet offsets differ");
        layout.Pack(bm25search::Queries(2),data.data());
        Check(std::all_of(data.begin(),data.end(),[](unsigned char x){return x==0;}),"Empty packet retained old contents");
        for (const auto &bad:std::vector<bm25search::Queries>{{},{{-1},{}},{std::vector<int32_t>(65,0),{}}}) {
            const auto before=data; bool rejected=false;
            try { layout.Pack(bad,data.data()); } catch (const std::exception &) { rejected=true; }
            Check(rejected && data==before,"Invalid packet was not atomically rejected");
        }
        // Every shard count, empty term lists, uneven partitions and repeated terms.
        std::vector<int64_t> starts{0,9,9,14};
        std::vector<int32_t> docs{0,1,2,5,8,13,21,32,40,1,5,9,32,40};
        bm25cpu::Index index(starts,docs,std::vector<float>(docs.size(),2.0F),std::vector<float>(41,1.7F),{1.2F,0.0F,2.3F});
        for (int shards=1;shards<=8;++shards) {
            bm25cpu::CpuSearchSession original(index,8,3,shards,2,128);
            bm25cpu::DirectoryCpuSearchSession directory(index,8,3,shards,2,128);
            for (const auto &query:std::vector<bm25search::Queries>{{{0,2,0},{2}},{{},{}},{{99},std::vector<int32_t>(64,2)},{{0,2,0},{2}}}) {
                auto a=original.Search(query),b=directory.Search(query);
                Check(bm25cpu::Same(a.scores,b.scores) && a.ids==b.ids,"CPU directory changed score bits/ranking");
                Check(bm25cpu::Same(original.DebugScores(2),directory.DebugScores(2)),"CPU directory changed complete scores");
            }
        }
        std::cout << "Portable packing and CPU directory exact cross-checks passed; no NPU measurement.\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
