// Independent original-document TF/DF oracle; never uses NPU or precomputed device outputs.
#include "cpu_index_search.h"
#include <iostream>
#include <numeric>
#include <set>

using namespace bm25cpu;
constexpr int V=7;
Queries Documents(int n)
{
    Queries docs(n);
    for (int i=0;i<n;++i) {
        if (i%7) { docs[i].assign(1+i%3,i%4); if (i%3==0) docs[i].push_back(4); }
        if (i%11==0) docs[i].insert(docs[i].end(),{0,5,5});
    }
    return docs;
}
Index Build(const Queries &raw)
{
    std::vector<int64_t> offsets{0}; std::vector<int32_t> ids; std::vector<float> tf,idf,norm;
    size_t total=0; for (const auto &doc:raw) total+=doc.size();
    const double avg=static_cast<double>(total)/raw.size();
    for (int t=0;t<V;++t) {
        size_t df=0;
        for (size_t d=0;d<raw.size();++d) {
            const auto count=std::count(raw[d].begin(),raw[d].end(),t);
            if (count) { ids.push_back(static_cast<int32_t>(d)); tf.push_back(static_cast<float>(count)); ++df; }
        }
        offsets.push_back(ids.size()); idf.push_back(static_cast<float>(std::log1p((raw.size()-df+.5)/(df+.5))));
    }
    for (const auto &doc:raw) norm.push_back(static_cast<float>(1.2*(1-.75+.75*doc.size()/avg)));
    return Index(std::move(offsets),std::move(ids),std::move(tf),std::move(norm),std::move(idf));
}
std::vector<float> Oracle(const Queries &documents,const Queries &queries)
{
    size_t total=0; for (const auto &d:documents) total+=d.size();
    const double avg=static_cast<double>(total)/documents.size();
    std::vector<float> idf(V);
    for (int t=0;t<V;++t) {
        const auto df=std::count_if(documents.begin(),documents.end(),[&](const auto &d){return std::find(d.begin(),d.end(),t)!=d.end();});
        idf[t]=static_cast<float>(std::log1p((documents.size()-df+.5)/(df+.5)));
    }
    std::vector<float> values;
    for (const auto &query:queries) for (const auto &doc:documents) {
        const float norm=static_cast<float>(1.2*(1-.75+.75*doc.size()/avg)); float score=0;
        for (int32_t t:query) if (t<V) {
            const float tf=static_cast<float>(std::count(doc.begin(),doc.end(),t));
            if (tf>0) { const float numerator=tf*2.2F; score+=(idf[t]*numerator)/(tf+norm); }
        }
        values.push_back(score);
    }
    return values;
}

int main()
{
    try {
        size_t calls=0,rejections=0;
        const Queries a={{0,0},{},{6},{7},{4,5,0},{2147483646,2},{0,1,2,3}};
        const Queries b={{4},{0},{},{7},{2,2,1}},empty(3);
        for (int n:{1,6,129,2048}) {
            const auto documents=Documents(n); const auto index=Build(documents); const auto saved=index;
            const std::set<int64_t> ks={1,std::min(n,3),std::min(n,10),std::min(n,32)};
            for (int64_t k:ks) for (int threads:{1,4,8}) {
                const int64_t shards=std::min(n,threads);
                CpuSearchSession session(index,threads,k,shards,7,128);
                for (const auto *query:{&a,&b,&a,&empty,&b}) {
                    session.PoisonForDiagnostic(); const auto result=session.Search(*query);
                    const auto expected=Oracle(documents,*query),actual=session.DebugScores(query->size());
                    Check(Same(expected,actual),"CPU full score bits differ from independent original-document FP32 oracle");
                    for (size_t q=0;q<query->size();++q) {
                        std::vector<int32_t> order(n); std::iota(order.begin(),order.end(),0);
                        std::sort(order.begin(),order.end(),[&](int32_t x,int32_t y){
                            const float sx=expected[q*n+x],sy=expected[q*n+y]; return sx!=sy?sx>sy:x<y;
                        });
                        for (int64_t rank=0;rank<k;++rank) {
                            const size_t pos=q*k+rank; const float score=expected[q*n+order[rank]];
                            Check(result.ids[pos]==order[rank] && std::memcmp(&result.scores[pos],&score,4)==0,"CPU strict Top-K or score bits mismatch");
                        }
                    }
                    Check(result.searchMs>0 && std::isfinite(result.searchMs) && index.Equal(saved),"Invalid timer or mutated index"); ++calls;
                }
            }
        }
        auto rejects=[&](auto function) {
            try { function(); } catch (const std::exception &) { ++rejections; return; }
            throw std::runtime_error("Invalid CPU Search input accepted");
        };
        const auto index=Build(Documents(6));
        rejects([&]{CpuSearchSession s(index,4,0,2,7,128);});
        rejects([&]{CpuSearchSession s(index,4,3,5,7,128);});
        rejects([&]{CpuSearchSession s(index,4,3,2,0,128);});
        for (const Queries bad:{Queries{},Queries{{-1}},Queries{{INT32_MAX}},Queries(8),Queries{{1,2,3}}}) {
            CpuSearchSession s(index,4,3,2,7,2); rejects([&]{s.Search(bad);}); rejects([&]{s.Search(Queries{{0}});});
        }
        auto corrupted=index; corrupted.tf[0]=-1; rejects([&]{corrupted.Validate();});
        corrupted=index; corrupted.offsets[0]=1; rejects([&]{corrupted.Validate();});
        corrupted=index; corrupted.norm[0]=std::numeric_limits<float>::quiet_NaN(); rejects([&]{corrupted.Validate();});
        std::cout << "CPU indexed Search passed " << calls << " poisoned/alternating calls with independent FP32 oracle and strict rankings; "
                  << rejections << " invalid/session-stop checks. No NPU or performance conclusion.\n"; return 0;
    } catch (const std::exception &e) { std::cerr << "FAILED: " << e.what() << '\n'; return 1; }
}
