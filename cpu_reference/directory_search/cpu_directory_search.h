#pragma once
// Same CPU implementation and flags; only index-only boundaries replace lower_bound.
#include "../index_search/cpu_index_search.h"
#include "../../common/shard_directory.h"
namespace bm25cpu {
class DirectoryCpuSearchSession {
    const Index &index;
    size_t maxQ,maxWords;
    int64_t k,shards,docs;
    std::vector<int64_t> boundaries;
    std::vector<float> fullScores;
    std::vector<Candidate> partials;
    bool failed=false;
public:
    int actualThreads=0;
    DirectoryCpuSearchSession(const Index &source,int threads,int64_t selectedK,int64_t documentShards,size_t queryCapacity,size_t wordCapacity)
        : index(source),maxQ(queryCapacity),maxWords(wordCapacity),k(selectedK),shards(documentShards),docs(static_cast<int64_t>(index.norm.size()))
    {
        Check(threads>=1 && threads<=65536 && maxQ>=1 && maxQ<=10000 && maxWords<=16777216 &&
              maxQ<=static_cast<uint64_t>(INT32_MAX)/docs && k>=1 && k<=std::min<int64_t>(32,docs) &&
              shards>=1 && shards<=std::min<int64_t>(threads,docs),"Invalid CPU search capacity/K/threads/document shards");
        boundaries=bm25plan::BuildDirectory(index.offsets,index.docs,index.norm.size(),shards);
        fullScores.resize(maxQ*docs); (void)Bytes(maxQ*shards*k,sizeof(Candidate)); partials.resize(maxQ*shards*k);
        omp_set_dynamic(0); omp_set_num_threads(threads);
        #pragma omp parallel
        {
            #pragma omp single
            actualThreads=omp_get_num_threads();
        }
        Check(actualThreads>=1 && actualThreads<=threads,"Invalid actual OpenMP team size");
        const auto bootstrap=Search(Queries(maxQ));
        for (size_t i=0;i<bootstrap.ids.size();++i)
            Check(bootstrap.scores[i]==0 && bootstrap.ids[i]==static_cast<int32_t>(i%k),"CPU empty capacity bootstrap failed");
    }
    Result Search(const Queries &queries)
    {
        const auto start=Clock::now(); // Same public boundary: raw grouped IDs already in CPU memory.
        try {
            Check(!failed,"CPU session already failed");
            Check(!queries.empty() && queries.size()<=maxQ,"CPU query batch exceeds capacity or is empty");
            size_t words=0;
            for (const auto &row:queries) {
                Check(row.size()<=maxWords-words,"CPU query words exceed capacity"); words+=row.size();
                for (int32_t term:row) Check(term>=0 && term<INT32_MAX,"Invalid CPU public word ID");
            }
            const int64_t q=static_cast<int64_t>(queries.size());
            // Disjoint article partitions preserve query-term FP32 addition order without atomics.
            // Lower-bound only the requested terms' posting lists; no full vocabulary scan.
            #pragma omp parallel for schedule(static)
            for (int64_t task=0;task<q*shards;++task) {
                const int64_t query=task/shards,shard=task%shards;
                const int64_t begin=docs*shard/shards,end=docs*(shard+1)/shards;
                auto *scores=fullScores.data()+query*docs;
                std::fill(scores+begin,scores+end,0.0F); // Required per-request work, inside Search.
                for (int32_t term:queries[query]) {
                    if (static_cast<size_t>(term)>=index.idf.size()) continue;
                    const auto entry=static_cast<size_t>(term)*(shards+1)+shard;
                    const auto low=index.docs.begin()+boundaries[entry],high=index.docs.begin()+boundaries[entry+1];
                    for (auto it=low;it!=high;++it) {
                        const size_t pos=static_cast<size_t>(it-index.docs.begin()); const int32_t doc=*it;
                        const float numerator=index.tf[pos]*2.2F;
                        const float contribution=(index.idf[term]*numerator)/(index.tf[pos]+index.norm[doc]);
                        scores[doc]+=contribution;
                    }
                }
            }
            Result result; result.words=words; result.scores.resize(static_cast<size_t>(q)*k); result.ids.resize(result.scores.size());
            TopKDocumentShards(fullScores,q,docs,k,shards,partials,result.scores,result.ids);
            for (size_t i=0;i<result.ids.size();++i) {
                Check(result.ids[i]>=0 && result.ids[i]<docs && std::isfinite(result.scores[i]) && result.scores[i]>=0,"Invalid CPU Top-K");
                if (i%static_cast<size_t>(k)) Check(result.scores[i-1]>result.scores[i] ||
                    (result.scores[i-1]==result.scores[i] && result.ids[i-1]<result.ids[i]),"CPU Top-K order/tie rule invalid");
            }
            result.searchMs=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
            return result;
        } catch (...) { failed=true; throw; }
    }
    void PoisonForDiagnostic()
    {
        Check(!failed,"CPU diagnostic on failed session");
        std::fill(fullScores.begin(),fullScores.end(),std::numeric_limits<float>::quiet_NaN());
        std::fill(partials.begin(),partials.end(),Candidate{std::numeric_limits<float>::quiet_NaN(),-1});
    }
    std::vector<float> DebugScores(size_t q) const
    {
        Check(!failed && q>=1 && q<=maxQ,"Invalid CPU diagnostic snapshot");
        return std::vector<float>(fullScores.begin(),fullScores.begin()+q*docs);
    }
};
}
