#pragma once
// Same public grouped word-ID input as NPU Search; standard C++/OpenMP, no ACL.
#include "../../npu_prototype/search_host/search_inputs.h"
#include "../topk_document_shards.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <omp.h>

namespace bm25cpu {
namespace fs=std::filesystem;
using Queries=bm25search::Queries;
using Clock=std::chrono::steady_clock;
inline void Check(bool ok,const std::string &message) { if (!ok) throw std::runtime_error(message); }
inline size_t Bytes(uint64_t n,size_t width)
{
    Check(n<=std::numeric_limits<size_t>::max()/width && n<=static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())/width,
          "Array exceeds host byte limits"); return static_cast<size_t>(n)*width;
}
template<class T> std::vector<T> Read(const fs::path &path)
{
    const auto bytes=fs::file_size(path);
    Check(bytes>0 && bytes%sizeof(T)==0 && bytes/sizeof(T)<=INT32_MAX,"Invalid index array size: "+path.string());
    std::vector<T> values(bytes/sizeof(T)); std::ifstream file(path,std::ios::binary);
    Check(file.is_open() && static_cast<bool>(file.read(reinterpret_cast<char *>(values.data()),Bytes(values.size(),sizeof(T)))),
          "Cannot read "+path.string()); return values;
}
template<class T> void Write(const fs::path &path,const std::vector<T> &values)
{
    std::ofstream file(path,std::ios::binary);
    file.write(reinterpret_cast<const char *>(values.data()),Bytes(values.size(),sizeof(T)));
    file.close(); Check(static_cast<bool>(file),"Cannot save "+path.string());
}
template<class T> bool Same(const std::vector<T> &a,const std::vector<T> &b)
{
    return a.size()==b.size() && std::memcmp(a.data(),b.data(),Bytes(a.size(),sizeof(T)))==0;
}

struct Index {
    std::vector<int64_t> offsets;
    std::vector<int32_t> docs;
    std::vector<float> tf,norm,idf;
    explicit Index(const fs::path &directory)
        : offsets(Read<int64_t>(directory/"term_offsets.bin")),docs(Read<int32_t>(directory/"posting_doc_ids.bin")),
          tf(Read<float>(directory/"posting_tf.bin")),norm(Read<float>(directory/"doc_norm.bin")),idf(Read<float>(directory/"idf.bin")) { Validate(); }
    Index(std::vector<int64_t> o,std::vector<int32_t> d,std::vector<float> t,std::vector<float> n,std::vector<float> i)
        : offsets(std::move(o)),docs(std::move(d)),tf(std::move(t)),norm(std::move(n)),idf(std::move(i)) { Validate(); }
    void Validate() const
    {
        Check(!norm.empty() && norm.size()<=INT32_MAX && !idf.empty() && idf.size()<INT32_MAX && !docs.empty() && docs.size()<=INT32_MAX &&
              offsets.size()==idf.size()+1 && docs.size()==tf.size() && offsets.front()==0 && offsets.back()==static_cast<int64_t>(docs.size()),
              "Invalid resident index dimensions");
        for (float value:norm) Check(std::isfinite(value) && value>0,"Invalid document norm");
        for (size_t term=0;term<idf.size();++term) {
            Check(std::isfinite(idf[term]) && idf[term]>=0,"Invalid IDF");
            const auto begin=offsets[term],end=offsets[term+1];
            Check(begin>=0 && begin<=end && end<=static_cast<int64_t>(docs.size()),"Invalid term posting boundaries");
            for (int64_t pos=begin;pos<end;++pos) {
                Check(docs[pos]>=0 && static_cast<size_t>(docs[pos])<norm.size() && (pos==begin || docs[pos-1]<docs[pos]),"Unsorted/duplicate/out-of-range posting ID");
                Check(std::isfinite(tf[pos]) && tf[pos]>=1 && tf[pos]<=16777216.0F && std::floor(tf[pos])==tf[pos],"Invalid TF");
            }
        }
    }
    uint64_t ByteCount() const { return Bytes(offsets.size(),8)+Bytes(docs.size(),4)+Bytes(tf.size(),4)+Bytes(norm.size(),4)+Bytes(idf.size(),4); }
    bool Equal(const Index &other) const { return Same(offsets,other.offsets) && Same(docs,other.docs) && Same(tf,other.tf) && Same(norm,other.norm) && Same(idf,other.idf); }
};

struct Result { std::vector<float> scores; std::vector<int32_t> ids; double searchMs=0; size_t words=0; };

class CpuSearchSession {
    const Index &index;
    size_t maxQ,maxWords;
    int64_t k,shards,docs;
    std::vector<float> fullScores;
    std::vector<Candidate> partials;
    bool failed=false;
public:
    int actualThreads=0;
    CpuSearchSession(const Index &source,int threads,int64_t selectedK,int64_t documentShards,size_t queryCapacity,size_t wordCapacity)
        : index(source),maxQ(queryCapacity),maxWords(wordCapacity),k(selectedK),shards(documentShards),docs(static_cast<int64_t>(index.norm.size()))
    {
        Check(threads>=1 && threads<=65536 && maxQ>=1 && maxQ<=10000 && maxWords<=16777216 &&
              maxQ<=static_cast<uint64_t>(INT32_MAX)/docs && k>=1 && k<=std::min<int64_t>(32,docs) &&
              shards>=1 && shards<=std::min<int64_t>(threads,docs),"Invalid CPU search capacity/K/threads/document shards");
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
                    const auto first=index.docs.begin()+index.offsets[term],last=index.docs.begin()+index.offsets[term+1];
                    const auto low=std::lower_bound(first,last,begin),high=std::lower_bound(low,last,end);
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
