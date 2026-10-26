#pragma once
// Index-only preprocessing. Entries are absolute int64 posting offsets, including empty shards.
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace bm25plan {
inline std::vector<int64_t> BuildDirectory(const std::vector<int64_t> &offsets,
    const std::vector<int32_t> &postings,size_t docs,size_t shards)
{
    if (!docs || !shards || shards>docs || offsets.size()<2 || offsets.front()!=0 ||
        offsets.back()!=static_cast<int64_t>(postings.size())) throw std::runtime_error("Invalid directory dimensions");
    const size_t vocab=offsets.size()-1;
    if (vocab>SIZE_MAX/(shards+1)) throw std::runtime_error("Directory size overflow");
    std::vector<int64_t> result(vocab*(shards+1));
    for (size_t term=0;term<vocab;++term) {
        const int64_t first=offsets[term],last=offsets[term+1];
        if (first<0 || last<first || last>static_cast<int64_t>(postings.size()))
            throw std::runtime_error("Invalid directory term boundaries");
        auto cursor=postings.begin()+first,end=postings.begin()+last;
        for (size_t shard=0;shard<=shards;++shard) {
            const int64_t threshold=static_cast<int64_t>(docs)*static_cast<int64_t>(shard)/static_cast<int64_t>(shards);
            cursor=std::lower_bound(cursor,end,threshold);
            result[term*(shards+1)+shard]=static_cast<int64_t>(cursor-postings.begin());
        }
        if (result[term*(shards+1)]!=first || result[term*(shards+1)+shards]!=last)
            throw std::runtime_error("Directory endpoints disagree with index");
    }
    return result;
}
}
