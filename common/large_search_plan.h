#pragma once
// Portable bounded plan. No dependency on ACL, CPU implementation, or device execution.
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
namespace bm25large {
constexpr uint64_t Leaf=8192,FanIn=32,MaxLeaves=65536,Limit=2147483647;
inline void RequirePlan(bool ok,const char *why) {if(!ok)throw std::runtime_error(why);}
inline uint64_t Add(uint64_t a,uint64_t b) {RequirePlan(a<=UINT64_MAX-b,"Plan addition overflow");return a+b;}
inline uint64_t Mul(uint64_t a,uint64_t b) {RequirePlan(!b || a<=UINT64_MAX/b,"Plan product overflow");return a*b;}
inline uint64_t Align(uint64_t n) {return Mul(Add(n,31)/32,32);}
struct Plan {
    uint64_t docs,padded,leaves,q,k,vocab,postings,inputBytes,outputBytes,indexBytes,candidateBytes=0,knownDeviceBytes;
    std::vector<uint64_t> lists;
    Plan(uint64_t n,uint64_t v,uint64_t m,uint64_t queries,uint64_t selected,uint64_t words)
        :docs(n),q(queries),k(selected),vocab(v),postings(m) {
        RequirePlan(n>=selected && n<=Limit && selected>=1 && selected<=32 && v>=1 && v<Limit &&
            m>=1 && m<=Limit && q>=1 && q<=10000 && words<=16777216,"Unsupported index/query/K dimensions");
        leaves=Add(n,Leaf-1)/Leaf;padded=Mul(leaves,Leaf);
        RequirePlan(leaves<=MaxLeaves && padded<=Limit && q<=Limit/padded,"Padded index/Q*N exceeds explicit INT32 contract");
        lists.push_back(leaves);
        do {lists.push_back(Add(lists.back(),FanIn-1)/FanIn);} while(lists.back()>1);
        inputBytes=Align(Add(Align(Mul(q+1,8)),Mul(words?words:1,4)));
        outputBytes=Align(Add(Align(Mul(Mul(q,k),4)),Mul(Mul(q,k),4)));
        indexBytes=Add(Add(Mul(m,8),Mul(padded,4)),Add(Mul(v,4),Mul(Mul(v,leaves+1),8)));
        for(auto count:lists)candidateBytes=Add(candidateBytes,Mul(Mul(Mul(q,count),k),8));
        // Five index arenas, score/id pair per tree level, input/output/placeholder arenas.
        knownDeviceBytes=Add(Add(indexBytes,candidateBytes),Add(Add(inputBytes,outputBytes),Add(4,Mul(8+2*lists.size(),128))));
    }
};
}
