#pragma once
// Portable packing into session-owned memory; no query result caching or ACL dependency.
#include "../npu_prototype/search_host/search_inputs.h"
#include <algorithm>
#include <cstddef>
#include <cstring>

namespace bm25plan {
inline size_t Align32(size_t n)
{
    bm25search::InputCheck(n <= SIZE_MAX-31,"Packet size overflow");
    return (n+31)&~size_t(31);
}
struct PacketLayout {
    size_t queries,words,wordOffset,inputBytes,idOffset,outputBytes,k;
    PacketLayout(size_t q,size_t w,size_t selectedK):queries(q),words(w),k(selectedK)
    {
        bm25search::InputCheck(q>=1 && q<=10000 && w<=16777216 && k>=1 && k<=32,"Invalid fixed packet capacity");
        wordOffset=Align32((q+1)*sizeof(int64_t));
        inputBytes=Align32(wordOffset+std::max<size_t>(1,w)*sizeof(int32_t));
        idOffset=Align32(q*k*sizeof(float));
        outputBytes=Align32(idOffset+q*k*sizeof(int32_t));
    }
    size_t Pack(const bm25search::Queries &input,void *memory) const
    {
        bm25search::InputCheck(memory && input.size()==queries,"Fixed session requires its declared query count");
        size_t total=0;
        // Validate the complete request before modifying memory used by the plan.
        for (const auto &row:input) {
            bm25search::InputCheck(row.size()<=64 && row.size()<=words-total,"Query exceeds fixed word/64-term capacity");
            total+=row.size();
            for (int32_t word:row) bm25search::InputCheck(word>=0 && word<INT32_MAX,"Invalid public word ID");
        }
        auto *bytes=static_cast<unsigned char *>(memory);
        std::memset(bytes,0,inputBytes); // Includes unused words, offsets and alignment padding.
        int64_t end=0;
        for (size_t row=0;row<input.size();++row) {
            const auto &terms=input[row];
            if (!terms.empty()) std::memcpy(bytes+wordOffset+static_cast<size_t>(end)*4,terms.data(),terms.size()*4);
            end+=static_cast<int64_t>(terms.size());
            std::memcpy(bytes+(row+1)*8,&end,8);
        }
        return (queries+1)*8+total*4; // Logical bytes; DMA always uses inputBytes.
    }
};
}
