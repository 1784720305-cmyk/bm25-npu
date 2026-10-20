#include "search_inputs.h"
#include <iostream>

int main()
{
    try {
        using namespace bm25search;
        const Queries raw={{2,2,2147483646},{},{0,9},{}};
        const auto packed=PackQueries(raw,4,5);
        InputCheck(packed.offsets==std::vector<int64_t>({0,3,3,5,5}),"Mixed query offsets mismatch");
        InputCheck(packed.words==std::vector<int32_t>({2,2,2147483646,0,9}),"Unknown/repeated ID order changed");
        const auto empty=PackQueries(Queries(3),3,0);
        InputCheck(empty.words.empty() && empty.offsets==std::vector<int64_t>({0,0,0,0}),"Empty query handling mismatch");
        size_t failures=0;
        auto rejects=[&](const Queries &q,size_t capQ,size_t capWords) {
            try { (void)PackQueries(q,capQ,capWords); }
            catch (const std::exception &) { ++failures; return; }
            throw std::runtime_error("Invalid public input accepted");
        };
        rejects({},4,5); rejects(raw,3,5); rejects(raw,4,4);
        rejects({{-1}},1,1); rejects({{INT32_MAX}},1,1); rejects({{1}},1,0);
        InputCheck(failures==6 && raw[0]==std::vector<int32_t>({2,2,2147483646}),"Input rejection or immutability mismatch");
        std::cout << "Public query-ID packing/empty/duplicate/unknown/capacity tests passed. No ACLNN or device execution.\n";
        return 0;
    } catch (const std::exception &e) { std::cerr << "FAILED: " << e.what() << '\n'; return 1; }
}
