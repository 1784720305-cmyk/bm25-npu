#ifndef BM25_COMMON_TOPK_HEAP_H
#define BM25_COMMON_TOPK_HEAP_H
#include <algorithm>
#include <array>
#include <cstdint>
namespace ascend::bm25 {
struct Candidate {
    float score;
    int32_t id;
};
struct Better {
    bool operator()(const Candidate &a, const Candidate &b) const
    {
        return a.score > b.score || (a.score == b.score && a.id < b.id);
    }
};

inline void KeepCandidate(std::array<Candidate, 32> &heap, int64_t &size, int64_t k, Candidate item)
{
    if (size < k) {
        heap[size++] = item;
        std::push_heap(heap.begin(), heap.begin() + size, Better{});
    } else if (Better{}(item, heap[0])) {
        std::pop_heap(heap.begin(), heap.begin() + size, Better{});
        heap[size - 1] = item;
        std::push_heap(heap.begin(), heap.begin() + size, Better{});
    }
}

} // namespace ascend::bm25
#endif // BM25_COMMON_TOPK_HEAP_H
