#pragma once
// Own bounded-heap + document-partition baseline. Standard C++/OpenMP only.
// A global top-K member must belong to its partition's top-K under the same total order.
#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

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
        std::push_heap(heap.begin(), heap.begin()+size, Better{});
    } else if (Better{}(item, heap[0])) {
        std::pop_heap(heap.begin(), heap.begin()+size, Better{});
        heap[size-1] = item;
        std::push_heap(heap.begin(), heap.begin()+size, Better{});
    }
}

// Caller validates shapes/K/shards and preallocates partials outside the compute timer.
inline void TopKDocumentShards(const std::vector<float> &scores, int64_t queries, int64_t docs,
                              int64_t k, int64_t shards, std::vector<Candidate> &partials,
                              std::vector<float> &topScores, std::vector<int32_t> &topIds)
{
    #pragma omp parallel for schedule(static)
    for (int64_t task = 0; task < queries*shards; ++task) {
        const int64_t query = task/shards;
        const int64_t shard = task%shards;
        // Uneven partitions are legal. These products fit int64 under existing Q*D/threads limits.
        const int64_t begin = docs*shard/shards;
        const int64_t end = docs*(shard+1)/shards;
        std::array<Candidate, 32> heap{};
        int64_t size = 0;
        for (int64_t doc = begin; doc < end; ++doc)
            KeepCandidate(heap, size, k, Candidate{scores[query*docs+doc], static_cast<int32_t>(doc)});
        // Local candidates do not need sorting: the final merge consumes an unordered bounded heap.
        for (int64_t rank = 0; rank < k; ++rank)
            partials[task*k+rank] = rank < size ? heap[rank] : Candidate{-1.0F, -1};
    }
    #pragma omp parallel for schedule(static)
    for (int64_t query = 0; query < queries; ++query) {
        std::array<Candidate, 32> heap{};
        int64_t size = 0;
        for (int64_t i = 0; i < shards*k; ++i) {
            const Candidate item = partials[query*shards*k+i];
            if (item.id >= 0) KeepCandidate(heap, size, k, item);
        }
        std::sort_heap(heap.begin(), heap.begin()+size, Better{});
        for (int64_t rank = 0; rank < k; ++rank) {
            topScores[query*k+rank] = heap[rank].score;
            topIds[query*k+rank] = heap[rank].id;
        }
    }
}
