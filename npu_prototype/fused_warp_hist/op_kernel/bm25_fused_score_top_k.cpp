// Independent first-radix-pass per-warp UB histogram experiment for Ascend950. SOURCE ONLY.
// Starts from the retained256-thread fused scorer/positive-compaction selector.
// Scoring and the rest of selection keep the accepted formula, ordering and operations.
// No full-score GM array in debug=0. No CPU fallback or high-level TopK library.
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_warp_functions.h"
#include "simt_api/device_sync_functions.h"

namespace {
constexpr uint32_t kThreads = 256;
constexpr uint32_t kWarp = 32;
constexpr uint32_t kWarps = kThreads / kWarp;
// Explicit TBuf allocation, never an assumed UB address. Layout in uint32_t words:
// histogram[2048], scan totals[16], control[8], selected keys/IDs[32], positive raw bits/IDs[1024].
constexpr uint32_t kScan = 2048;
constexpr uint32_t kControl = kScan + 2*kWarps;
constexpr uint32_t kCandidateKeys = kControl + 8;
constexpr uint32_t kCandidateIds = kCandidateKeys + 32;
constexpr uint32_t kCapacity = 1024;
constexpr uint32_t kMaskDocs = kThreads * 64; // two 32-bit membership masks per thread
constexpr uint32_t kPositiveBits = kCandidateIds + 32;
constexpr uint32_t kPositiveIds = kPositiveBits + kCapacity;
// Per-warp histogram layout idea: CANN official950 sample (not its pure-SIMT launcher).
// https://gitcode.com/cann/cann-samples/blob/d4f652e821a3ffc3282400771e1e785c874bb26c/Samples/2_Performance/simt_histogram_story/src/1_basic_simt.asc
// Own first-radix-pass adaptation: warp0 uses the existing histogram; other warps
// use disjoint padded UB tables. No library dependency or GM histogram.
constexpr uint32_t kHistBins = 2048, kHistStride = kHistBins + 8;
constexpr uint32_t kPrivateHist = kPositiveIds + kCapacity;
constexpr uint32_t kPrivateHistWords = (kWarps - 1)*kHistStride;
constexpr uint32_t kSelectScratchBytes = (kPrivateHist + kPrivateHistWords)*sizeof(uint32_t);
static_assert(kSelectScratchBytes == 74304 && kSelectScratchBytes % 32 == 0);
constexpr uint32_t kNoId = 0x7fffffffU;

__simt_callee__ __aicore__ inline bool ValidBits(uint32_t raw)
{
    const uint32_t magnitude = raw & 0x7fffffffU;
    return magnitude < 0x7f800000U && ((raw & 0x80000000U) == 0 || magnitude == 0);
}

__simt_callee__ __aicore__ inline uint32_t ScoreKey(uint32_t raw)
{
    // Bitwise zero test preserves positive subnormals. Original output bits are not changed.
    return ~((raw & 0x7fffffffU) == 0 ? 0U : raw);
}

// All 256 threads call together, including padding lanes. Two independent inclusive scans
// inside each warp, then eight explicit UB warp totals. Returns block-exclusive ranks/totals.
__simt_callee__ __aicore__ inline void PrefixPair(
    uint32_t a, uint32_t b, __ubuf__ uint32_t *scratch,
    uint32_t &rankA, uint32_t &rankB, uint32_t &totalA, uint32_t &totalB)
{
    const uint32_t thread = AscendC::Simt::GetThreadIdx<0>();
    const uint32_t lane = thread % kWarp;
    const uint32_t warp = thread / kWarp;
    uint32_t sumA = a;
    uint32_t sumB = b;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const uint32_t otherA = __shfl_up(sumA, offset, 32);
        const uint32_t otherB = __shfl_up(sumB, offset, 32);
        if (lane >= static_cast<uint32_t>(offset)) {
            sumA += otherA;
            sumB += otherB;
        }
    }
    if (lane == 31) {
        scratch[kScan + warp] = sumA;
        scratch[kScan + kWarps + warp] = sumB;
    }
    asc_syncthreads();
    rankA = sumA - a;
    rankB = sumB - b;
    totalA = 0;
    totalB = 0;
#pragma unroll
    for (uint32_t w = 0; w < kWarps; ++w) {
        const uint32_t wa = scratch[kScan + w];
        const uint32_t wb = scratch[kScan + kWarps + w];
        if (w < warp) { rankA += wa; rankB += wb; }
        totalA += wa;
        totalB += wb;
    }
    // Protect the totals from the next call until every reader has consumed them.
    asc_syncthreads();
}

// Merge eight warp minima. All threads participate, including exhausted streams.
// Reuse scan totals only after PrefixPair has finished; both barriers protect readers.
__simt_callee__ __aicore__ inline uint32_t MinimumHead(
    uint32_t head, __ubuf__ uint32_t *scratch)
{
    const uint32_t thread = AscendC::Simt::GetThreadIdx<0>();
    uint32_t minimum = head;
#pragma unroll
    for (int32_t stride = 16; stride > 0; stride >>= 1) {
        const uint32_t other = asc_shfl_xor(minimum, stride, 32);
        minimum = other < minimum ? other : minimum;
    }
    if (thread % kWarp == 0) scratch[kScan + thread / kWarp] = minimum;
    asc_syncthreads();
#pragma unroll
    for (uint32_t warp = 0; warp < kWarps; ++warp) {
        const uint32_t other = scratch[kScan + warp];
        minimum = other < minimum ? other : minimum;
    }
    asc_syncthreads();
    return minimum;
}


__simt_callee__ __aicore__ inline uint32_t ReadBits(
    __ubuf__ const uint32_t *scores, __ubuf__ const uint32_t *scratch,
    uint64_t base, uint64_t position, bool compact)
{
    return compact ? scratch[kPositiveBits + position] : scores[base + position];
}

__simt_callee__ __aicore__ inline uint32_t ReadId(
    __ubuf__ const uint32_t *scratch, uint64_t position, bool compact)
{
    return compact ? scratch[kPositiveIds + position] : static_cast<uint32_t>(position);
}

__simt_callee__ __aicore__ inline void WriteSelected(
    __ubuf__ const uint32_t *scores, __gm__ uint32_t *topScores, __gm__ int32_t *topIds,
    __ubuf__ uint32_t *scratch, uint64_t base, uint64_t query,
    uint32_t k, uint32_t docCount, bool selectionValid)
{
    const uint32_t thread = AscendC::Simt::GetThreadIdx<0>();
    const uint32_t lane = thread % kWarp;
    if (thread < kWarp) {
        uint32_t key = scratch[kCandidateKeys + lane];
        uint32_t id = scratch[kCandidateIds + lane];
        // Only the selected K entries plus invalid padding enter this 32-slot network.
#pragma unroll
        for (uint32_t span = 2; span <= kWarp; span <<= 1) {
#pragma unroll
            for (uint32_t stride = span >> 1; stride > 0; stride >>= 1) {
                const uint32_t otherKey = asc_shfl_xor(key, static_cast<int32_t>(stride), 32);
                const uint32_t otherId = asc_shfl_xor(id, static_cast<int32_t>(stride), 32);
                const bool otherBetter = otherKey < key || (otherKey == key && otherId < id);
                const bool selfBetter = key < otherKey || (key == otherKey && id < otherId);
                const bool wantBetter = ((lane & span) == 0) == ((lane & stride) == 0);
                if ((wantBetter && otherBetter) || (!wantBetter && selfBetter)) {
                    key = otherKey;
                    id = otherId;
                }
            }
        }
        if (lane < k) {
            const bool valid = selectionValid && id < docCount;
            topIds[query*k + lane] = valid ? static_cast<int32_t>(id) : -1;
            topScores[query*k + lane] = valid ? scores[base + id] : 0xbf800000U;
        }
    }
    asc_syncthreads(); // candidate UB can now be reused by the next grid-stride query
}

__simt_callee__ __aicore__ inline void SelectLocal(
    __ubuf__ const uint32_t *scores, __gm__ uint32_t *topScores, __gm__ int32_t *topIds,
    __ubuf__ uint32_t *scratch, uint64_t query, uint32_t docCount, uint32_t k)
{
    const uint32_t thread = AscendC::Simt::GetThreadIdx<0>();
    const uint64_t base = 0; // This block's on-chip scores only; output row remains query.
        uint32_t privatePositive = 0, positiveBegin = 0, positiveTotal = 0;
        uint32_t firstZero = kNoId, lowMask = 0, highMask = 0;
        bool compact = false;
        if (docCount > kCapacity && docCount <= kMaskDocs) {
            uint32_t invalid = 0, ordinal = 0;
            for (uint64_t doc = thread; doc < docCount; doc += kThreads, ++ordinal) {
                const uint32_t raw = scores[base + doc];
                if (!ValidBits(raw)) { invalid = 1; continue; }
                if ((raw & 0x7fffffffU) != 0) {
                    ++privatePositive;
                    if (ordinal < 32) lowMask |= 1U << ordinal;
                    else highMask |= 1U << (ordinal - 32);
                } else if (firstZero == kNoId) firstZero = static_cast<uint32_t>(doc);
            }
            uint32_t ignoredRank, invalidTotal;
            PrefixPair(privatePositive, invalid, scratch, positiveBegin, ignoredRank, positiveTotal, invalidTotal);
            if (invalidTotal != 0) {
                if (thread < k) { topScores[query*k + thread] = 0xbf800000U; topIds[query*k + thread] = -1; }
                asc_syncthreads(); return;
            }
            compact = positiveTotal <= kCapacity; // uniform; never truncate at capacity
            if (compact) {
                uint32_t written = 0;
                // Membership was recorded while validating/counting. Only hits reread GM.
                for (uint32_t ordinal = 0; written < privatePositive && ordinal < 64; ++ordinal) {
                    const uint32_t membership = ordinal < 32 ? lowMask : highMask;
                    const uint32_t bit = ordinal < 32 ? ordinal : ordinal - 32;
                    if ((membership & (1U << bit)) != 0) {
                        const uint32_t id = thread + ordinal * kThreads;
                        scratch[kPositiveBits + positiveBegin + written] = scores[base + id];
                        scratch[kPositiveIds + positiveBegin + written] = id;
                        ++written;
                    }
                }
            }
            asc_syncthreads();
        }
        if (compact && positiveTotal <= k) {
            if (thread < kWarp) {
                scratch[kCandidateKeys + thread] = 0xffffffffU;
                scratch[kCandidateIds + thread] = kNoId;
            }
            asc_syncthreads();
            for (uint32_t slot = positiveBegin; slot < positiveBegin + privatePositive; ++slot) {
                scratch[kCandidateKeys + slot] = ScoreKey(scratch[kPositiveBits + slot]);
                scratch[kCandidateIds + slot] = scratch[kPositiveIds + slot];
            }
            asc_syncthreads();
            const uint32_t zerosNeeded = k - positiveTotal;
            uint32_t zeroHead = firstZero, badSmall = 0;
            for (uint32_t rank = 0; rank < zerosNeeded; ++rank) {
                const uint32_t winner = MinimumHead(zeroHead, scratch);
                if (winner == kNoId) { badSmall = 1; break; }
                if (zeroHead == winner) {
                    scratch[kCandidateKeys + positiveTotal + rank] = 0xffffffffU;
                    scratch[kCandidateIds + positiveTotal + rank] = winner;
                    if (rank + 1 < zerosNeeded) {
                        zeroHead = kNoId;
                        for (uint64_t doc = static_cast<uint64_t>(winner) + kThreads;
                             doc < docCount; doc += kThreads) {
                            if ((scores[base + doc] & 0x7fffffffU) == 0) {
                                zeroHead = static_cast<uint32_t>(doc); break;
                            }
                        }
                    }
                }
                asc_syncthreads();
            }
            asc_syncthreads();
            WriteSelected(scores, topScores, topIds, scratch, base, query, k, docCount, badSmall == 0);
            return;
        }
        // A compressed segment is owned by its original producer thread and retains
        // ascending ORIGINAL IDs. Flattened compact positions must never break ties.
        const uint64_t sourceBegin = compact ? positiveBegin : thread;
        const uint64_t sourceEnd = compact ? positiveBegin + privatePositive : docCount;
        const uint64_t sourceStep = compact ? 1 : kThreads;
        uint32_t prefix = 0;
        uint32_t mask = 0;
        uint32_t remaining = k;
        uint32_t badInput = 0;
        // Three integer passes: select the exact Kth key, not an approximate score cutoff.
        for (uint32_t pass = 0; pass < 3; ++pass) {
            const uint32_t shift = pass == 0 ? 21 : (pass == 1 ? 10 : 0);
            const uint32_t bins = pass == 2 ? 1024 : 2048;
            for (uint32_t bin = thread; bin < bins; bin += kThreads) scratch[bin] = 0;
            // One clear owner per UB word, including the private-table padding.
            if (pass == 0) {
                for (uint32_t word = thread; word < kPrivateHistWords; word += kThreads)
                    scratch[kPrivateHist + word] = 0;
            }
            const uint32_t warp = thread/kWarp;
            const uint32_t histBase = pass == 0 && warp != 0 ?
                kPrivateHist + (warp - 1)*kHistStride : 0;
            if (thread == 0) { scratch[kControl] = 0xffffffffU; scratch[kControl + 1] = 0; }
            asc_syncthreads();
            uint32_t invalid = 0;
            for (uint64_t doc = sourceBegin; doc < sourceEnd; doc += sourceStep) {
                const uint32_t raw = ReadBits(scores, scratch, base, doc, compact);
                if (!ValidBits(raw)) { invalid = 1; continue; }
                const uint32_t key = ScoreKey(raw);
                if ((key & mask) == prefix) {
                    // Basic SIMT UB integer atomic, not an algorithm-library operation.
                    atomicAdd(&scratch[histBase + ((key >> shift) & (bins - 1))], 1U);
                }
            }
            asc_syncthreads();
            if (pass == 0) {
                // One owner per bin reads every warp after the preceding barrier.
                // Only that owner overwrites warp0[bin], so in-place reduction is race-free.
                for (uint32_t bin = thread; bin < bins; bin += kThreads) {
                    uint32_t total = scratch[bin];
#pragma unroll
                    for (uint32_t w = 1; w < kWarps; ++w)
                        total += scratch[kPrivateHist + (w - 1)*kHistStride + bin];
                    scratch[bin] = total;
                }
                asc_syncthreads(); // all merged bins ready before prefix selection
            }
            uint32_t unusedA, unusedB, invalidTotal, unusedTotal;
            PrefixPair(invalid, 0, scratch, unusedA, unusedB, invalidTotal, unusedTotal);
            if (invalidTotal != 0) { badInput = 1; break; } // uniform across the block

            const uint32_t chunk = bins / kThreads;
            const uint32_t begin = thread * chunk;
            uint32_t count = 0;
            for (uint32_t j = 0; j < chunk; ++j) count += scratch[begin + j];
            uint32_t before, ignored, total, ignoredTotal;
            PrefixPair(count, 0, scratch, before, ignored, total, ignoredTotal);
            if (total < remaining || remaining == 0) { badInput = 1; break; }
            for (uint32_t j = 0; j < chunk; ++j) {
                const uint32_t countHere = scratch[begin + j];
                if (before < remaining && remaining <= before + countHere) {
                    // Exactly one bin/thread owns this rank; no racing atomic quota.
                    scratch[kControl] = begin + j;
                    scratch[kControl + 1] = remaining - before;
                }
                before += countHere;
            }
            asc_syncthreads();
            const uint32_t bucket = scratch[kControl];
            remaining = scratch[kControl + 1];
            if (bucket >= bins || remaining == 0) { badInput = 1; break; }
            prefix |= bucket << shift;
            mask |= (bins - 1) << shift;
            // Prevent the next pass's control reset while any thread is still reading it.
            asc_syncthreads();
        }
        if (badInput != 0) {
            if (thread < k) { topScores[query*k + thread] = 0xbf800000U; topIds[query*k + thread] = -1; }
            asc_syncthreads();
            return; // explicit invalid result; host must reject, never substitute a CPU answer
        }
        const uint32_t strictCount = k - remaining;
        if (thread < kWarp) {
            scratch[kCandidateKeys + thread] = 0xffffffffU;
            scratch[kCandidateIds + thread] = kNoId;
        }
        asc_syncthreads();
        // Scan each thread's ordered ID stream once. Keep only a strict count
        // and its first equal-score ID, not a per-thread K-entry shortlist.
        uint32_t privateStrict = 0;
        uint32_t equalHead = kNoId;
        uint64_t equalPosition = 0;
        for (uint64_t doc = sourceBegin; doc < sourceEnd; doc += sourceStep) {
            const uint32_t key = ScoreKey(ReadBits(scores, scratch, base, doc, compact));
            if (key < prefix) ++privateStrict;
            if (key == prefix && equalHead == kNoId) { equalHead = ReadId(scratch, doc, compact); equalPosition = doc; }
        }
        uint32_t strictBegin, ignoredRank, strictTotal, ignoredTotal;
        PrefixPair(privateStrict, 0, scratch, strictBegin, ignoredRank, strictTotal, ignoredTotal);
        if (strictTotal != strictCount) {
            if (thread < k) { topScores[query*k + thread] = 0xbf800000U; topIds[query*k + thread] = -1; }
            asc_syncthreads();
            return; // uniform failure, never return a partial answer
        }
        // Only owners of strict candidates rescan. Disjoint ranges come from the
        // single whole-row prefix above; stop at the last local strict candidate.
        if (privateStrict != 0) {
            uint32_t written = 0;
            for (uint64_t doc = sourceBegin; doc < sourceEnd; doc += sourceStep) {
                const uint32_t key = ScoreKey(ReadBits(scores, scratch, base, doc, compact));
                if (key < prefix) {
                    const uint32_t slot = strictBegin + written;
                    scratch[kCandidateKeys + slot] = key;
                    scratch[kCandidateIds + slot] = ReadId(scratch, doc, compact);
                    if (++written == privateStrict) break;
                }
            }
        }
        asc_syncthreads(); // all strict candidates visible before head merging/final sorting
        for (uint32_t rank = 0; rank < remaining; ++rank) {
            const uint32_t winner = MinimumHead(equalHead, scratch);
            if (winner == kNoId) { badInput = 1; break; } // uniform across all 256 threads
            if (equalHead == winner) {
                const uint32_t slot = strictCount + rank;
                scratch[kCandidateKeys + slot] = prefix;
                scratch[kCandidateIds + slot] = winner;
                // Only the winning stream advances. Do not scan for a successor
                // after the last required equal candidate has already been chosen.
                if (rank + 1 < remaining) {
                    equalHead = kNoId;
                    for (uint64_t doc = equalPosition + sourceStep;
                         doc < sourceEnd; doc += sourceStep) {
                        if (ScoreKey(ReadBits(scores, scratch, base, doc, compact)) == prefix) {
                            equalHead = ReadId(scratch, doc, compact);
                            equalPosition = doc;
                            break;
                        }
                    }
                }
            }
            asc_syncthreads(); // winner's candidate write and next head ready for all threads
        }
        asc_syncthreads();
        WriteSelected(scores, topScores, topIds, scratch, base, query, k, docCount,
                      badInput == 0 && strictTotal == strictCount);
}

// Three disjoint regions: full local scores, score metadata, accepted selector scratch.
// 108,416 bytes of explicit UB inside the Host's 128 KiB dynamic-UB declaration.
// SIMT DCache (at least32 KiB) and the compiler reserve are separate capacity checks.
constexpr uint32_t kMaxDocs = 8192, kTerms = 64;
constexpr uint32_t kBegins = kMaxDocs, kEnds = kBegins + 2*kTerms;
constexpr uint32_t kTermIds = kEnds + 2*kTerms, kScoreControl = kTermIds + kTerms;
constexpr uint32_t kSelection = kScoreControl + 16;
constexpr uint32_t kBufferBytes = kSelection*sizeof(uint32_t) + kSelectScratchBytes;
static_assert(kBufferBytes == 108416 && kSelection%8 == 0);
constexpr float kMaxFloat = 3.402823466e+38F;

__simt_callee__ __aicore__ inline int64_t LowerBound(
    __gm__ const int32_t *docs, int64_t begin, int64_t end, uint32_t target)
{
    while (begin < end) {
        const int64_t mid = begin + (end-begin)/2;
        if (docs[mid] < static_cast<int32_t>(target)) begin = mid+1;
        else end = mid;
    }
    return begin;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(kThreads) inline void FusedScoreTopKSimt(
    __gm__ const int64_t *termOffsets, __gm__ const int32_t *postingDocs,
    __gm__ const float *postingTf, __gm__ const float *docNorm, __gm__ const float *idf,
    __gm__ const int64_t *queryOffsets, __gm__ const int32_t *queryTerms,
    __gm__ uint32_t *partialScores, __gm__ int32_t *partialIds, __gm__ float *debugScores,
    __ubuf__ uint32_t *memory, uint32_t queryCount, uint32_t docCount, uint32_t vocabSize,
    uint32_t postingCount, uint32_t queryStorageCount, uint32_t shards, uint32_t k,
    uint32_t debug, uint32_t blockId, uint32_t blockNum)
{
    const uint32_t thread = AscendC::Simt::GetThreadIdx<0>();
    const uint32_t width = docCount/shards;
    auto *scores = reinterpret_cast<__ubuf__ float *>(memory);
    auto *begins = reinterpret_cast<__ubuf__ int64_t *>(memory+kBegins);
    auto *ends = reinterpret_cast<__ubuf__ int64_t *>(memory+kEnds);
    auto *terms = reinterpret_cast<__ubuf__ int32_t *>(memory+kTermIds);
    auto *control = memory+kScoreControl;
    auto *bounds = reinterpret_cast<__ubuf__ int64_t *>(control+8);
    // A disabled debug output is exactly one float, overwritten on every invocation.
    if (debug == 0 && blockId == 0 && thread == 0) debugScores[0] = 0.0F;
    const uint64_t tasks = static_cast<uint64_t>(queryCount)*shards;
    for (uint64_t task = blockId; task < tasks; task += blockNum) {
        const uint32_t query = static_cast<uint32_t>(task/shards);
        const uint32_t first = static_cast<uint32_t>(task%shards)*width;
        const uint32_t last = first+width;
        if (thread == 0) {
            bounds[0] = queryOffsets[query]; bounds[1] = queryOffsets[query+1];
            const bool valid = bounds[0]>=0 && bounds[1]>=bounds[0] &&
                static_cast<uint64_t>(bounds[1])<=queryStorageCount && bounds[1]-bounds[0]<=kTerms;
            control[0] = valid ? 0U : 1U;
            control[1] = valid ? static_cast<uint32_t>(bounds[1]-bounds[0]) : 0U;
        }
        for (uint32_t doc = thread; doc < width; doc += kThreads) scores[doc] = 0.0F;
        asc_syncthreads();
        const uint32_t count = control[1];
        // The resident index was validated before upload: each term's postings have
        // strictly increasing, unique document IDs. Exactly one writer per term/doc.
        if (thread < count) {
            const int32_t term = queryTerms[bounds[0]+thread];
            terms[thread] = -1; begins[thread] = ends[thread] = 0;
            if (term < 0) atomicAdd(&control[0], 1U);
            else if (static_cast<uint32_t>(term) < vocabSize) {
                const int64_t begin = termOffsets[term], end = termOffsets[term+1];
                if (begin<0 || end<begin || static_cast<uint64_t>(end)>postingCount) atomicAdd(&control[0], 1U);
                else {
                    const int64_t lo = LowerBound(postingDocs, begin, end, first);
                    begins[thread] = lo; ends[thread] = LowerBound(postingDocs, lo, end, last);
                    terms[thread] = term;
                }
            }
        }
        asc_syncthreads();
        // One complete on-chip shard: no per-tile cursor update or warp count.
        // Query term order is unchanged, including repeated words. The barrier
        // after each term makes its FP32 additions visible before the next term.
        for (uint32_t slot = 0; slot < count; ++slot) {
            const int32_t term = terms[slot];
            if (term >= 0) {
                for (int64_t pos = begins[slot]+thread; pos < ends[slot]; pos += kThreads) {
                    const int32_t doc = postingDocs[pos];
                    if (doc < static_cast<int32_t>(first) || doc >= static_cast<int32_t>(last)) {
                        atomicAdd(&control[0], 1U); continue;
                    }
                    const float tf = postingTf[pos], norm = docNorm[doc], weight = idf[term];
                    if (!(tf>=1.0F && tf<=16777216.0F && norm>0.0F && norm<=kMaxFloat &&
                          weight>=0.0F && weight<=kMaxFloat)) { atomicAdd(&control[0], 1U); continue; }
                    const float numerator = tf*2.2F;
                    const float contribution = (weight*numerator)/(tf+norm);
                    const uint32_t local = static_cast<uint32_t>(doc)-first;
                    const float total = scores[local]+contribution;
                    if (!(contribution>=0.0F && contribution<=kMaxFloat && total>=0.0F && total<=kMaxFloat))
                        atomicAdd(&control[0], 1U);
                    else scores[local] = total;
                }
            }
            asc_syncthreads();
        }
        const bool bad = control[0] != 0;
        asc_syncthreads(); // all readers captured score control before any later task can reset it
        if (debug != 0) {
            const uint64_t base = static_cast<uint64_t>(query)*docCount+first;
            for (uint32_t doc = thread; doc < width; doc += kThreads)
                debugScores[base+doc] = bad ? -1.0F : scores[doc];
        }
        if (bad) {
            if (thread < k) { partialScores[task*k+thread] = 0xbf800000U; partialIds[task*k+thread] = -1; }
        } else {
            SelectLocal(memory, partialScores, partialIds, memory+kSelection, task, width, k);
        }
        asc_syncthreads(); // grid-stride next task cannot overwrite scores while selection is reading
    }
}
}

extern "C" __global__ __aicore__ void bm25_fused_score_top_k(
    GM_ADDR term_offsets, GM_ADDR posting_doc_ids, GM_ADDR posting_tf, GM_ADDR doc_norm,
    GM_ADDR idf, GM_ADDR query_offsets, GM_ADDR query_term_ids,
    GM_ADDR partial_scores, GM_ADDR partial_ids, GM_ADDR debug_scores, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(data, tiling);
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, kBufferBytes); auto local = buffer.Get<uint32_t>();
    AscendC::Simt::VF_CALL<FusedScoreTopKSimt>(AscendC::Simt::Dim3{kThreads,1,1},
        reinterpret_cast<__gm__ const int64_t *>(term_offsets), reinterpret_cast<__gm__ const int32_t *>(posting_doc_ids),
        reinterpret_cast<__gm__ const float *>(posting_tf), reinterpret_cast<__gm__ const float *>(doc_norm),
        reinterpret_cast<__gm__ const float *>(idf), reinterpret_cast<__gm__ const int64_t *>(query_offsets),
        reinterpret_cast<__gm__ const int32_t *>(query_term_ids), reinterpret_cast<__gm__ uint32_t *>(partial_scores),
        reinterpret_cast<__gm__ int32_t *>(partial_ids), reinterpret_cast<__gm__ float *>(debug_scores),
        reinterpret_cast<__ubuf__ uint32_t *>(local.GetPhyAddr()), data.queryCount, data.docCount, data.vocabSize,
        data.postingCount, data.queryStorageCount, data.shards, data.k, data.debug,
        static_cast<uint32_t>(AscendC::GetBlockIdx()), data.blockNum);
}
