// Independent exact Top-K: bounded positive compaction, original radix/collection, exact dense path.
// No IndexSDK/Faiss/high-level TopK calls. Pending actual CANN compilation and A5 validation.
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_warp_functions.h"
#include "simt_api/device_sync_functions.h"

namespace {
constexpr uint32_t kThreads = 128;
constexpr uint32_t kWarp = 32;
constexpr uint32_t kWarps = kThreads / kWarp;
// Explicit TBuf allocation, never an assumed UB address. Layout in uint32_t words:
// histogram[2048], scan totals[8], control[8], selected keys/IDs[32], positive raw bits/IDs[1024].
constexpr uint32_t kScan = 2048;
constexpr uint32_t kControl = kScan + 8;
constexpr uint32_t kCandidateKeys = kControl + 8;
constexpr uint32_t kCandidateIds = kCandidateKeys + 32;
constexpr uint32_t kCapacity = 1024;
constexpr uint32_t kMaskDocs = kThreads * 64; // two 32-bit membership masks per thread
constexpr uint32_t kPositiveBits = kCandidateIds + 32;
constexpr uint32_t kPositiveIds = kPositiveBits + kCapacity;
constexpr uint32_t kScratchBytes = (kPositiveIds + kCapacity) * sizeof(uint32_t); // 16704
static_assert(kScratchBytes == 16704 && kScratchBytes % 32 == 0);
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

// All 128 threads call together, including padding lanes. Two independent inclusive scans
// inside each warp, then four explicit UB warp totals. Returns block-exclusive ranks/totals.
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

// Merge four warp minima. All threads participate, including exhausted streams.
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
    __gm__ const uint32_t *scores, __ubuf__ const uint32_t *scratch,
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
    __gm__ const uint32_t *scores, __gm__ uint32_t *topScores, __gm__ int32_t *topIds,
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

__simt_vf__ __aicore__ LAUNCH_BOUND(kThreads) inline void RadixTopKSimt(
    __gm__ const uint32_t *scores, __gm__ uint32_t *topScores, __gm__ int32_t *topIds,
    __ubuf__ uint32_t *scratch, uint32_t queryCount, uint32_t docCount, uint32_t k,
    uint32_t blockId, uint32_t blockNum)
{
    const uint32_t thread = AscendC::Simt::GetThreadIdx<0>();
    for (uint64_t query = blockId; query < queryCount; query += blockNum) {
        const uint64_t base = query * docCount;
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
                asc_syncthreads(); continue;
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
            continue;
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
            if (thread == 0) { scratch[kControl] = 0xffffffffU; scratch[kControl + 1] = 0; }
            asc_syncthreads();
            uint32_t invalid = 0;
            for (uint64_t doc = sourceBegin; doc < sourceEnd; doc += sourceStep) {
                const uint32_t raw = ReadBits(scores, scratch, base, doc, compact);
                if (!ValidBits(raw)) { invalid = 1; continue; }
                const uint32_t key = ScoreKey(raw);
                if ((key & mask) == prefix) {
                    // Basic SIMT UB integer atomic, not an algorithm-library operation.
                    atomicAdd(&scratch[(key >> shift) & (bins - 1)], 1U);
                }
            }
            asc_syncthreads();
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
            continue; // explicit invalid result; host must reject, never substitute a CPU answer
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
            continue; // uniform failure, never return a partial answer
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
            if (winner == kNoId) { badInput = 1; break; } // uniform across all 128 threads
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
}
}

extern "C" __global__ __aicore__ void bm25_top_k(
    GM_ADDR scores, GM_ADDR top_scores, GM_ADDR top_ids, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(data, tiling);
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, kScratchBytes);
    auto local = buffer.Get<uint32_t>();
    AscendC::Simt::VF_CALL<RadixTopKSimt>(
        AscendC::Simt::Dim3{kThreads, 1, 1}, reinterpret_cast<__gm__ const uint32_t *>(scores),
        reinterpret_cast<__gm__ uint32_t *>(top_scores), reinterpret_cast<__gm__ int32_t *>(top_ids),
        reinterpret_cast<__ubuf__ uint32_t *>(local.GetPhyAddr()), data.queryCount, data.docCount,
        data.k, static_cast<uint32_t>(AscendC::GetBlockIdx()), data.blockNum);
}
