// Isolated fused FP32 scoring + exact local selection for Ascend950. SOURCE ONLY.
// Selection is adapted from the accepted self-developed positive-compaction implementation;
// only its score pointer address space and single-row control wrapper change.
// No full-score GM array in debug=0. No CPU fallback or high-level TopK library.
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_warp_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_functions.h"

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
constexpr uint32_t kSelectScratchBytes = (kPositiveIds + kCapacity) * sizeof(uint32_t); // 16736
static_assert(kSelectScratchBytes == 16736 && kSelectScratchBytes % 32 == 0);
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
            topIds[query*k + lane] = valid ? static_cast<int32_t>(scratch[kControl+7]+id) : -1;
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
// 50,848 bytes of explicit UB inside the Host's 64 KiB dynamic-UB declaration.
// SIMT DCache (at least32 KiB) and the compiler reserve are separate capacity checks.
constexpr uint32_t kMaxDocs = 8192, kTerms = 64;
constexpr uint32_t kBegins = kMaxDocs, kEnds = kBegins + 2*kTerms;
constexpr uint32_t kTermIds = kEnds + 2*kTerms, kScoreControl = kTermIds + kTerms;
constexpr uint32_t kSelection = kScoreControl + 16;
constexpr uint32_t kBufferBytes = kSelection*sizeof(uint32_t) + kSelectScratchBytes;
static_assert(kBufferBytes == 50848 && kSelection%8 == 0);
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


/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE-CANN in this directory for the full text of the License.
 *
 * Histogram register conversion adapted for Ascend950 from ops-math a8d06fda,
 * math/top_k_v2/op_kernel/arch35/radix_sort_topk_b32.h and radix_sort_topk_b8.h.
 * Own bounded on-chip selection, not the official TopKV2 implementation/library.
 */

namespace V = AscendC::Reg;
// Existing 2048-word histogram area is reused. No additional TBuf or dynamic UB.
constexpr uint32_t kOwnerBegin = 256, kOwnerCount = 512;
static_assert(kOwnerCount + kThreads <= kScan);

template <bool BitmapReady>
__simt_callee__ __aicore__ inline void PackForSimd(
    __ubuf__ const uint32_t *scores, __ubuf__ uint32_t *scratch, uint32_t width, uint32_t k)
{
    const uint32_t tid = AscendC::Simt::GetThreadIdx<0>();
    if constexpr (BitmapReady) {
        // 正常评分已用warp聚合追加局部文档ID；这里只补连续分数，不再读取位图或做块前缀。
        const uint32_t total = scratch[kControl];
        if (tid == 0) {
            scratch[kControl+1] = total > k && total <= kCapacity ? 1U : 0U;
            scratch[kControl+4] = 1U; // 无序、按线程跨步访问的直接追加布局
        }
        if (total > k && total <= kCapacity) {
            for (uint32_t pos = tid; pos < total; pos += kThreads) {
                const uint32_t doc = scratch[kPositiveIds+pos];
                scratch[kPositiveBits+pos] = scores[doc];
            }
        }
        asc_syncthreads();
    } else {
        // debug=2原始分数检查保留已验收的稠密验证、保序Pack和Sparse发射。
        uint32_t positive = 0, invalid = 0, lowMask = 0, highMask = 0, ordinal = 0;
        for (uint32_t doc = tid; doc < width; doc += kThreads, ++ordinal) {
            const uint32_t bits = scores[doc];
            invalid |= !ValidBits(bits);
            if ((bits & 0x7fffffffU) != 0) {
                ++positive;
                if (ordinal < 32) lowMask |= 1U << ordinal;
                else highMask |= 1U << (ordinal-32);
            }
        }
        uint32_t begin, unused, total, bad;
        PrefixPair(positive, invalid, scratch, begin, unused, total, bad);
        if (tid == 0) {
            scratch[kControl] = total;
            scratch[kControl+1] = bad != 0 ? 2U : (total > k && total <= kCapacity ? 1U : 0U);
            scratch[kControl+4] = 0U; // owner连续段、文档号升序布局
        }
        scratch[kOwnerBegin+tid] = begin;
        scratch[kOwnerCount+tid] = positive;
        if (bad == 0 && total > k && total <= kCapacity) {
            uint32_t written = 0;
            for (uint32_t half = 0; half < 2 && written < positive; ++half) {
                uint32_t remaining = half == 0 ? lowMask : highMask;
                while (remaining != 0 && written < positive) {
                    const uint32_t lowest = remaining & (0U-remaining);
                    const uint32_t bit = static_cast<uint32_t>(__popc(lowest-1U));
                    const uint32_t pos = half*32+bit;
                    const uint32_t doc = tid+pos*kThreads;
                    scratch[kPositiveBits+begin+written] = scores[doc];
                    scratch[kPositiveIds+begin+written] = doc;
                    ++written;
                    remaining ^= lowest;
                }
            }
        }
        asc_syncthreads();
    }
}

/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 * See ../fused_simd_select/LICENSE-CANN for the full license.
 * Histogram packing follows the retained ops-math a8d06fda-derived implementation.
 * Register-only boundary/rank control is a new exact integer adaptation.
 */

// All four byte passes remain in one vector execution scope. Each Reduce MIN/MAX
// produces value plus index; Duplicate explicitly broadcasts ONLY its lowest value.
// Integer16 represents histogram counts <=1024 and bin indices <=255, never FP32 scores.
__aicore__ inline void RegisterThreshold(
    __ubuf__ uint32_t *input, __ubuf__ uint32_t *result, uint32_t count, uint32_t k)
{
    __VEC_SCOPE__ {
        auto all32=V::CreateMask<uint32_t>();
        auto all16=V::CreateMask<uint16_t>();
        V::RegTensor<uint32_t> prefix,maskVec,zero32;
        V::RegTensor<uint16_t> rank,zero16,indexLo,indexHi,missing;
        V::Duplicate(prefix,0U,all32); V::Duplicate(zero32,0U,all32);
        V::Duplicate(rank,static_cast<uint16_t>(k),all16);
        V::Duplicate(zero16,static_cast<uint16_t>(0),all16);
        V::Duplicate(missing,static_cast<uint16_t>(0xffff),all16);
        V::Arange((V::RegTensor<int16_t>&)indexLo,static_cast<int16_t>(0));
        V::Arange((V::RegTensor<int16_t>&)indexHi,static_cast<int16_t>(128));
        uint32_t prefixMask=0;
        const uint16_t tiles=static_cast<uint16_t>((count+63)/64);
        // CANN9.2 vector loops require a uint16_t induction variable.
        // Derive the descending byte shift from a canonical ascending pass loop.
        const uint16_t passes=4;
        for (uint16_t pass=0;pass<passes;++pass) {
            const int16_t shift=static_cast<int16_t>(24-8*pass);
            V::RegTensor<uint16_t> histLo,histHi;
            V::Duplicate(histLo,static_cast<uint16_t>(0),all16);
            V::Duplicate(histHi,static_cast<uint16_t>(0),all16);
            V::Duplicate(maskVec,prefixMask,all32);
            auto *cursor=input;
            uint32_t remaining=count;
            for (uint16_t tile=0;tile<tiles;++tile) {
                V::RegTensor<uint32_t> bits,keys,masked,shifted;
                auto valid=V::UpdateMask<uint32_t>(remaining);
                V::LoadAlign<uint32_t,V::PostLiteral::POST_MODE_UPDATE>(bits,cursor,64);
                V::Not(keys,bits,valid);
                V::And(masked,keys,maskVec,valid);
                V::MaskReg compared,selected,mask16,mask8;
                V::Compare<uint32_t,AscendC::CMPMODE::EQ>(compared,masked,prefix,valid);
                V::And(selected,compared,valid,all32);
                V::ShiftRights<uint32_t,int16_t>(shifted,keys,shift,selected);
                V::RegTensor<uint16_t> low16,unused16;
                V::RegTensor<uint8_t> low8,unused8;
                V::DeInterleave(low16,unused16,(V::RegTensor<uint16_t>&)shifted,(V::RegTensor<uint16_t>&)zero32);
                V::DeInterleave(low8,unused8,(V::RegTensor<uint8_t>&)low16,(V::RegTensor<uint8_t>&)zero16);
                V::Pack(mask16,selected); V::Pack(mask8,mask16);
                V::Histograms<uint8_t,uint16_t,V::HistogramsBinType::BIN0,V::HistogramsType::ACCUMULATE>(histLo,low8,mask8);
                V::Histograms<uint8_t,uint16_t,V::HistogramsBinType::BIN1,V::HistogramsType::ACCUMULATE>(histHi,low8,mask8);
            }
            V::MaskReg reachesLo,reachesHi;
            V::Compare<uint16_t,AscendC::CMPMODE::GE>(reachesLo,histLo,rank,all16);
            V::Compare<uint16_t,AscendC::CMPMODE::GE>(reachesHi,histHi,rank,all16);
            V::RegTensor<uint16_t> candidateLo,candidateHi,candidates,bucketReduced,bucket;
            V::Select(candidateLo,indexLo,missing,reachesLo);
            V::Select(candidateHi,indexHi,missing,reachesHi);
            V::Min(candidates,candidateLo,candidateHi,all16);
            V::Reduce<V::ReduceType::MIN>(bucketReduced,candidates,all16);
            V::Duplicate<uint16_t>(bucket,bucketReduced,all16);

            // CDF immediately before the selected bucket: max of CDF[i], i<bucket.
            // Selected bin0 has no preceding entries, so explicit zero padding is required.
            V::MaskReg beforeLo,beforeHi;
            V::Compare<uint16_t,AscendC::CMPMODE::LT>(beforeLo,indexLo,bucket,all16);
            V::Compare<uint16_t,AscendC::CMPMODE::LT>(beforeHi,indexHi,bucket,all16);
            V::RegTensor<uint16_t> countsLo,countsHi,counts,beforeReduced,before,nextRank;
            V::Select(countsLo,histLo,zero16,beforeLo);
            V::Select(countsHi,histHi,zero16,beforeHi);
            V::Max(counts,countsLo,countsHi,all16);
            V::Reduce<V::ReduceType::MAX>(beforeReduced,counts,all16);
            V::Duplicate<uint16_t>(before,beforeReduced,all16);
            V::Sub(nextRank,rank,before,all16);
            V::Duplicate<uint16_t>(rank,nextRank,all16);

            V::RegTensor<uint32_t> bucket32,unused32,shiftedBucket;
            V::Interleave((V::RegTensor<uint16_t>&)bucket32,(V::RegTensor<uint16_t>&)unused32,bucket,zero16);
            V::ShiftLefts<uint32_t,int16_t>(shiftedBucket,bucket32,shift,all32);
            V::Or(prefix,prefix,shiftedBucket,all32);
            prefixMask|=255U<<shift;
        }
        V::RegTensor<uint32_t> rank32,unused32;
        V::Interleave((V::RegTensor<uint16_t>&)rank32,(V::RegTensor<uint16_t>&)unused32,rank,zero16);
        // Two aligned stores into the old CDF area, words[0..127]. Owner metadata
        // starts at word256 and is not overwritten. No extra UB allocation.
        V::StoreAlign<uint32_t>(result,prefix,all32);
        V::StoreAlign<uint32_t>(result+64,rank32,all32);
    }
}

__aicore__ inline void VectorToScalar()
{
    const auto e=static_cast<event_t>(::GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
    AscendC::SetFlag<AscendC::HardEvent::V_S>(e); AscendC::WaitFlag<AscendC::HardEvent::V_S>(e);
}
__aicore__ inline void ScalarToVector()
{
    const auto e=static_cast<event_t>(::GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V));
    AscendC::SetFlag<AscendC::HardEvent::S_V>(e); AscendC::WaitFlag<AscendC::HardEvent::S_V>(e);
}
__aicore__ inline void FindSimdThreshold(AscendC::LocalTensor<uint32_t> local,uint32_t k)
{
    const uint32_t base=kSelection;
    VectorToScalar();
    if (local.GetValue(base+kControl+1)!=1) return;
    const uint32_t count=local.GetValue(base+kControl);
    auto *memory=reinterpret_cast<__ubuf__ uint32_t *>(local.GetPhyAddr());
    RegisterThreshold(memory+base+kPositiveBits,memory+base,count,k);
    VectorToScalar();
    const uint32_t prefix=local.GetValue(base),rank=local.GetValue(base+64),raw=~prefix;
    if (rank==0 || rank>k || raw==0 || raw>=0x7f800000U) local.SetValue(base+kControl+1,2U);
    local.SetValue(base+kControl+2,prefix); local.SetValue(base+kControl+3,rank);
    ScalarToVector();
}

__simt_vf__ __aicore__ LAUNCH_BOUND(kThreads) inline void FusedScorePackSimt(
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
    if (thread==0) memory[kSelection+kControl+7]=(blockId%shards)*width;
    auto *scores = reinterpret_cast<__ubuf__ float *>(memory);
    auto *begins = reinterpret_cast<__ubuf__ int64_t *>(memory+kBegins);
    auto *ends = reinterpret_cast<__ubuf__ int64_t *>(memory+kEnds);
    auto *terms = reinterpret_cast<__ubuf__ int32_t *>(memory+kTermIds);
    auto *control = memory+kScoreControl;
    auto *bounds = reinterpret_cast<__ubuf__ int64_t *>(control+8);
    auto *bitmap = memory+kSelection;
    if (debug == 2) {
        const uint32_t query = blockId/shards, first = (blockId%shards)*width;
        const auto *bits = reinterpret_cast<__gm__ const uint32_t *>(postingTf);
        auto *out = reinterpret_cast<__gm__ uint32_t *>(debugScores);
        for (uint32_t d = thread; d < width; d += kThreads) {
            const uint64_t at = static_cast<uint64_t>(query)*docCount+first+d;
            memory[d] = bits[at]; out[at] = bits[at];
        }
        asc_syncthreads(); PackForSimd<false>(memory, memory+kSelection, width, k); return;
    }
    // A disabled debug output is exactly one float, overwritten on every invocation.
    if (debug == 0 && blockId == 0 && thread == 0) debugScores[0] = 0.0F;
    const uint64_t task = blockId;
    {
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
        if (thread == 0) bitmap[kControl] = 0; // 本任务直接追加游标
        asc_syncthreads();
        const uint32_t count = control[1];
        // The resident index was validated before upload: each term's postings have
        // strictly increasing, unique document IDs. Exactly one writer per term/doc.
        if (thread < count) {
            const int32_t term = queryTerms[bounds[0]+thread];
            terms[thread] = -1; begins[thread] = ends[thread] = 0;
            if (term < 0) atomicAdd(&control[0], 1U);
            else if (static_cast<uint32_t>(term) < vocabSize) {
                const uint64_t entry = static_cast<uint64_t>(term)*(shards+1)+task%shards;
                const int64_t begin = termOffsets[entry], end = termOffsets[entry+1];
                if (begin<0 || end<begin || static_cast<uint64_t>(end)>postingCount) atomicAdd(&control[0], 1U);
                else {
                    begins[thread] = begin; ends[thread] = end;
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
                // 每个warp具有相同迭代次数，ballot/shuffle始终在warp完整汇合点执行。
                const uint32_t lane = thread%kWarp;
                const int64_t warpBegin = begins[slot]+(thread/kWarp)*kWarp;
                const int64_t span = ends[slot]-warpBegin;
                const uint32_t iterations = span > 0 ? static_cast<uint32_t>((span+kThreads-1)/kThreads) : 0U;
                int64_t pos = warpBegin+lane;
                for (uint32_t iteration = 0; iteration < iterations; ++iteration, pos += kThreads) {
                    const bool active = pos < ends[slot];
                    uint32_t local = 0;
                    bool firstPositive = false;
                    if (active) {
                    const int32_t doc = postingDocs[pos];
                    if (doc < static_cast<int32_t>(first) || doc >= static_cast<int32_t>(last)) {
                        atomicAdd(&control[0], 1U);
                    } else {
                    const float tf = postingTf[pos], norm = docNorm[doc], weight = idf[term];
                    if (!(tf>=1.0F && tf<=16777216.0F && norm>0.0F && norm<=kMaxFloat &&
                          weight>=0.0F && weight<=kMaxFloat)) { atomicAdd(&control[0], 1U); }
                    else {
                    const float numerator = tf*2.2F;
                    const float contribution = (weight*numerator)/(tf+norm);
                    local = static_cast<uint32_t>(doc)-first;
                    const float previous = scores[local];
                    const float total = previous+contribution;
                    if (!(contribution>=0.0F && contribution<=kMaxFloat && total>=0.0F && total<=kMaxFloat))
                        atomicAdd(&control[0], 1U);
                    else {
                        scores[local] = total;
                        // 同词文档唯一、词间有屏障；首次由零变正的文档至多追加一次。
                        if (previous == 0.0F && total > 0.0F) firstPositive = true;
                    }
                    }
                    }
                    }
                    // 一次warp原子分配替代每个正分一次原子位图更新。槽位总数包含有界数组外的溢出。
                    const uint32_t hits = asc_ballot(static_cast<int32_t>(firstPositive));
                    if (hits != 0) {
                        const uint32_t lower = (1U << lane)-1U;
                        const uint32_t rank = static_cast<uint32_t>(__popc(hits & lower));
                        uint32_t warpBase = 0;
                        if (lane == 0) warpBase = atomicAdd(bitmap+kControl, static_cast<uint32_t>(__popc(hits)));
                        warpBase = asc_shfl(warpBase, 0, kWarp);
                        if (firstPositive) {
                            const uint32_t slot = warpBase+rank;
                            if (slot < kCapacity) bitmap[kPositiveIds+slot] = local;
                        }
                    }
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
            if (thread == 0) memory[kSelection+kControl+1] = 2U;
            if (thread < k) { partialScores[task*k+thread] = 0xbf800000U; partialIds[task*k+thread] = -1; }
        } else {
            PackForSimd<true>(memory, memory+kSelection, width, k);
        }
        asc_syncthreads(); // grid-stride next task cannot overwrite scores while selection is reading
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(kThreads) inline void FusedCollectSimt(
    __ubuf__ const uint32_t *scores, __ubuf__ uint32_t *scratch,
    __gm__ uint32_t *topScores, __gm__ int32_t *topIds,
    uint32_t query, uint32_t docCount, uint32_t k)
{
    const uint32_t thread = AscendC::Simt::GetThreadIdx<0>();
    const uint32_t mode = scratch[kControl+1];
    if (mode == 0) { SelectLocal(scores, topScores, topIds, scratch, query, docCount, k); return; }
    if (mode != 1) {
        if (thread < k) { topScores[query*k+thread] = 0xbf800000U; topIds[query*k+thread] = -1; }
        asc_syncthreads(); return;
    }
    const uint32_t prefix = scratch[kControl+2], remaining = scratch[kControl+3];
    const uint64_t base = 0;
    const bool compact = true;
    const bool direct = scratch[kControl+4] != 0;
    const uint64_t sourceBegin = direct ? thread : scratch[kOwnerBegin+thread];
    const uint64_t sourceEnd = direct ? scratch[kControl] : sourceBegin+scratch[kOwnerCount+thread];
    const uint64_t sourceStep = direct ? kThreads : 1;
    uint32_t badInput = 0;
    asc_syncthreads(); // consume owner metadata before the accepted collector reuses scan scratch
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
        // 无序流按ID寻找下一名，不保存紧凑位置。
        for (uint64_t doc = sourceBegin; doc < sourceEnd; doc += sourceStep) {
            const uint32_t key = ScoreKey(ReadBits(scores, scratch, base, doc, compact));
            if (key < prefix) ++privateStrict;
            if (key == prefix) {
                const uint32_t id = ReadId(scratch, doc, compact);
                if (id < equalHead) equalHead = id; // 直接追加无需有序，先取本线程最小同分ID
            }
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
                    // 无序紧凑流不能从旧位置前进；严格选择大于上个赢家ID的最小同分文档。
                    for (uint64_t doc = sourceBegin; doc < sourceEnd; doc += sourceStep) {
                        if (ScoreKey(ReadBits(scores, scratch, base, doc, compact)) == prefix) {
                            const uint32_t id = ReadId(scratch, doc, compact);
                            if (id > winner && id < equalHead) equalHead = id;
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

extern "C" __global__ __aicore__ void bm25_fused_score_top_k(
    GM_ADDR term_offsets, GM_ADDR posting_doc_ids, GM_ADDR posting_tf, GM_ADDR doc_norm,
    GM_ADDR idf, GM_ADDR query_offsets, GM_ADDR query_term_ids,
    GM_ADDR partial_scores, GM_ADDR partial_ids, GM_ADDR debug_scores, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(data, tiling);
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, kBufferBytes); auto local = buffer.Get<uint32_t>();
    const uint32_t tasks = data.queryCount*data.shards;
    for (uint32_t task = AscendC::GetBlockIdx(); task < tasks; task += data.blockNum) {
    AscendC::Simt::VF_CALL<FusedScorePackSimt>(AscendC::Simt::Dim3{kThreads,1,1},
        reinterpret_cast<__gm__ const int64_t *>(term_offsets), reinterpret_cast<__gm__ const int32_t *>(posting_doc_ids),
        reinterpret_cast<__gm__ const float *>(posting_tf), reinterpret_cast<__gm__ const float *>(doc_norm),
        reinterpret_cast<__gm__ const float *>(idf), reinterpret_cast<__gm__ const int64_t *>(query_offsets),
        reinterpret_cast<__gm__ const int32_t *>(query_term_ids), reinterpret_cast<__gm__ uint32_t *>(partial_scores),
        reinterpret_cast<__gm__ int32_t *>(partial_ids), reinterpret_cast<__gm__ float *>(debug_scores),
        reinterpret_cast<__ubuf__ uint32_t *>(local.GetPhyAddr()), data.queryCount, data.docCount, data.vocabSize,
        data.postingCount, data.queryStorageCount, data.shards, data.k, data.debug,
        task, data.blockNum);
    AscendC::PipeBarrier<PIPE_ALL>();
    FindSimdThreshold(local, data.k);
    AscendC::Simt::VF_CALL<FusedCollectSimt>(AscendC::Simt::Dim3{kThreads,1,1},
        reinterpret_cast<__ubuf__ const uint32_t *>(local.GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint32_t *>(local.GetPhyAddr())+kSelection,
        reinterpret_cast<__gm__ uint32_t *>(partial_scores), reinterpret_cast<__gm__ int32_t *>(partial_ids),
        task, data.docCount/data.shards, data.k);
    AscendC::PipeBarrier<PIPE_ALL>();
    }
}
