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

__simt_callee__ __aicore__ inline void PackForSimd(
    __ubuf__ const uint32_t *scores, __ubuf__ uint32_t *scratch, uint32_t width, uint32_t k)
{
    const uint32_t tid = AscendC::Simt::GetThreadIdx<0>();
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
    }
    scratch[kOwnerBegin+tid] = begin;
    scratch[kOwnerCount+tid] = positive;
    if (bad == 0 && total > k && total <= kCapacity) {
        uint32_t written = 0;
        // Each producer owns a contiguous compact segment with ascending REAL IDs.
        // Retain the accepted membership-mask compaction: reread only known hits,
        // rather than performing a second complete score scan.
        for (uint32_t pos = 0; written < positive && pos < 64; ++pos) {
            const uint32_t membership = pos < 32 ? lowMask : highMask;
            const uint32_t bit = pos < 32 ? pos : pos-32;
            if ((membership & (1U << bit)) != 0) {
                const uint32_t doc = tid+pos*kThreads;
                scratch[kPositiveBits+begin+written] = scores[doc];
                scratch[kPositiveIds+begin+written] = doc;
                ++written;
            }
        }
    }
    asc_syncthreads();
}

// Hardware ACCUMULATE returns a CDF, not individual bin frequencies. The two
// uint16 register halves are widened to 256 uint32 counts (at most 1024 inputs).
// Four bounded byte passes; masks filter candidates but never truncate FP32 bits.
__aicore__ inline void HistogramCdf(__ubuf__ uint32_t *input, __ubuf__ uint32_t *cdf,
    uint32_t count, uint32_t prefix, uint32_t mask, int16_t shift)
{
    __VEC_SCOPE__ {
        V::RegTensor<uint32_t> bits, keys, prefixBits, maskVec, zero32;
        V::RegTensor<uint16_t> histLo, histHi, zero16;
        auto all32 = V::CreateMask<uint32_t>();
        auto all16 = V::CreateMask<uint16_t>();
        V::Duplicate(histLo, 0, all16); V::Duplicate(histHi, 0, all16);
        V::Duplicate(zero16, 0, all16); V::Duplicate(zero32, 0, all32);
        V::Duplicate(maskVec, mask, all32);
        uint32_t remaining = count;
        const uint16_t tiles = static_cast<uint16_t>((count+63)/64);
        for (uint16_t tile = 0; tile < tiles; ++tile) {
            auto valid = V::UpdateMask<uint32_t>(remaining);
            V::LoadAlign<uint32_t, V::PostLiteral::POST_MODE_UPDATE>(bits, input, 64);
            V::Not(keys, bits, valid); // input contains only validated positive FP32
            V::And(prefixBits, keys, maskVec, valid);
            V::MaskReg compared, selected, mask16, mask8;
            V::Compares<uint32_t, AscendC::CMPMODE::EQ>(compared, prefixBits, prefix, valid);
            V::And(selected, compared, valid, all32); // explicitly suppress tail/old mask bits
            V::RegTensor<uint32_t> shifted;
            V::ShiftRights<uint32_t, int16_t>(shifted, keys, shift, selected);
            V::RegTensor<uint16_t> low16, ignored16;
            V::RegTensor<uint8_t> low8, ignored8;
            V::DeInterleave(low16, ignored16, (V::RegTensor<uint16_t>&)shifted,
                (V::RegTensor<uint16_t>&)zero32);
            V::DeInterleave(low8, ignored8, (V::RegTensor<uint8_t>&)low16,
                (V::RegTensor<uint8_t>&)zero16);
            V::Pack(mask16, selected); V::Pack(mask8, mask16);
            V::Histograms<uint8_t, uint16_t, V::HistogramsBinType::BIN0,
                V::HistogramsType::ACCUMULATE>(histLo, low8, mask8);
            V::Histograms<uint8_t, uint16_t, V::HistogramsBinType::BIN1,
                V::HistogramsType::ACCUMULATE>(histHi, low8, mask8);
        }
        V::RegTensor<uint32_t> c0, c1, c2, c3;
        V::Interleave((V::RegTensor<uint16_t>&)c0, (V::RegTensor<uint16_t>&)c1, histLo, zero16);
        V::Interleave((V::RegTensor<uint16_t>&)c2, (V::RegTensor<uint16_t>&)c3, histHi, zero16);
        V::StoreAlign<uint32_t, V::PostLiteral::POST_MODE_UPDATE>(cdf, c0, 64, all32);
        V::StoreAlign<uint32_t, V::PostLiteral::POST_MODE_UPDATE>(cdf, c1, 64, all32);
        V::StoreAlign<uint32_t, V::PostLiteral::POST_MODE_UPDATE>(cdf, c2, 64, all32);
        V::StoreAlign<uint32_t, V::PostLiteral::POST_MODE_UPDATE>(cdf, c3, 64, all32);
    }
}

__aicore__ inline void VectorToScalar()
{
    const auto e = static_cast<event_t>(::GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
    AscendC::SetFlag<AscendC::HardEvent::V_S>(e);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(e);
}

__aicore__ inline void ScalarToVector()
{
    const auto e = static_cast<event_t>(::GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V));
    AscendC::SetFlag<AscendC::HardEvent::S_V>(e);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(e);
}

__aicore__ inline void FindSimdThreshold(AscendC::LocalTensor<uint32_t> local, uint32_t k)
{
    const uint32_t base = kSelection;
    VectorToScalar();
    if (local.GetValue(base+kControl+1) != 1) return;
    const uint32_t count = local.GetValue(base+kControl);
    uint32_t rank = k, prefix = 0, mask = 0;
    auto *memory = reinterpret_cast<__ubuf__ uint32_t *>(local.GetPhyAddr());
    for (int16_t shift = 24; shift >= 0; shift -= 8) {
        HistogramCdf(memory+base+kPositiveBits, memory+base, count, prefix, mask, shift);
        VectorToScalar();
        if (rank == 0 || local.GetValue(base+255) < rank) {
            local.SetValue(base+kControl+1, 2U); ScalarToVector(); return;
        }
        // The same bounded binary-boundary principle used by official TopKV2:
        // eight scalar CDF comparisons rather than scanning/sorting all candidates.
        uint32_t lo = 0, hi = 255;
        while (lo < hi) {
            const uint32_t mid = (lo+hi)/2;
            if (local.GetValue(base+mid) >= rank) hi = mid;
            else lo = mid+1;
        }
        const uint32_t before = lo == 0 ? 0 : local.GetValue(base+lo-1);
        rank -= before; prefix |= lo << shift; mask |= 255U << shift;
        // Complete scalar reads before the next vector CDF overwrites the region.
        ScalarToVector();
    }
    local.SetValue(base+kControl+2, prefix);
    local.SetValue(base+kControl+3, rank);
    ScalarToVector();
}
