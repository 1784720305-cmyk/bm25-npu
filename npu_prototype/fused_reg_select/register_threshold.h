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
        for (int16_t shift=24;shift>=0;shift-=8) {
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
