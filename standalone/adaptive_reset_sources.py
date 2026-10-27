"""Reset only previous exact positive IDs; vector-clear first/fallback tasks on950."""
from standalone import sparse_pack_sources as base

simd=base.base.simd
VENDOR='bm25_fused_score_topk_adaptive_reset_large'
KERNEL=base.KERNEL
SCORE_MARKER='__simt_vf__ __aicore__ LAUNCH_BOUND(kThreads) inline void FusedScorePackSimt('
META_OLD='''        scratch[kControl+1] = bad != 0 ? 2U : (total > k && total <= kCapacity ? 1U : 0U);
    }'''
META_NEW='''        scratch[kControl+1] = bad != 0 ? 2U : (total > k && total <= kCapacity ? 1U : 0U);
        // Retain a complete positive-ID count only for the exact compact path.
        // Control+4 is outside histogram/owner metadata and survives Collect.
        scratch[kControl+4] = bad == 0 && total > k && total <= kCapacity ? total : kNoId;
    }'''
ZERO_OLD=base.base.ZERO_NEW
ZERO_NEW='''        // Full clears were completed by SIMD before this VF. Otherwise every
        // previously nonzero score is listed exactly once in the old compact IDs.
        uint32_t resetInvalid = 0;
        if (resetCount <= kCapacity) {
            for (uint32_t pos = thread; pos < resetCount; pos += kThreads) {
                const uint32_t previousDoc = bitmap[kPositiveIds+pos];
                if (previousDoc < width) scores[previousDoc] = 0.0F;
                else resetInvalid = 1; // never write outside UB
            }
        }
        bitmap[kOwnerBegin+thread] = 0;
        bitmap[kOwnerCount+thread] = 0;
        asc_syncthreads();
        // Thread0 initialized control before this barrier; do not race its reset.
        // The existing query-metadata barrier publishes this error before use.
        if (resetInvalid != 0) atomicAdd(&control[0], 1U);'''
VECTOR_CLEAR='''// IndexSDK-style SIMD initialization before SIMT; exact positive-zero bits.
__aicore__ inline void ClearScoreVector(AscendC::LocalTensor<uint32_t> local, uint32_t width)
{
    auto *cursor = reinterpret_cast<__ubuf__ uint32_t *>(local.GetPhyAddr());
    __VEC_SCOPE__ {
        V::RegTensor<uint32_t> zero;
        auto all = V::CreateMask<uint32_t>();
        V::Duplicate(zero, 0U, all);
        uint32_t remaining = width;
        const uint16_t tiles = static_cast<uint16_t>((width+63)/64);
        for (uint16_t tile = 0; tile < tiles; ++tile) {
            auto valid = V::UpdateMask<uint32_t>(remaining);
            V::StoreAlign<uint32_t>(cursor, zero, valid);
            cursor += 64;
        }
    }
}

'''
CALL_OLD='''    AscendC::Simt::VF_CALL<FusedScorePackSimt>'''
CALL_NEW='''    uint32_t resetCount = kNoId;
    if (data.debug != 2) {
        // task=blockIdx initially, then advances by blockNum inside this launch.
        // Never read scratch from a previous launch or another AIV block.
        if (task >= data.blockNum) {
            VectorToScalar();
            const uint32_t oldMode = local.GetValue(kSelection+kControl+1);
            const uint32_t oldCount = local.GetValue(kSelection+kControl+4);
            if (oldMode == 1 && oldCount > data.k && oldCount <= kCapacity) resetCount = oldCount;
        }
        if (resetCount == kNoId) {
            ClearScoreVector(local, data.docCount/data.shards);
            AscendC::PipeBarrier<PIPE_ALL>(); // vector stores complete before SIMT reads
        }
    }
    AscendC::Simt::VF_CALL<FusedScorePackSimt>'''
SIGNATURE_OLD='''    uint32_t debug, uint32_t blockId, uint32_t blockNum)'''
SIGNATURE_NEW='''    uint32_t debug, uint32_t blockId, uint32_t blockNum, uint32_t resetCount)'''
ARGS_OLD='''        task, data.blockNum);'''
ARGS_NEW='''        task, data.blockNum, resetCount);'''
CHANGES=((META_OLD,META_NEW),(ZERO_OLD,ZERO_NEW),(SCORE_MARKER,VECTOR_CLEAR+SCORE_MARKER),
    (SIGNATURE_OLD,SIGNATURE_NEW),(CALL_OLD,CALL_NEW),(ARGS_OLD,ARGS_NEW))


def candidate_sources(original):
    sources=base.candidate_sources(original)
    text=sources[KERNEL].decode('utf-8')
    for old,new in CHANGES:text=simd.t.replace_once(text,old,new)
    sources[KERNEL]=text.encode('utf-8')
    return sources


def prepare(original,target):
    metadata=base.prepare(original,target);sources=candidate_sources(original)
    for name,raw in sources.items():(target/name).write_bytes(raw)
    preset=target/'CMakePresets.json';data=simd.json.loads(preset.read_text(encoding='utf-8'));changed=0
    for item in data.get('configurePresets',[]):
        variables=item.get('cacheVariables',{});value=variables.get('vendor_name')
        if isinstance(value,dict) and value.get('value')==base.VENDOR:value['value']=VENDOR;changed+=1
        elif value==base.VENDOR:variables['vendor_name']=VENDOR;changed+=1
    simd.r.require(changed>0,'Adaptive reset vendor missing');simd.r.write_json(preset,data)
    metadata.update(status='adaptive_reset_source_prepared_not_device_verified',vendor_name=VENDOR,
        source_sha256={name:simd.hashlib.sha256(raw).hexdigest() for name,raw in sources.items()},
        reset_policy='first task/full fallback SIMD zero; subsequent exact compact path clears complete prior positive IDs in original score VF; preserve zero bits and per-term FP32 order',
        scope='Derived from accepted Sparse Pack, not warp candidate. Only score-buffer initialization changes; same public ABI/UB/threads/threshold/collector/tree/CPU/Search. Internal VF gets resetCount. Actual950 compilation, guards and paired timing required.')
    simd.r.write_json(target/'bm25_project.json',metadata)
    return metadata
