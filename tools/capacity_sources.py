"""Generate isolated capacity experiments from the locked 1024 implementation."""

from pathlib import Path

from standalone import final_support as s
from standalone import resident_index as r

ROOT = Path(__file__).resolve().parents[1]
FUSED = ROOT / 'feature_retrieval/src/ascendfaiss/ops/ascendc/bm25/fused'
CAPACITIES = (1024, 2048, 4096)


def layout(capacity):
    r.require(capacity in CAPACITIES, 'Capacity must be 1024, 2048 or 4096')
    scratch = 8544 + capacity * 8
    buffer = 34112 + scratch
    dynamic = ((buffer + 32767) // 32768) * 32768
    return {'capacity': capacity, 'scratch_bytes': scratch, 'buffer_bytes': buffer, 'dynamic_ub_bytes': dynamic}


def once(text, old, new):
    r.require(text.count(old) == 1, 'Locked source transform no longer matches: ' + old[:80])
    return text.replace(old, new, 1)


def sources(capacity, profile=False):
    plan = layout(capacity)
    kernel = (FUSED / 'op_kernel/bm25_fused_score_top_k.cpp').read_text(encoding='utf-8')
    host = (FUSED / 'op_host/bm25_fused_score_top_k.cpp').read_text(encoding='utf-8')
    if capacity != 1024:
        kernel = once(kernel, 'constexpr uint32_t kCapacity = 1024;', f'constexpr uint32_t kCapacity = {capacity};')
        kernel = once(kernel, 'kSelectScratchBytes == 16736', f'kSelectScratchBytes == {plan["scratch_bytes"]}')
        kernel = once(kernel, 'kBufferBytes == 50848', f'kBufferBytes == {plan["buffer_bytes"]}')
        kernel = kernel.replace('// 16736', f'// {plan["scratch_bytes"]}')
        kernel = kernel.replace('positive raw bits/IDs[1024]', f'positive raw bits/IDs[{capacity}]')
        kernel = kernel.replace('histogram counts <=1024', f'histogram counts <={capacity}')
        kernel = kernel.replace('// 50,848 bytes', f'// {plan["buffer_bytes"]:,} bytes')
        kernel = kernel.replace("Host's 64 KiB", f"Host's {plan['dynamic_ub_bytes'] // 1024} KiB")
    host = once(
        host, 'constexpr uint32_t kBufferBytes = 50848;', f'constexpr uint32_t kBufferBytes = {plan["buffer_bytes"]};'
    )
    host = once(
        host,
        'constexpr uint32_t kDynamicUbBytes = 64 * 1024;',
        f'constexpr uint32_t kDynamicUbBytes = {plan["dynamic_ub_bytes"]};',
    )
    host = once(
        host,
        '#include "ops/ascendc/bm25/fused/op_host/bm25_fused_score_top_k_tiling.h"',
        '#include "bm25_fused_score_top_k_tiling.h"',
    )
    if profile:
        # Only the diagnostic package accepts mode 3. Production mode 0 remains uninstrumented
        # in the separate normal package; no timed full Search uses these timestamps.
        host = once(
            host,
            '(*debug != 0 && *debug != 1 && *debug != 2)',
            '(*debug != 0 && *debug != 1 && *debug != 2 && *debug != 3)',
        )
        host = once(
            host, '    dims = {', '    if (*debug == 3 && docs / (*shards) < 8)\n        return false;\n    dims = {'
        )
        kernel = once(kernel, '        if (debug != 0) {', '        if (debug == 1) {')
        kernel = once(
            kernel,
            '            PackForSimd<true>(memory, memory + kSelection, width, k);',
            '            if (debug != 3)\n                PackForSimd<true>(memory, memory + kSelection, width, k);',
        )
        wrapper = '''
__simt_vf__ __aicore__ LAUNCH_BOUND(kThreads) inline void CapacityProfilePack(
    __ubuf__ uint32_t *memory, uint32_t width, uint32_t k)
{
    if (memory[kScoreControl] == 0)
        PackForSimd<true>(memory, memory + kSelection, width, k);
}
'''
        kernel = once(kernel, '} // namespace\n\nextern "C"', wrapper + '\n} // namespace\n\nextern "C"')
        kernel = once(
            kernel,
            '        AscendC::Simt::VF_CALL<FusedScorePackSimt>(',
            '        const int64_t profileStart = data.debug == 3 ? AscendC::GetSystemCycle() : 0;\n'
            '        AscendC::Simt::VF_CALL<FusedScorePackSimt>(',
        )
        kernel = once(
            kernel,
            '        FindSimdThreshold(local, data.k);',
            '        const int64_t profileScored = data.debug == 3 ? AscendC::GetSystemCycle() : 0;\n'
            '        if (data.debug == 3) {\n'
            '            AscendC::Simt::VF_CALL<CapacityProfilePack>(AscendC::Simt::Dim3{kThreads, 1, 1},\n'
            '                reinterpret_cast<__ubuf__ uint32_t *>(local.GetPhyAddr()), data.docCount / data.shards, data.k);\n'
            '            AscendC::PipeBarrier<PIPE_ALL>();\n'
            '        }\n'
            '        FindSimdThreshold(local, data.k);',
        )
        anchor = '        AscendC::PipeBarrier<PIPE_ALL>();\n    }\n}'
        kernel = once(
            kernel,
            anchor,
            '''        AscendC::PipeBarrier<PIPE_ALL>();
        if (data.debug == 3) {
            const int64_t profileSelected = AscendC::GetSystemCycle();
            AscendC::GlobalTensor<int64_t> trace;
            trace.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(debug_scores));
            trace.SetValue(task * 4, profileStart);
            trace.SetValue(task * 4 + 1, profileScored);
            trace.SetValue(task * 4 + 2, profileSelected);
            trace.SetValue(task * 4 + 3, kCapacity);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
}''',
        )
    return host, kernel, plan


def write_variant(target, capacity, profile):
    host, kernel, plan = sources(capacity, profile)
    for folder, text in (('op_host', host), ('op_kernel', kernel)):
        (target / folder / 'bm25_fused_score_top_k.cpp').write_bytes(text.encode('utf-8'))
    files = {
        p.relative_to(target).as_posix(): s.digest(p)
        for p in target.rglob('*')
        if p.is_file() and p.suffix in ('.h', '.cpp')
    }
    return {**plan, 'profile': profile, 'source_sha256': files}
