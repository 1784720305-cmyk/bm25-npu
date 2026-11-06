"""Generate a bounded dense-row SIMD threshold candidate; retain locked production sources."""

from tools.capacity_sources import once, sources


def dense_sources(profile=False):
    host, kernel, plan = sources(1024, profile)
    start = kernel.index('__aicore__ inline void RegisterThreshold(')
    end = kernel.index('__aicore__ inline void VectorToScalar()', start)
    original = kernel[start:end]
    dense = once(original, 'void RegisterThreshold(', 'void RegisterDenseThreshold(')
    dense = once(
        dense,
        'V::RegTensor<uint32_t> prefix, maskVec, zero32;',
        'V::RegTensor<uint32_t> prefix, maskVec, zero32, magnitudeMask;',
    )
    dense = once(
        dense,
        'V::Duplicate(zero32, 0U, all32);',
        'V::Duplicate(zero32, 0U, all32);\n        V::Duplicate(magnitudeMask, 0x7fffffffU, all32);',
    )
    dense = once(
        dense,
        'V::RegTensor<uint32_t> bits, keys, masked, shifted;',
        'V::RegTensor<uint32_t> bits, keys, masked, shifted, canonical;',
    )
    dense = once(
        dense,
        'V::Not(keys, bits, valid);',
        '// Inputs have already passed normal scoring/debug2 validation.\n'
        '                // Clearing the sign bit only canonicalizes negative zero for valid scores.\n'
        '                V::And(canonical, bits, magnitudeMask, valid);\n'
        '                V::Not(keys, canonical, valid);',
    )
    kernel = (
        kernel[:end]
        + '// Dense CDF counts are bounded by the existing 8192-document local score row.\n'
        + dense
        + kernel[end:]
    )
    start = kernel.index('__aicore__ inline void FindSimdThreshold(')
    end = kernel.index('__simt_vf__', start)
    old = kernel[start:end]
    new = '''__aicore__ inline void FindDenseOrCompactThreshold(AscendC::LocalTensor<uint32_t> local,
                                                   uint32_t k, uint32_t width)
{
    static_assert(kMaxDocs < (1U << 16), "Dense uint16 CDF count must not overflow");
    const uint32_t base = kSelection;
    VectorToScalar();
    const uint32_t mode = local.GetValue(base + kControl + 1);
    const uint32_t count = local.GetValue(base + kControl);
    // Mode 2 is invalid; few/zero candidates retain the accepted generic path.
    const bool dense = mode == 0 && count > kCapacity;
    if (mode != 1 && !dense)
        return;
    auto *memory = reinterpret_cast<__ubuf__ uint32_t *>(local.GetPhyAddr());
    if (dense)
        RegisterDenseThreshold(memory, memory + base, width, k);
    else
        RegisterThreshold(memory + base + kPositiveBits, memory + base, count, k);
    VectorToScalar();
    const uint32_t prefix = local.GetValue(base), rank = local.GetValue(base + 64), raw = ~prefix;
    if (rank == 0 || rank > k || raw == 0 || raw >= 0x7f800000U)
        local.SetValue(base + kControl + 1, 2U);
    else if (dense)
        local.SetValue(base + kControl + 1, 3U); // Internal dense collection mode, not the debug attribute.
    local.SetValue(base + kControl + 2, prefix);
    local.SetValue(base + kControl + 3, rank);
    ScalarToVector();
}

'''
    kernel = once(kernel, old, new)
    # Modify only the compact collector, leaving SelectLocal and its few/zero handling intact.
    start = kernel.index('__simt_vf__ __aicore__ LAUNCH_BOUND(kThreads) inline void FusedCollectSimt(')
    before, collector = kernel[:start], kernel[start:]
    collector = once(collector, 'if (mode != 1) {', 'if (mode != 1 && mode != 3) {')
    collector = once(collector, 'const bool compact = true;', 'const bool compact = mode == 1;')
    collector = once(
        collector,
        'const bool direct = scratch[kControl + 4] != 0;',
        'const bool direct = compact && scratch[kControl + 4] != 0;',
    )
    collector = once(
        collector,
        'const uint64_t sourceBegin = direct ? thread : scratch[kOwnerBegin + thread];',
        'const uint64_t sourceBegin = !compact || direct ? thread : scratch[kOwnerBegin + thread];',
    )
    collector = once(
        collector,
        'const uint64_t sourceEnd = direct ? scratch[kControl] : sourceBegin + scratch[kOwnerCount + thread];',
        'const uint64_t sourceEnd = !compact ? docCount : (direct ? scratch[kControl] : sourceBegin + scratch[kOwnerCount + thread]);',
    )
    collector = once(
        collector,
        'const uint64_t sourceStep = direct ? kThreads : 1;',
        'const uint64_t sourceStep = !compact || direct ? kThreads : 1;',
    )
    kernel = before + collector
    kernel = once(
        kernel,
        'FindSimdThreshold(local, data.k);',
        'FindDenseOrCompactThreshold(local, data.k, data.docCount / data.shards);',
    )
    return host, kernel, plan
