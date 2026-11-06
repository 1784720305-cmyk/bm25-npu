"""One bounded follow-up: keep the accepted compact collector as a separate VF."""

from tools.capacity_sources import once, sources
from tools.dense_sources import dense_sources

COLLECTOR = '__simt_vf__ __aicore__ LAUNCH_BOUND(kThreads) inline void FusedCollectSimt('
END = '\n}\n'


def collector(text):
    begin = text.index(COLLECTOR)
    end = text.index(END, begin) + len(END)
    return text[begin:end]


def split_sources(profile=False):
    host, kernel, plan = dense_sources(profile)
    _, accepted, _ = sources(1024, profile)
    old = collector(kernel)
    dense = once(old, 'void FusedCollectSimt(', 'void FusedDenseCollectSimt(')
    dense = once(
        dense,
        '''    if (mode == 0) {
        SelectLocal(scores, topScores, topIds, scratch, query, docCount, k);
        return;
    }
''',
        '',
    )
    dense = once(dense, 'if (mode != 1 && mode != 3)', 'if (mode != 3)')
    dense = once(dense, 'const bool compact = mode == 1;', 'const bool compact = false;')
    dense = once(dense, '    const bool direct = compact && scratch[kControl + 4] != 0;\n', '')
    dense = once(
        dense,
        'const uint64_t sourceBegin = !compact || direct ? thread : scratch[kOwnerBegin + thread];',
        'const uint64_t sourceBegin = thread;',
    )
    dense = once(
        dense,
        'const uint64_t sourceEnd = !compact ? docCount : (direct ? scratch[kControl] : sourceBegin + scratch[kOwnerCount + thread]);',
        'const uint64_t sourceEnd = docCount;',
    )
    dense = once(
        dense, 'const uint64_t sourceStep = !compact || direct ? kThreads : 1;', 'const uint64_t sourceStep = kThreads;'
    )
    kernel = once(kernel, old, collector(accepted) + '\n' + dense)
    start = kernel.index('__aicore__ inline void FindDenseOrCompactThreshold(')
    end = kernel.index('__simt_vf__', start)
    threshold = kernel[start:end]
    changed = once(threshold, 'inline void FindDenseOrCompactThreshold(', 'inline bool FindDenseOrCompactThreshold(')
    changed = once(changed, '        return;', '        return false;')
    # Return the already-known route without another UB read or an additional barrier.
    changed = once(changed, '    ScalarToVector();\n}', '    ScalarToVector();\n    return dense;\n}')
    kernel = once(kernel, threshold, changed)
    kernel = once(
        kernel,
        '        FindDenseOrCompactThreshold(local, data.k, data.docCount / data.shards);',
        '        const bool denseRoute = FindDenseOrCompactThreshold(local, data.k, data.docCount / data.shards);',
    )
    start = kernel.index('        AscendC::Simt::VF_CALL<FusedCollectSimt>(')
    end = kernel.index('\n        AscendC::PipeBarrier<PIPE_ALL>();', start)
    call = kernel[start:end]
    dense_call = once(call, 'VF_CALL<FusedCollectSimt>', 'VF_CALL<FusedDenseCollectSimt>')
    indent = lambda block: '\n'.join('    ' + line for line in block.splitlines())
    kernel = once(
        kernel,
        call,
        '        if (denseRoute) {\n' + indent(dense_call) + '\n        } else {\n' + indent(call) + '\n        }',
    )
    return host, kernel, plan
