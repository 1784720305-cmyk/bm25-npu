"""候选收集修订的整数/线程模型与读取计数；不生成、编译或运行NPU内核，不测性能。"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone import check_radix_model as original


def minimum_head(heads):
    """Four 32-lane XOR min reductions, then four UB heads; no FP32 comparison."""
    assert len(heads) == original.THREADS == 128
    winners = list(heads)
    for mask in (16, 8, 4, 2, 1):
        before = winners[:]
        winners = [min(before[t], before[(t//32)*32 + ((t % 32) ^ mask)])
                   for t in range(original.THREADS)]
    warp_heads = [winners[w*32] for w in range(4)]
    result = min(warp_heads)
    assert result == min(heads)
    return result


def collect(raw, k, threshold, remaining, thread_order):
    """Unchanged exact Kth threshold is an input; model ONLY candidate collection."""
    if type(k) is not int or not 1 <= k <= min(32, len(raw)):
        raise ValueError('K或文章数非法')
    keys = [original.key(value) for value in raw]
    assert sorted(thread_order) == list(range(original.THREADS))
    strict_count = k - remaining
    assert 1 <= remaining <= k
    assert sum(value < threshold for value in keys) == strict_count
    assert sum(value == threshold for value in keys) >= remaining
    counts = [0] * original.THREADS
    heads = [original.INVALID] * original.THREADS
    first_reads = 0
    for thread in thread_order:
        for doc in range(thread, len(raw), original.THREADS):
            value = keys[doc]
            first_reads += 1
            counts[thread] += int(value < threshold)
            if value == threshold and heads[thread] == original.INVALID:
                heads[thread] = doc

    # ONE whole-row count scan over the 128 private counters, not one per 128 documents.
    starts, _, total, _ = original.prefix_pair(counts, [0] * original.THREADS)
    assert total == strict_count and total < k
    selected = [None] * k
    strict_reads = 0
    for thread in thread_order:
        if counts[thread] == 0:
            continue
        written = 0
        for doc in range(thread, len(raw), original.THREADS):
            strict_reads += 1
            if keys[doc] < threshold:
                slot = starts[thread] + written
                assert selected[slot] is None and slot < strict_count
                selected[slot] = (keys[doc], doc)
                written += 1
                if written == counts[thread]:
                    break  # No further reads after this thread's last strict candidate.
        assert written == counts[thread]

    lazy_reads = 0
    for rank in range(remaining):
        winner = minimum_head(heads)
        assert winner != original.INVALID and winner < len(raw)
        owner = winner % original.THREADS
        assert heads[owner] == winner
        slot = strict_count + rank
        assert selected[slot] is None
        selected[slot] = (threshold, winner)
        if rank+1 < remaining:
            # Only the winning stream advances; its cursor never moves backwards.
            heads[owner] = original.INVALID
            for doc in range(winner+original.THREADS, len(raw), original.THREADS):
                lazy_reads += 1
                if keys[doc] == threshold:
                    heads[owner] = doc
                    break
    assert all(value is not None for value in selected)
    assert len({value[1] for value in selected}) == k
    ids = [value[1] for value in original.sort_candidates(selected)]
    width = (len(raw) + original.THREADS-1)//original.THREADS
    # At most strict_count owner streams rescan, and remaining-1 winner streams advance.
    # Each stream cursor is monotone. This is a READ-COUNT upper bound, NOT elapsed time.
    assert first_reads == len(raw)
    assert strict_reads + lazy_reads <= (k-1)*width
    return ids, [raw[i] for i in ids], {
        'original_tile_prefix_calls': width, 'proposed_whole_row_prefix_calls': 1,
        'proposed_head_merge_rounds': remaining, 'first_scan_score_reads': first_reads,
        'strict_scatter_score_reads': strict_reads, 'tie_cursor_score_reads': lazy_reads,
        'extra_score_read_upper_bound': (k-1)*width,
        'scope': 'Integer/thread schedule model and logical counts only; not instructions, device time or speedup.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', default=str(ROOT/'results/radix_collect_model_check.json'))
    args = parser.parse_args()
    rng = random.Random(95020261019)
    count = schedules = 0
    examples = []
    for n in (1, 2, 6, 31, 32, 33, 127, 128, 129, 257, 513, 1025, 8192, 65536):
        for k in sorted({1, min(10, n), min(32, n)}):
            patterns = [
                [0]*n,
                [0x80000000 if i % 2 else 0 for i in range(n)],
                [original.float_bits(1.)]*n,
                [original.float_bits(float(i)) for i in range(n)],
                [original.float_bits(float(n-i)) for i in range(n)],
                [original.float_bits(float(rng.randrange(9))) for _ in range(n)],
                [rng.randrange(1, 0x7f800000) for _ in range(n)],
                [original.float_bits(float(rng.randrange(9))) if i % 11 == 0 else 0 for i in range(n)],
                [1 if i % 3 == 0 else 0 for i in range(n)],
                [0x7f7fffff if i % 7 == 0 else 0 for i in range(n)],
                [0x3f800001 if i % 2 else 0x3f800000 for i in range(n)],
            ]
            for positives in (k-1, k, min(n, k+1)):
                row = [0x80000000 if i % 2 else 0 for i in range(n)]
                for i in range(positives):
                    row[n-1 if i == 0 else i-1] = i+1
                patterns.append(row)
            # Concentrate cutoff ties in a single strided stream, including late heads.
            for tail in (False, True):
                row = [0]*n
                equal_ids = list(range(0 if not tail else n//2, n, original.THREADS))
                for doc in equal_ids:
                    row[doc] = original.float_bits(1.)
                patterns.append(row)
            # Exactly K cutoff ties in different streams; advancing most winning
            # streams must search to their end before declaring the next head absent.
            row = [0]*n
            for doc in range(k):
                row[doc] = original.float_bits(1.)
            patterns.append(row)
            for pattern_index, raw in enumerate(patterns):
                old_ids, old_bits = original.select(raw, k)
                expected = sorted(range(n), key=lambda i: (-original.as_float(raw[i]), i))[:k]
                assert old_ids == expected and old_bits == [raw[i] for i in expected]
                # Keep the original threshold algorithm: it is NOT replaced by this proposal.
                threshold = original.key(old_bits[-1])
                remaining = k-sum(original.key(value) < threshold for value in raw)
                forward = list(range(original.THREADS))
                shuffled = forward[:]
                rng.shuffle(shuffled)
                for order in (forward, list(reversed(forward)), shuffled):
                    ids, bits, metrics = collect(raw, k, threshold, remaining, order)
                    assert ids == expected and bits == old_bits, (n, k, pattern_index)
                    schedules += 1
                if n == 8192 and k == 10 and pattern_index in (0, 3, 7, 14, 15):
                    examples.append({'docs': n, 'k': k, 'pattern_index': pattern_index, **metrics})
                count += 1
    invalid = 0
    for raw in (0x7f800000, 0xff800000, 0x7fc00001, 0xbf800000, 0x80000001):
        try:
            collect([0, raw], 1, 0xffffffff, 1, list(range(original.THREADS)))
        except ValueError:
            invalid += 1
        else:
            raise AssertionError('非法分数必须拒绝')
    for k in (0, 3, 33, True):
        try:
            collect([0, 0], k, 0xffffffff, 1, list(range(original.THREADS)))
        except ValueError:
            invalid += 1
        else:
            raise AssertionError('非法K必须拒绝')
    paths = ('standalone/check_radix_collect_model.py', 'standalone/check_radix_model.py',
             'npu_prototype/topk_radix/op_kernel/bm25_top_k.cpp',
             'npu_prototype/topk_merge/op_kernel/bm25_top_k_merge.cpp')
    record = {'status': 'candidate_collection_model_passed_not_device_or_performance_evidence',
              'cases_passed': count, 'thread_schedules_checked': schedules, 'invalid_cases_passed': invalid,
              'strict_ids_and_original_score_bits': True, 'extra_read_bound_checked': True,
              'examples': examples, 'new_kernel_written': False, 'compiled': False,
              'npu_verified': False, 'performance_verified': False,
              'source_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}}
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((json.dumps(record, indent=2)+'\n').encode('utf-8'))
    print(f'候选收集模型通过：{count}组、{schedules}种线程执行顺序，完整编号/原FP32字节一致；另{invalid}组非法输入通过。')
    print('只核对整数/线程模型与逻辑读取上界，没有新NPU内核、设备执行或性能结论。报告：', target)


if __name__ == '__main__':
    main()
