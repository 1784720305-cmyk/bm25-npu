"""零分汇总的整数/线程模型核对；不编译CANN、不执行NPU、不测性能。"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone import check_radix_model as original


def histogram(keys, mask, prefix, shift, bins):
    """Model private zero counts and BOTH uses of the logical histogram in the kernel."""
    reference, shared = [0] * bins, [0] * bins
    zeros = [0] * original.THREADS
    for thread in range(original.THREADS):
        for doc in range(thread, len(keys), original.THREADS):
            value = keys[doc]
            if value & mask != prefix:
                continue
            bucket = (value >> shift) & (bins - 1)
            reference[bucket] += 1
            if value == 0xffffffff:
                zeros[thread] += 1
            else:
                shared[bucket] += 1
    # This is the second (previously unused) channel of the EXISTING validation scan.
    _, _, _, zero_total = original.prefix_pair([0] * original.THREADS, zeros)
    logical = [n + (zero_total if bucket == bins - 1 else 0)
               for bucket, n in enumerate(shared)]
    assert logical == reference, '每个桶的整数计数必须与原逐项计数完全相同'
    assert sum(zeros) == zero_total
    return logical, zero_total


def select(raw, k):
    if type(k) is not int or not 1 <= k <= min(32, len(raw)):
        raise ValueError('K或文章数不合法')
    keys = [original.key(value) for value in raw]
    prefix, mask, remaining = 0, 0, k
    removed = 0
    for shift, bins in ((21, 2048), (10, 2048), (0, 1024)):
        logical, zeros = histogram(keys, mask, prefix, shift, bins)
        removed += zeros
        chunk = bins // original.THREADS
        counts = [sum(logical[t*chunk:(t+1)*chunk]) for t in range(original.THREADS)]
        ranks, _, total, _ = original.prefix_pair(counts, [0] * original.THREADS)
        assert 0 < remaining <= total
        owners = []
        for thread, before in enumerate(ranks):
            for bucket in range(thread*chunk, (thread+1)*chunk):
                count = logical[bucket]
                if before < remaining <= before + count:
                    owners.append((bucket, remaining-before))
                before += count
        assert len(owners) == 1
        bucket, remaining = owners[0]
        prefix |= bucket << shift
        mask |= (bins-1) << shift

    # Original ordered 128-ID tiles and 32-slot sorting schedule, intentionally unchanged.
    strict_count = k - remaining
    selected = [None] * k
    strict_seen = equal_seen = 0
    for tile in range(0, len(raw), original.THREADS):
        strict = [int(tile+t < len(raw) and keys[tile+t] < prefix) for t in range(original.THREADS)]
        equal = [int(tile+t < len(raw) and keys[tile+t] == prefix) for t in range(original.THREADS)]
        rs, re, ns, ne = original.prefix_pair(strict, equal)
        for thread in range(original.THREADS):
            slot = k
            if strict[thread]:
                slot = strict_seen + rs[thread]
            if equal[thread] and equal_seen + re[thread] < remaining:
                slot = strict_count + equal_seen + re[thread]
            if slot < k:
                assert selected[slot] is None
                selected[slot] = (keys[tile+thread], tile+thread)
        strict_seen += ns
        equal_seen += ne
    assert strict_seen == strict_count and equal_seen >= remaining
    assert all(value is not None for value in selected)
    ids = [value[1] for value in original.sort_candidates(selected)]
    return ids, [raw[i] for i in ids], removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', default=str(ROOT / 'results/radix_zero_model_check.json'))
    args = parser.parse_args()
    rng = random.Random(95020261017)
    count = removed = 0
    for n in (1, 2, 6, 31, 32, 33, 127, 128, 129, 257, 513, 1025, 8192, 65536):
        for k in sorted({1, min(10, n), min(32, n)}):
            patterns = [
                [0] * n,
                [0x80000000 if i % 2 else 0 for i in range(n)],
                [original.float_bits(1.)] * n,
                [rng.randrange(1, 0x7f800000) for _ in range(n)],  # no zeros
                [rng.randrange(0x7f800000) if i % 11 == 0 else
                 (0x80000000 if i % 2 else 0) for i in range(n)],
                [1 if i % 3 == 0 else 0 for i in range(n)],  # positive subnormals
                [0x7f7fffff if i % 7 == 0 else 0 for i in range(n)],
                [original.float_bits(float(i % 9)) for i in range(n)],
                [0x3f800001 if i % 2 else 0x3f800000 for i in range(n)],
            ]
            for positives in (k-1, k, min(k+1, n)):
                row = [0x80000000 if i % 2 else 0 for i in range(n)]
                for i in range(positives):
                    row[n-1 if i == 0 else i-1] = i+1
                patterns.append(row)
            for raw in patterns:
                expected = sorted(range(n), key=lambda i: (-original.as_float(raw[i]), i))[:k]
                actual, bits, skipped = select(raw, k)
                assert actual == expected and bits == [raw[i] for i in expected], (n, k)
                # Original, independently retained radix thread-schedule implementation.
                assert (actual, bits) == original.select(raw, k)
                count += 1
                removed += skipped
    invalid = 0
    for raw in (0x7f800000, 0xff800000, 0x7fc00001, 0xbf800000, 0x80000001):
        for position in (0, 128, 256):
            row = [0] * 257
            row[position] = raw
            try:
                select(row, 10)
            except ValueError:
                invalid += 1
            else:
                raise AssertionError('非法输入必须拒绝')
    for k in (0, 3, 33, True):
        try:
            select([0, 0], k)
        except ValueError:
            invalid += 1
        else:
            raise AssertionError('非法K必须拒绝')
    paths = ['standalone/check_radix_zero_model.py', 'standalone/check_radix_model.py',
             'npu_prototype/topk_radix_zero/op_kernel/bm25_top_k.cpp']
    record = {'status': 'python_zero_histogram_schedule_passed_not_device_validation',
              'cases_passed': count, 'invalid_cases_passed': invalid,
              'all_pass_histogram_bins_exact': True, 'original_radix_and_full_sort_exact': True,
              'strict_ids_and_original_score_bits': True,
              'modeled_zero_atomic_updates_removed': removed,
              'compiled': False, 'npu_verified': False, 'performance_verified': False,
              'source_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}}
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((json.dumps(record, indent=2) + '\n').encode('utf-8'))
    print(f'零分汇总模型通过：{count}组，每轮全部桶计数、新旧选择、完整排序及原FP32字节一致；另{invalid}组非法输入通过。')
    print('仅Python整数/线程模型，不是CANN编译、NPU正确性或性能验证。报告：', target)


if __name__ == '__main__':
    main()
