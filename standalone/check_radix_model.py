"""检查radix算子的整数选择/线程分组模型；不编译、不运行NPU，不作性能基准。"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import struct

ROOT = Path(__file__).resolve().parents[1]
THREADS = 128
INVALID = 0x7fffffff


def float_bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def as_float(raw):
    return struct.unpack('<f', struct.pack('<I', raw))[0]


def key(raw):
    magnitude = raw & 0x7fffffff
    if magnitude >= 0x7f800000 or (raw & 0x80000000 and magnitude != 0):
        raise ValueError('只接受非负有限FP32；允许负零')
    return (~(0 if magnitude == 0 else raw)) & 0xffffffff


def prefix_pair(a, b):
    """Model all-lane shuffles with snapshots, then four UB warp totals."""
    assert len(a) == len(b) == THREADS
    partials = []
    totals = []
    for values in (a, b):
        scan = list(values)
        for offset in (1, 2, 4, 8, 16):
            previous = scan[:]
            for thread in range(THREADS):
                if thread % 32 >= offset:
                    scan[thread] += previous[thread-offset]
        warp_totals = [scan[w*32+31] for w in range(4)]
        partials.append([scan[t]-values[t]+sum(warp_totals[:t//32]) for t in range(THREADS)])
        totals.append(sum(warp_totals))
    return partials[0], partials[1], totals[0], totals[1]


def sort_candidates(chosen):
    """Model the actual 32-lane compare/exchange schedule, including invalid padding."""
    entries = list(chosen) + [(0xffffffff, INVALID)] * (32-len(chosen))
    for span in (2, 4, 8, 16, 32):
        stride = span // 2
        while stride:
            old = entries[:]
            for lane in range(32):
                other = old[lane ^ stride]
                want_better = ((lane & span) == 0) == ((lane & stride) == 0)
                if (want_better and other < old[lane]) or (not want_better and old[lane] < other):
                    entries[lane] = other
            stride //= 2
    return entries[:len(chosen)]


def select(raw, k):
    if not 1 <= k <= min(32, len(raw)):
        raise ValueError('K或文章数不合法')
    keys = [key(b) for b in raw]
    prefix, mask, remaining = 0, 0, k
    for shift, bins in ((21, 2048), (10, 2048), (0, 1024)):
        histogram = [0] * bins
        for value in keys:
            if value & mask == prefix:
                histogram[(value >> shift) & (bins-1)] += 1
        chunk = bins // THREADS
        counts = [sum(histogram[t*chunk:(t+1)*chunk]) for t in range(THREADS)]
        ranks, _, total, _ = prefix_pair(counts, [0]*THREADS)
        assert 0 < remaining <= total
        owners = []
        for thread in range(THREADS):
            before = ranks[thread]
            for j in range(chunk):
                bucket = thread*chunk+j
                count = histogram[bucket]
                if before < remaining <= before+count:
                    owners.append((bucket, remaining-before))
                before += count
        assert len(owners) == 1
        bucket, remaining = owners[0]
        prefix |= bucket << shift
        mask |= (bins-1) << shift
    strict_count = k-remaining
    selected = [None] * k
    strict_seen = equal_seen = 0
    for tile in range(0, len(raw), THREADS):
        strict = [int(tile+t < len(raw) and keys[tile+t] < prefix) for t in range(THREADS)]
        equal = [int(tile+t < len(raw) and keys[tile+t] == prefix) for t in range(THREADS)]
        strict_rank, equal_rank, tile_strict, tile_equal = prefix_pair(strict, equal)
        for thread in range(THREADS):
            slot = k
            if strict[thread]:
                slot = strict_seen+strict_rank[thread]
            if equal[thread] and equal_seen+equal_rank[thread] < remaining:
                slot = strict_count+equal_seen+equal_rank[thread]
            if slot < k:
                assert selected[slot] is None, '候选槽不能由两个线程抢写'
                selected[slot] = (keys[tile+thread], tile+thread)
        strict_seen += tile_strict
        equal_seen += tile_equal
    assert strict_seen == strict_count and equal_seen >= remaining
    assert all(item is not None for item in selected)
    ordered = sort_candidates(selected)
    ids = [item[1] for item in ordered]
    return ids, [raw[i] for i in ids]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', default=str(ROOT/'results/radix_model_check.json'))
    args = parser.parse_args()
    rng = random.Random(95020261016)
    cases = 0
    for n in (1, 2, 6, 31, 32, 33, 40, 127, 128, 129, 257, 513, 1025, 2048, 4097, 65536):
        patterns = [
            [0]*n, [float_bits(1.)]*n,
            [0x80000000 if i % 2 else 0 for i in range(n)],
            [float_bits(float(i)) for i in range(n)],
            [float_bits(float(n-i)) for i in range(n)],
            [float_bits(float(rng.randrange(9))) for _ in range(n)],
            [rng.randrange(0x7f800000) for _ in range(n)],
            [0x3f800001 if i % 2 else 0x3f800000 for i in range(n)],
            [1 if i % 3 == 0 else 0 for i in range(n)],
            [0x7f7fffff if i % 7 == 0 else 0 for i in range(n)],
            [float_bits(99.) if i in (0, n//2, n-1) else 0 for i in range(n)],
        ]
        for raw in patterns:
            for k in sorted({1, min(10,n), min(32,n)}):
                # Independent oracle: full numerical score sort, then original ID; no radix keys.
                expected = sorted(range(n), key=lambda i: (-as_float(raw[i]), i))[:k]
                actual, output = select(raw, k)
                assert actual == expected, (n, k, actual, expected)
                assert output == [raw[i] for i in expected]
                cases += 1
    for raw in (0x7f800000, 0xff800000, 0x7fc00001, 0xbf800000, 0x80000001):
        try:
            select([0, raw], 1)
        except ValueError:
            pass
        else:
            raise AssertionError('非法分数必须拒绝')
    for k in (0, 33, 3):
        try:
            select([0, 0], k)
        except ValueError:
            pass
        else:
            raise AssertionError('非法K必须拒绝')
    for blocks in (1, 3, 56):
        rows = [q for block in range(blocks) for q in range(block, 513, blocks)]
        assert sorted(rows) == list(range(513)) and len(set(rows)) == 513
    names = ['npu_prototype/topk_radix/op_host/bm25_top_k.cpp',
             'npu_prototype/topk_radix/op_host/bm25_top_k_tiling.h',
             'npu_prototype/topk_radix/op_kernel/bm25_top_k.cpp',
             'standalone/check_radix_model.py']
    record = {'status': 'python_schedule_model_passed_not_device_validation', 'cases_passed': cases,
              'threads': THREADS, 'radix_widths': [11,11,10], 'ub_algorithm_bytes': 8512,
              'strict_ids_and_original_score_bits': True, 'invalid_input_cases_passed': 8,
              'compiled': False, 'npu_verified': False, 'performance_verified': False,
              'source_sha256': {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}}
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((json.dumps(record, ensure_ascii=False, indent=2)+'\n').encode('utf-8'))
    print(f'Radix线程分组模型通过：{cases}组，另含8组非法输入、513行分核覆盖检查。')
    print('只验证Python模型中的门槛、前缀编号、最终排序及原FP32字节；未编译或执行NPU。')
    print('报告：', target)


if __name__ == '__main__':
    main()
