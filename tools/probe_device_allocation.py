"""Compare two ACL allocation policies for the actual directory size; no Search or kernels."""

import argparse
import ctypes
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone import final_support as support
from standalone import large_corpus as corpus
from standalone import resident_index as r

POLICIES = (('ACL_MEM_MALLOC_NORMAL_ONLY', 2), ('ACL_MEM_MALLOC_HUGE_FIRST', 0))


def require_acl(api, status, operation):
    if status:
        detail = (api.aclGetRecentErrMsg() or b'').decode('utf-8', errors='replace')
        raise RuntimeError(f'{operation}: code={status}, detail={detail}')


def memory(api):
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    require_acl(api, api.aclrtGetMemInfo(0, ctypes.byref(free), ctypes.byref(total)), 'aclrtGetMemInfo')
    return {'free': free.value, 'total': total.value}


def trial(api, size, name, policy):
    before = memory(api)
    r.require(0 < size < before['free'] * 4 // 5, 'Probe request exceeds 80% of actual free HBM')
    pointer = ctypes.c_void_p()
    status = api.aclrtMalloc(ctypes.byref(pointer), size, policy)
    detail = (api.aclGetRecentErrMsg() or b'').decode('utf-8', errors='replace') if status else ''
    result = {
        'policy': name,
        'policy_value': policy,
        'requested_bytes': size,
        'before': before,
        'allocation_code': status,
        'allocation_error': detail,
        'allocated': status == 0,
    }
    if status == 0:
        try:
            r.require(pointer.value is not None, 'ACL returned success with a null pointer')
            result['allocated_memory'] = memory(api)
        finally:
            if pointer.value is not None:
                require_acl(api, api.aclrtFree(pointer), 'aclrtFree probe allocation')
        result['freed'] = True
    result['after'] = memory(api)
    return result


def load_api(home):
    candidates = (home / 'lib64/libascendcl.so', home / 'x86_64-linux/lib64/libascendcl.so')
    library = next((path.resolve() for path in candidates if path.is_file()), None)
    r.require(library is not None, 'CANN libascendcl.so not found')
    api = ctypes.CDLL(str(library))
    signatures = {
        'aclInit': ([ctypes.c_char_p], ctypes.c_int),
        'aclFinalize': ([], ctypes.c_int),
        'aclrtSetDevice': ([ctypes.c_int32], ctypes.c_int),
        'aclrtResetDevice': ([ctypes.c_int32], ctypes.c_int),
        'aclrtGetMemInfo': (
            [ctypes.c_int, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)],
            ctypes.c_int,
        ),
        'aclrtMalloc': ([ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int], ctypes.c_int),
        'aclrtFree': ([ctypes.c_void_p], ctypes.c_int),
        'aclGetRecentErrMsg': ([], ctypes.c_char_p),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(api, name)
        function.argtypes, function.restype = arguments, result
    return api, library


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', type=int, default=5)
    parser.add_argument('--toolkit-home', type=Path, default=Path('/usr/local/Ascend/cann-9.2.0/x86_64-linux'))
    args = parser.parse_args()
    args.input, args.output = args.input.resolve(), args.output.resolve()
    r.require(sys.platform == 'linux' and ctypes.sizeof(ctypes.c_size_t) == 8, '64-bit A5 Linux required')
    r.require(args.output.is_relative_to(ROOT / 'results') and not args.output.exists(), 'Fresh results JSON required')
    r.require(0 <= args.device <= 2**31 - 1, 'Invalid device ID')
    description, index = corpus.validate(args.input, rehash=False)
    plan = support.layout(
        description['docs'],
        index['vocab_size'],
        index['posting_count'],
        description['queries'],
        description['k'],
        description['capacity_words'],
    )
    size = plan['directory_bytes'] + 128
    api, library = load_api(args.toolkit_home)
    report = {
        'status': 'running',
        'device': args.device,
        'library': str(library),
        'requested_bytes': size,
        'scope': 'One resident-directory-sized allocation at a time; freed after each policy. No index upload, kernels, graph or Search. Aggregate free HBM is not a per-policy allocation guarantee.',
        'trials': [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    initialized = selected = False
    try:
        require_acl(api, api.aclInit(None), 'aclInit')
        initialized = True
        require_acl(api, api.aclrtSetDevice(args.device), 'aclrtSetDevice')
        selected = True
        for name, policy in POLICIES:
            result = trial(api, size, name, policy)
            report['trials'].append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        report['status'] = 'allocation_probe_completed_search_not_run'
    except (Exception, KeyboardInterrupt) as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        cleanup = {}
        if selected:
            cleanup['reset_device_code'] = api.aclrtResetDevice(args.device)
        if initialized:
            cleanup['finalize_code'] = api.aclFinalize()
        report['cleanup'] = cleanup
        if any(cleanup.values()):
            report['status'] = 'failed_cleanup'
        r.write_json(args.output, report)
    print('REPORT: ' + str(args.output), flush=True)
    r.require(report['status'] == 'allocation_probe_completed_search_not_run', 'Probe cleanup failed')
    r.require(
        report['trials'][-1]['allocated'], 'Huge-first allocation failed; retain report before rebuilding/rerunning'
    )


if __name__ == '__main__':
    main()
