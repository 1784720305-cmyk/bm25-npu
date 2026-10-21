"""FAKE install/runner and MOCK child protocol tests; never execute ACLNN or NPU."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone import validate_topk_positive_compact as tool


def main():
    checks = 0
    (ROOT/'results').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='positive_compact_MOCK_', dir=ROOT/'results') as temp:
        root = Path(temp)
        vendor, project = root/tool.VENDOR, root/'project'
        header = vendor/'op_api/include/aclnn_bm25_top_k.h'
        header.parent.mkdir(parents=True)
        header.write_text('\n'.join('aclnnStatus '+key+'('+value+');' for key, value in tool.old.PROTOTYPES.items()), encoding='utf-8')
        library = vendor/'op_api/lib/libcust_opapi.so'
        library.parent.mkdir(parents=True)
        library.write_bytes(b'FAKE LIBRARY NOT EXECUTABLE')
        obj = vendor/'op_impl/ai_core/tbe/kernel/ascend950/bm25_top_k/MOCK.o'
        obj.parent.mkdir(parents=True)
        obj.write_bytes(b'FAKE DEVICE OBJECT NEVER EXECUTED')
        build = project/'build_out'
        build.mkdir(parents=True)
        for path in (header, library, obj):
            (build/path.name).write_bytes(path.read_bytes())
        info = {'operator': 'topk_positive_compact', 'vendor_name': tool.VENDOR, 'compute_unit': 'ascend950',
                'positive_capacity': 1024, 'compaction_min_docs': 1025, 'membership_mask_max_docs': 8192,
                'selection_rule': 'FP32_score_desc_original_ID_asc_signed_zero_equal', 'source_sha256': {}}
        for name in tool.SOURCES:
            path = project/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT/'npu_prototype/topk_positive_compact'/name).read_bytes())
            info['source_sha256'][name] = tool.digest(path)
        tool.write_json(project/'bm25_project.json', info)
        runner = root/'MOCK_NOT_AN_EXECUTABLE'
        runner.write_bytes(b'MOCK ONLY')
        tool.validate_install(vendor, project); checks += 1

        def fails(call):
            nonlocal checks
            try:
                call()
            except (ValueError, OSError, subprocess.SubprocessError):
                checks += 1
            else:
                raise AssertionError('invalid state accepted')

        # Build-tree RPATH can differ, but at least one packaged API copy must match.
        extra = build/'intermediate/libcust_opapi.so'
        extra.parent.mkdir()
        extra.write_bytes(b'FAKE DIFFERENT BUILD RPATH')
        tool.validate_install(vendor, project); checks += 1
        for path in (header, library, obj, project/tool.SOURCES[0]):
            original = path.read_bytes()
            path.write_bytes(original+b'CORRUPT')
            fails(lambda: tool.validate_install(vendor, project))
            path.write_bytes(original)
        for key, value in (('positive_capacity', 512), ('compute_unit', 'ascend310'), ('membership_mask_max_docs', 8193),
                           ('selection_rule', 'approximately_equal'), ('vendor_name', 'bm25_topk_radix_collect')):
            bad = copy.deepcopy(info); bad[key] = value
            tool.write_json(project/'bm25_project.json', bad)
            fails(lambda: tool.validate_install(vendor, project))
        tool.write_json(project/'bm25_project.json', info)
        original = header.read_bytes()
        header.write_bytes(original.replace(b'int64_tk', b'int32_tk'))
        fails(lambda: tool.validate_install(vendor, project))
        header.write_bytes(original)

        def record(q, docs, k, device=2, calls=5):
            return {'status': 'positive_compact_device_calls_and_cpp_full_sort_passed', 'queries': q, 'docs': docs, 'k': k,
                    'device': device, 'validated_calls': calls, 'workspace_bytes': 16777216,
                    'declared_user_gm_scratch_bytes': 0, 'declared_ub_algorithm_bytes': 16704,
                    'input_calls': [(calls+1)//2, calls//2], 'every_call_poisoned': True,
                    'every_call_full_sort_checked': True, 'host_source_sha256': tool.host_hashes()}
        good = record(65, 8192, 10)
        tool.validate_record(good, 65, 8192, 10, 2, 5); checks += 1
        for key, value in (('declared_ub_algorithm_bytes', 8512), ('workspace_bytes', True),
                           ('every_call_poisoned', False), ('input_calls', [5, 0]),
                           ('host_source_sha256', {}), ('validated_calls', 3), ('queries', True)):
            bad = copy.deepcopy(good); bad[key] = value
            fails(lambda: tool.validate_record(bad, 65, 8192, 10, 2, 5))
        bad = copy.deepcopy(good); bad['unexpected'] = 0
        fails(lambda: tool.validate_record(bad, 65, 8192, 10, 2, 5))

        # Verify the prepared domain, exact counts and source-ID trap, not just metadata labels.
        for docs in (1024, 1025, 8192, 8193):
            for k in (1, 10, 32):
                for alternate in (False, True):
                    values, labels = tool.boundary_rows(docs, k, alternate)
                    assert len(values) == 65*docs and len(labels) == 65
                    counts = set()
                    for query, label in enumerate(labels):
                        row = values[query*docs:(query+1)*docs]
                        count = sum(value > 0 for value in row)
                        if label.startswith('positive_count_'):
                            assert count == int(label.rsplit('_', 1)[1])
                            counts.add(count)
                        if label.startswith('original_ID'):
                            ranked = sorted(range(docs), key=lambda doc: (-row[doc], doc))[:k]
                            assert ranked[0] == 1 and (k == 1 or ranked[1] == 128)
                    assert {0, min(1023, docs), min(1024, docs), min(1025, docs)} <= counts
                    checks += 1

        child_calls = []
        def mock_child(command, log, environment):
            assert environment['ASCEND_CUSTOM_OPP_PATH'] == str(vendor.resolve())
            assert command[0] == str(runner.resolve())
            log.write_bytes(b'MOCK CHILD ONLY -- NO DEVICE EXECUTION\n')
            if command[1] == '--check-library':
                assert command[2] == str(library.resolve())
                return
            directory, output = Path(command[1]), Path(command[2])
            q, docs, k, device = map(int, command[3:7])
            assert command[7] == str(library.resolve())
            output.mkdir()
            for name in ('a', 'b'):
                source = directory/f'input_{name}'
                for suffix in ('ids', 'scores'):
                    (output/f'{name}_top_{suffix}.bin').write_bytes((source/f'expected_{suffix}.bin').read_bytes())
            tool.write_json(output/'device_validation.json', record(q, docs, k, device, int(command[8])))
            child_calls.append(directory)

        # Full orchestration is mocked; all 37 real fixtures and independent Python sorts execute.
        with patch.object(tool.old, 'execute', mock_child), redirect_stdout(io.StringIO()):
            tool.run(root/'all_mock', runner, vendor, project, 2, 5)
        report = json.loads((root/'all_mock/report.json').read_text(encoding='utf-8'))
        assert report['status'] == tool.STATUS and len(report['cases']) == len(child_calls) == 37
        assert all(item['status'] == 'passed' for item in report['cases']); checks += 3
        fails(lambda: tool.run(root/'all_mock', runner, vendor, project, 2, 5))
        fails(lambda: tool.run(root/'bad_calls', runner, vendor, project, 2, 3))
        fails(lambda: tool.run(root/'bad_device', runner, vendor, project, -1, 5))

        def child_failure(command, log, environment):
            raise subprocess.CalledProcessError(1, command)
        with patch.object(tool.old, 'execute', child_failure):
            fails(lambda: tool.run(root/'failed_child', runner, vendor, project, 2, 5))
        failed = json.loads((root/'failed_child/report.json').read_text(encoding='utf-8'))
        assert failed['status'] == 'failed' and failed['cases'] == []; checks += 1

        def stale_or_wrong(command, log, environment, mode):
            mock_child(command, log, environment)
            if command[1] == '--check-library':
                return
            output = Path(command[2])
            if mode == 'stale_binary':
                bad = json.loads((output/'device_validation.json').read_text(encoding='utf-8'))
                bad['host_source_sha256'] = {}
                tool.write_json(output/'device_validation.json', bad)
            else:
                (output/'a_top_ids.bin').write_bytes(struct.pack('<i', -1))
        for mode in ('stale_binary', 'wrong_output'):
            with patch.object(tool.old, 'execute', lambda command, log, environment: stale_or_wrong(command, log, environment, mode)), redirect_stdout(io.StringIO()):
                fails(lambda: tool.run(root/mode, runner, vendor, project, 2, 5))
            failed = json.loads((root/mode/'report.json').read_text(encoding='utf-8'))
            assert failed['status'] == 'failed' and failed['cases'][0]['status'] == 'failed'; checks += 1
    print(f'PASS: {checks} fixture/provenance/MOCK-child/failure checks, including all 37 case preparations. '
          'Fake binaries only; no C++ build, ACLNN or NPU execution/performance evidence.')


if __name__ == '__main__':
    main()
