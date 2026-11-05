"""Check bounded diagnostic allocations, original errors, 64-bit sizes and exact release."""

import ctypes
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import probe_device_allocation as probe


class FakeAcl:
    def __init__(self):
        self.calls = []
        self.live = False
        self.memory_calls = 0
        self.fail_memory_call = 0

    def aclrtGetMemInfo(self, kind, free, total):
        self.memory_calls += 1
        if self.memory_calls == self.fail_memory_call:
            return 42
        ctypes.cast(free, ctypes.POINTER(ctypes.c_size_t))[0] = (100 if self.live else 122) * 1024**3
        ctypes.cast(total, ctypes.POINTER(ctypes.c_size_t))[0] = 123 * 1024**3
        return 0

    def aclrtMalloc(self, pointer, size, policy):
        self.calls.append(('allocate', size, policy))
        if policy == 2:
            return 207001
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p))[0] = 64
        self.live = True
        return 0

    def aclrtFree(self, pointer):
        self.calls.append(('free', pointer.value))
        self.live = False
        return 0

    def aclGetRecentErrMsg(self):
        return b'injected allocation failure'


class AllocationProbeTests(unittest.TestCase):
    def test_policy_failure_keeps_error_and_success_frees_exact_large_allocation(self):
        api = FakeAcl()
        size = 14051296472
        normal = probe.trial(api, size, *probe.POLICIES[0])
        huge = probe.trial(api, size, *probe.POLICIES[1])
        self.assertFalse(normal['allocated'])
        self.assertEqual(207001, normal['allocation_code'])
        self.assertIn('injected allocation failure', normal['allocation_error'])
        self.assertTrue(huge['allocated'] and huge['freed'])
        self.assertEqual([('allocate', size, 2), ('allocate', size, 0), ('free', 64)], api.calls)
        self.assertFalse(api.live)

    def test_memory_query_failure_after_allocation_still_releases_it(self):
        api = FakeAcl()
        api.fail_memory_call = 2
        with self.assertRaisesRegex(RuntimeError, 'aclrtGetMemInfo: code=42'):
            probe.trial(api, 14051296472, *probe.POLICIES[1])
        self.assertEqual(('free', 64), api.calls[-1])
        self.assertFalse(api.live)

    def test_budget_failure_does_not_attempt_allocation(self):
        api = FakeAcl()
        with self.assertRaisesRegex(ValueError, '80%'):
            probe.trial(api, 122 * 1024**3, *probe.POLICIES[1])
        self.assertEqual([], api.calls)


if __name__ == '__main__':
    unittest.main()
