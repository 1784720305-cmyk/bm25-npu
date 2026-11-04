"""Check that incomplete or stale device reports cannot be accepted by the orchestration gate."""

import copy
import tempfile
import unittest
from pathlib import Path

from run_a5_acceptance import (
    MERGE_STATUS,
    SEARCH_STATUS,
    SELECTOR_STATUS,
    SHARED_STATUS,
    dependency_snapshot,
    validate_device_reports,
)


class AcceptanceContractTests(unittest.TestCase):
    def setUp(self):
        self.selector = {"status": SELECTOR_STATUS, "calls": 54}
        self.merge = {"status": MERGE_STATUS, "calls": 36}
        self.shared = {"status": SHARED_STATUS, "device": 5, "init_close_cycles": 2, "queries": 8, "k": 10}
        self.runners = {"npu": "current_npu", "cpu": "current_cpu", "cpu_directory": "current_directory"}

        def summary(median, p95):
            return {
                "summary": {
                    case: {"samples": 30, "median_batch_ms": median, "p95_batch_ms": p95} for case in ("A", "B")
                }
            }

        rounds = [
            {"cpu": summary(2.0, 3.0), "cpu_directory": summary(1.9, 2.9), "npu": summary(1.0, 1.5)} for _ in range(6)
        ]
        self.search = {
            "status": SEARCH_STATUS,
            "per_round": rounds,
            "runner_sha256": self.runners,
            "cpu_decision": {
                "performance_target_met_at_this_scale": True,
                "cpu_roles": ["cpu", "cpu_directory"],
                "cases": {case: {"passed": True} for case in ("A", "B")},
            },
        }

    def check(self, **changes):
        values = {"selector": self.selector, "merge": self.merge, "shared": self.shared, "search": self.search}
        values.update(changes)
        return validate_device_reports(**values, expected_device=5, expected_runners=self.runners)

    def test_complete_report_and_separate_performance_failure(self):
        self.assertTrue(self.check())
        slower = copy.deepcopy(self.search)
        slower["cpu_decision"]["performance_target_met_at_this_scale"] = False
        for row in slower["per_round"]:
            for case in ("A", "B"):
                row["npu"]["summary"][case].update(median_batch_ms=3.0, p95_batch_ms=4.0)
                slower["cpu_decision"]["cases"][case]["passed"] = False
        self.assertFalse(self.check(search=slower))

    def test_missing_backend_samples_and_nonfinite_timings_rejected(self):
        missing = copy.deepcopy(self.search)
        del missing["per_round"][0]["cpu_directory"]
        bad_samples = copy.deepcopy(self.search)
        bad_samples["per_round"][0]["npu"]["summary"]["A"]["samples"] = 29
        invalid_timing = copy.deepcopy(self.search)
        invalid_timing["per_round"][0]["npu"]["summary"]["B"]["p95_batch_ms"] = float("nan")
        for data in (missing, bad_samples, invalid_timing):
            with self.assertRaises(ValueError):
                self.check(search=data)

    def test_reported_performance_cannot_disagree_with_rounds(self):
        data = copy.deepcopy(self.search)
        data["cpu_decision"]["performance_target_met_at_this_scale"] = False
        with self.assertRaises(ValueError):
            self.check(search=data)

    def test_incomplete_operator_checks_rejected(self):
        for name in ("selector", "merge"):
            data = copy.deepcopy(getattr(self, name))
            data["calls"] -= 1
            with self.subTest(stage=name), self.assertRaises(ValueError):
                self.check(**{name: data})

    def test_shared_context_must_run_on_expected_device(self):
        for key, value in (("device", 0), ("init_close_cycles", 1), ("queries", 1), ("k", 1), ("status", "not_run")):
            data = {**self.shared, key: value}
            with self.subTest(field=key), self.assertRaises(ValueError):
                self.check(shared=data)

    def test_missing_rounds_or_stale_runner_rejected(self):
        for changes in (
            {"per_round": []},
            {"per_round": None},
            {"per_round": [{}] * 5},
            {"runner_sha256": {"npu": "old"}},
            {"status": "running"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.check(search={**self.search, **changes})

    def test_performance_decision_must_be_an_explicit_boolean(self):
        for value in (None, "true", 1, {}, {"performance_target_met_at_this_scale": "false"}):
            decision = {"performance_target_met_at_this_scale": value}
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.check(search={**self.search, "cpu_decision": decision})
        with self.assertRaises(ValueError):
            self.check(search={**self.search, "cpu_decision": None})


class DependencySnapshotTests(unittest.TestCase):
    def test_cann_leaf_header_changes_and_removal_invalidate_snapshot(self):
        # This fixture tests fingerprint invalidation, not ELF or CANN acceptance.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = (
                "sdk/library.so",
                "sdk/feature_retrieval/src/ascendfaiss/common/ErrorCode.h",
                "sdk/feature_retrieval/src/ascendfaiss/ascenddaemon/AscendResourcesProxy.h",
                "sdk/feature_retrieval/src/ascendfaiss/ascenddaemon/StandardAscendResources.h",
                "toolkit/lib64/libascendcl.so",
                "toolkit/lib64/libnnopbase.so",
                "toolkit/include/acl/acl.h",
                "toolkit/include/acl/acl_base_rt.h",
                "toolkit/include/acl/acl_rt.h",
                "toolkit/include/aclnn/acl_meta.h",
                "vendors/bm25_fused_score_topk/op_api/include/aclnn_bm25_fused_score_top_k.h",
                "vendors/bm25_fused_score_topk/op_api/lib/libcust_opapi.so",
                "vendors/bm25_topk_merge/op_api/include/aclnn_bm25_top_k_merge.h",
                "vendors/bm25_topk_merge/op_api/lib/libcust_opapi.so",
            )
            for name in files:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fingerprint fixture")
            values = {
                "toolkit_home": str(root / "toolkit"),
                "indexsdk_source": str(root / "sdk"),
                "sdk_library": str(root / "sdk/library.so"),
                "vendors": str(root / "vendors"),
            }
            baseline = dependency_snapshot(values)
            for name in ("acl/acl_base_rt.h", "acl/acl_rt.h", "aclnn/acl_meta.h"):
                with self.subTest(header=name):
                    header = root / "toolkit/include" / name
                    header.write_bytes(b"changed header")
                    self.assertNotEqual(baseline, dependency_snapshot(values))
                    header.unlink()
                    with self.assertRaises(FileNotFoundError):
                        dependency_snapshot(values)
                    header.write_bytes(b"fingerprint fixture")
            meta = root / "toolkit/include/aclnn/acl_meta.h"
            alternate = root / "toolkit/include/aclnnop/aclnn/acl_meta.h"
            alternate.parent.mkdir(parents=True)
            meta.rename(alternate)
            self.assertEqual(str(alternate.resolve()), dependency_snapshot(values)["acl_meta_header"]["path"])


if __name__ == "__main__":
    unittest.main()
