"""Check fail-fast environment diagnostics and the CPU comparison's correctness guardrails."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from a5_environment import runtime_preflight
from compare_cpu_builds import ROOT, checkpoint_hashes, cpu_list, restore_legacy_cpu_flags, timing_summary


class RuntimePreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        home = root / "cann"
        toolkit = home / "x86_64-linux"
        toolkit.mkdir(parents=True)
        (toolkit / "include").mkdir()
        (toolkit / "include/securec.h").write_text("unit fixture", encoding="utf-8")
        (home / "opp").mkdir()
        vendors = root / "vendors"
        for alias in ("bm25_fused_score_topk", "bm25_topk_merge"):
            vendor = vendors / alias
            config = vendor / "op_impl/ai_core/tbe/kernel/config/ascend950/binary_info_config.json"
            config.parent.mkdir(parents=True)
            config.write_text('{"fixture":true}', encoding="utf-8")
            tiling = vendor / "op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/libfixture.so"
            tiling.parent.mkdir(parents=True)
            tiling.write_bytes(b"unit fixture only")
        self.values = {"toolkit_home": str(toolkit), "vendors": str(vendors), "sdk_library": str(root / "sdk.so")}
        self.environment = {
            "ASCEND_HOME_PATH": str(home),
            "ASCEND_OPP_PATH": str(home / "opp"),
            "MX_INDEX_USE_ONLINEOP": "1",
        }

    def inspect(self, environment, dependency_output="resolved"):
        # Test metadata checks only: no actual ELF/CANN/device acceptance is claimed.
        with (
            patch("a5_environment.platform.system", return_value="Linux"),
            patch("a5_environment.cpu_snapshot", return_value={}),
            patch("a5_environment.command_output", return_value={"exit_code": 0, "output": dependency_output}),
        ):
            return runtime_preflight(self.values, environment)

    def test_reports_all_missing_environment_prerequisites_together(self):
        result = self.inspect({})
        self.assertEqual("failed", result["status"])
        message = ";".join(result["errors"])
        for name in ("ASCEND_HOME_PATH", "ASCEND_OPP_PATH", "MX_INDEX_USE_ONLINEOP"):
            self.assertIn(name, message)

    def test_valid_metadata_does_not_claim_device_execution(self):
        result = self.inspect(self.environment)
        self.assertEqual("passed", result["status"])
        self.assertFalse(result["device_execution_performed"])
        self.assertFalse(result["offline_model_execution_validated"])

    def test_missing_dependency_and_corrupt_config_rejected(self):
        self.assertEqual("failed", self.inspect(self.environment, "libmissing.so => not found")["status"])
        path = next(Path(self.values["vendors"]).rglob("binary_info_config.json"))
        path.write_text("broken json", encoding="utf-8")
        self.assertEqual("failed", self.inspect(self.environment)["status"])

    def test_mismatched_cann_and_invalid_sdk_mode_rejected(self):
        for changes in ({"ASCEND_HOME_PATH": self.temporary.name + "/vendors"}, {"MX_INDEX_USE_ONLINEOP": "11"}):
            with self.subTest(changes=changes):
                self.assertEqual("failed", self.inspect({**self.environment, **changes})["status"])


class CpuComparisonTests(unittest.TestCase):
    def test_cpu_list_rejects_reversed_and_malformed_ranges(self):
        self.assertEqual([0, 1, 2, 8], cpu_list("0-2,8,1"))
        for text in ("3-1", "-1", "1-2-3", "", "x"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                cpu_list(text)

    def test_p95_uses_the_same_nearest_rank_definition_as_acceptance(self):
        record = {
            "queries": 8,
            "measured_cycles": 30,
            "index_readback_byte_identical": True,
            "samples": [{"case": case, "call_wall_ms": i + 1.0} for case in ("A", "B") for i in range(30)],
        }
        summary = timing_summary(record, 8, 30)
        self.assertEqual(29, summary["A"]["p95_batch_ms"])
        self.assertEqual(15.5, summary["B"]["median_batch_ms"])
        record["samples"][0]["call_wall_ms"] = float("nan")
        with self.assertRaises(ValueError):
            timing_summary(record, 8, 30)

    def test_unreviewed_experiment_recipe_rejected(self):
        recipe = (ROOT / "feature_retrieval/src/ascendfaiss/CMakeLists.txt").read_text(encoding="utf-8")
        changed = restore_legacy_cpu_flags(recipe)
        self.assertIn("-ffp-contract=off -fno-fast-math", changed)
        with self.assertRaises(ValueError):
            restore_legacy_cpu_flags(changed)

    def test_output_bit_changes_stop_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for checkpoint in ("before_A", "before_B", "after_A", "after_B"):
                (root / checkpoint).mkdir()
                for name in ("top_scores.bin", "top_ids.bin", "scores.bin"):
                    (root / checkpoint / name).write_bytes(checkpoint[-1].encode())
            self.assertEqual(12, len(checkpoint_hashes(root)))
            (root / "after_A/scores.bin").write_bytes(b"different")
            with self.assertRaises(ValueError):
                checkpoint_hashes(root)


if __name__ == "__main__":
    unittest.main()
