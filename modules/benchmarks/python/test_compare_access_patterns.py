from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import compare_access_patterns as comparison
import numpy_access_patterns as numpy_suite


class CompareAccessPatternsTest(unittest.TestCase):
    def test_score_to_ns(self) -> None:
        self.assertEqual(comparison.score_to_ns(1.5, "us/op"), 1_500.0)

    def test_signature_validation_accepts_matched_rows(self) -> None:
        row = {
            "logical_work_units": 16,
            "work_unit": "element",
            "result_size": 8,
            "result_layout": "c_contiguous",
            "sum": 4.0,
            "weighted_sum": 12.0,
        }
        comparison.validate_signatures(
            {("inner_stride_add", 4): row},
            {("inner_stride_add", 4): dict(row)},
        )

    def test_signature_validation_rejects_layout_mismatch(self) -> None:
        ravel = {
            ("transpose_add", 4): {
                "logical_work_units": 16,
                "work_unit": "element",
                "result_size": 16,
                "result_layout": "c_contiguous",
                "sum": 4.0,
                "weighted_sum": 12.0,
            }
        }
        numpy = {
            ("transpose_add", 4): {
                **ravel[("transpose_add", 4)],
                "result_layout": "strided_view",
            }
        }
        with self.assertRaisesRegex(ValueError, "result_layout"):
            comparison.validate_signatures(ravel, numpy)

    def test_signatures_only_emits_all_cases_without_timings(self) -> None:
        payload = numpy_suite.run_suite(
            [4],
            warmups=0,
            samples=1,
            target_ms=1.0,
            signatures_only=True,
        )
        self.assertEqual(payload["metadata"]["mode"], "signatures_only")
        names = {row["case"] for row in payload["results"]}
        self.assertEqual(names, set(comparison.CASE_ORDER))
        for row in payload["results"]:
            self.assertNotIn("median_ns", row)
            self.assertIn("weighted_sum", row)

    def test_parity_only_cli_accepts_matching_payloads(self) -> None:
        row = {
            "case": "contiguous_add",
            "side": 4,
            "family": "elementwise",
            "logical_work_units": 16,
            "work_unit": "element",
            "result_size": 16,
            "result_layout": "c_contiguous",
            "sum": 1.0,
            "weighted_sum": 2.0,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            signatures = root / "ravel.json"
            numpy_path = root / "numpy.json"
            signatures.write_text(
                json.dumps({"results": [row]}),
                encoding="utf-8",
            )
            numpy_path.write_text(
                json.dumps({"metadata": {}, "results": [row]}),
                encoding="utf-8",
            )
            old_argv = sys.argv
            try:
                sys.argv = [
                    "compare_access_patterns.py",
                    "--signatures",
                    str(signatures),
                    "--numpy",
                    str(numpy_path),
                    "--parity-only",
                ]
                comparison.main()
            finally:
                sys.argv = old_argv


if __name__ == "__main__":
    unittest.main()
