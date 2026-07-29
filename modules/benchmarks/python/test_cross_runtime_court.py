from __future__ import annotations

import sys
import statistics
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import compare_cross_runtime_court as court  # noqa: E402


def row(
    *,
    name: str = "case",
    side: int = 64,
    family: str = "binary",
    comparison: str = "exact",
    result_dtype: str = "int32",
    total: float = 3.0,
    weighted: float = 5.0,
    median_ns: float | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "case": name,
        "family": family,
        "side": side,
        "input_dtype": result_dtype,
        "result_dtype": result_dtype,
        "input_layout": "contiguous",
        "logical_work_units": side * side,
        "work_unit": "element",
        "comparison": comparison,
        "timing_scope": "all_platforms",
        "timing_status": "not_requested" if median_ns is None else "measured",
        "result_size": side * side,
        "result_layout": "c_contiguous",
        "sum": total,
        "weighted_sum": weighted,
        "nan_count": 0,
        "positive_infinity_count": 0,
        "negative_infinity_count": 0,
        "negative_zero_count": 0,
    }
    if median_ns is not None:
        result["median_ns"] = median_ns
        result["relative_standard_deviation"] = 0.0
        result["samples_ns"] = [median_ns] * 7
    return result


def set_timing_samples(
    result: dict[str, object],
    samples: list[float],
) -> None:
    result["samples_ns"] = samples
    result["median_ns"] = statistics.median(samples)
    result["relative_standard_deviation"] = statistics.pstdev(
        samples
    ) / statistics.fmean(samples)


def payload(
    rows: list[dict[str, object]],
    *,
    runtime_name: str,
    runtime_version: str = "v24.1.0",
) -> dict[str, object]:
    return {
        "schema": "ravel.cross-runtime-court.v1",
        "metadata": {
            "runtime_name": runtime_name,
            "runtime_version": runtime_version,
            "vm_name": "V8 test",
            "scala_version": "3.7.4",
            "os_name": "darwin",
            "os_arch": "arm64",
            "available_processors": 10,
            "timer": "performance.now",
            "explicit_gc_available": True,
            "mode": "timed",
            "sides": [64],
            "warmup_ms": 200,
            "sample_ms": 250,
            "samples": 7,
            "case_filter": None,
        },
        "results": rows,
    }


class CrossRuntimeCourtTest(unittest.TestCase):
    def test_parity_accepts_exact_and_tolerant_floating_rows(self) -> None:
        exact = row()
        floating = row(
            name="floating",
            comparison="floating",
            result_dtype="float64",
            total=1.0,
            weighted=2.0,
        )
        jvm = payload([exact, floating], runtime_name="jvm")
        js = payload(
            [
                dict(exact),
                {
                    **floating,
                    "sum": 1.0 + 1.0e-11,
                    "weighted_sum": 2.0 - 1.0e-11,
                },
            ],
            runtime_name="node",
        )
        court.validate_semantic_parity(jvm, js)

    def test_parity_rejects_metadata_and_exact_signature_drift(self) -> None:
        exact = row()
        with self.assertRaisesRegex(ValueError, "input_layout"):
            court.validate_semantic_parity(
                payload([exact], runtime_name="jvm"),
                payload([{**exact, "input_layout": "reversed"}], runtime_name="node"),
            )
        with self.assertRaisesRegex(ValueError, "weighted_sum"):
            court.validate_semantic_parity(
                payload([exact], runtime_name="jvm"),
                payload([{**exact, "weighted_sum": 6.0}], runtime_name="node"),
            )

    def test_node_comparison_reports_every_row_and_visible_regression(self) -> None:
        baseline = payload(
            [
                row(name="faster", median_ns=100.0),
                row(name="slower", family="predicate", median_ns=100.0),
            ],
            runtime_name="node",
        )
        slower_candidate = row(
            name="slower",
            family="predicate",
            median_ns=125.0,
        )
        set_timing_samples(
            slower_candidate,
            [87.5, 100.0, 112.5, 125.0, 137.5, 150.0, 162.5],
        )
        candidate = payload(
            [
                row(name="faster", median_ns=50.0),
                slower_candidate,
            ],
            runtime_name="node",
        )
        rows, report = court.compare_node_timings(
            baseline, candidate, regression_limit=0.10
        )
        self.assertEqual([entry["case"] for entry in rows], ["faster", "slower"])
        self.assertFalse(rows[0]["regression"])
        self.assertTrue(rows[1]["regression"])
        self.assertTrue(rows[1]["unstable"])
        self.assertIn("2.000x", report)
        self.assertIn("0.800x", report)
        self.assertIn("REGRESSION", report)
        self.assertIn("UNSTABLE", report)

    def test_node_comparison_rejects_runtime_drift(self) -> None:
        baseline = payload([row(median_ns=100.0)], runtime_name="node")
        candidate = payload(
            [row(median_ns=100.0)],
            runtime_name="node",
            runtime_version="v25.0.0",
        )
        with self.assertRaisesRegex(ValueError, "runtime_version"):
            court.compare_node_timings(baseline, candidate, regression_limit=0.10)

    def test_node_baseline_summary_reports_every_row_and_instability(self) -> None:
        stable = row(name="stable", median_ns=100.0)
        unstable = row(name="unstable", family="predicate", median_ns=125.0)
        set_timing_samples(
            unstable,
            [87.5, 100.0, 112.5, 125.0, 137.5, 150.0, 162.5],
        )
        receipt = payload([stable, unstable], runtime_name="node")
        receipt["metadata"]["mode"] = "timed"  # type: ignore[index]

        rows, report = court.summarize_node_receipt(receipt, instability_limit=0.10)

        self.assertEqual([entry["case"] for entry in rows], ["stable", "unstable"])
        self.assertFalse(rows[0]["unstable"])
        self.assertTrue(rows[1]["unstable"])
        self.assertIn("`stable`", report)
        self.assertIn("`unstable`", report)
        self.assertIn("UNSTABLE", report)

    def test_node_baseline_summary_rejects_signature_receipt(self) -> None:
        receipt = payload([row(median_ns=100.0)], runtime_name="node")
        receipt["metadata"]["mode"] = "signatures_only"  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "timed Node receipt"):
            court.summarize_node_receipt(receipt, instability_limit=0.10)

    def test_timing_evidence_rejects_single_sample_smoke(self) -> None:
        receipt = payload([row(median_ns=100.0)], runtime_name="node")
        receipt["metadata"]["samples"] = 1  # type: ignore[index]
        receipt["results"][0]["samples_ns"] = [100.0]  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "at least 5"):
            court.summarize_node_receipt(receipt, instability_limit=0.10)

    def test_timing_evidence_rejects_inconsistent_sample_count(self) -> None:
        receipt = payload([row(median_ns=100.0)], runtime_name="node")
        receipt["results"][0]["samples_ns"] = [100.0] * 6  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "sample-count mismatch"):
            court.summarize_node_receipt(receipt, instability_limit=0.10)

    def test_timing_evidence_rejects_inconsistent_reported_variability(self) -> None:
        receipt = payload([row(median_ns=100.0)], runtime_name="node")
        receipt["results"][0]["relative_standard_deviation"] = 0.25  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "RSD mismatch"):
            court.summarize_node_receipt(receipt, instability_limit=0.10)


if __name__ == "__main__":
    unittest.main()
