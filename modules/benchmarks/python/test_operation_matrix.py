from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import compare_operation_matrix as comparison
import numpy_operation_matrix as numpy_suite


class OperationMatrixTest(unittest.TestCase):
    def test_registry_covers_required_dimensions(self) -> None:
        cases = numpy_suite.benchmark_cases(numpy_suite.Fixture(4))
        self.assertEqual(len(cases), 79)
        self.assertEqual(len({case.name for case in cases}), len(cases))
        self.assertEqual(
            {case.input_dtype for case in cases},
            {"float64", "float32", "int32", "int64"},
        )
        self.assertEqual(
            {case.input_layout for case in cases},
            {
                "contiguous",
                "inner_stride",
                "outer_stride",
                "reversed",
                "transposed",
                "broadcast",
            },
        )
        self.assertEqual(
            {case.family for case in cases},
            {
                "binary",
                "scalar",
                "unary",
                "comparison",
                "predicate",
                "cast",
                "reduction",
                "in-place",
            },
        )
        names = {case.name for case in cases}
        required = {
            "contiguous_subtract_double",
            "inner_stride_multiply_double",
            "outer_stride_divide_double",
            "scalar_add_double",
            "clip_double",
            "tan_double",
            "greater_equal_scalar_double",
            "is_finite_double",
            "cast_long_float",
            "full_product_double",
            "full_arg_min_double",
            "axis0_mean_double",
            "axis1_arg_max_double",
            "sum_as_long_int",
            "sum_as_double_float",
            "inplace_add_int",
            "inplace_inner_stride_multiply_double",
        }
        self.assertTrue(required <= names)

    def test_signatures_only_emits_metadata_and_no_timings(self) -> None:
        payload = numpy_suite.run_suite(
            [4],
            warmups=0,
            samples=1,
            target_ms=1.0,
            signatures_only=True,
        )
        self.assertEqual(payload["metadata"]["mode"], "signatures_only")
        self.assertEqual(len(payload["results"]), 79)
        for row in payload["results"]:
            self.assertNotIn("median_ns", row)
            for field in comparison.METADATA_FIELDS:
                self.assertIn(field, row)

    def test_signature_validation_uses_exact_and_floating_contracts(self) -> None:
        exact = {
            "family": "reduction",
            "input_dtype": "int32",
            "result_dtype": "int64",
            "input_layout": "contiguous",
            "logical_work_units": 16,
            "work_unit": "element",
            "comparison": "exact",
            "result_size": 1,
            "result_layout": "scalar",
            "sum": 12.0,
            "weighted_sum": 12.0,
        }
        floating = {
            **exact,
            "input_dtype": "float32",
            "result_dtype": "float32",
            "comparison": "floating",
            "sum": 1.0,
            "weighted_sum": 1.0,
        }
        comparison.validate_signatures(
            {
                ("exact", 4): exact,
                ("floating", 4): floating,
            },
            {
                ("exact", 4): dict(exact),
                ("floating", 4): {
                    **floating,
                    "sum": 1.0 + 1.0e-7,
                    "weighted_sum": 1.0 - 1.0e-7,
                },
            },
        )
        with self.assertRaisesRegex(ValueError, "sum"):
            comparison.validate_signatures(
                {("exact", 4): exact},
                {("exact", 4): {**exact, "sum": 13.0}},
            )

    def test_signature_validation_rejects_matrix_metadata_mismatch(self) -> None:
        row = {
            "family": "binary",
            "input_dtype": "float64",
            "result_dtype": "float64",
            "input_layout": "contiguous",
            "logical_work_units": 16,
            "work_unit": "element",
            "comparison": "floating",
            "result_size": 16,
            "result_layout": "c_contiguous",
            "sum": 4.0,
            "weighted_sum": 12.0,
        }
        with self.assertRaisesRegex(ValueError, "input_layout"):
            comparison.validate_signatures(
                {("case", 4): row},
                {("case", 4): {**row, "input_layout": "reversed"}},
            )

    def test_jmh_reader_uses_case_parameter(self) -> None:
        payload = [
            {
                "benchmark": "ravel.bench.OperationMatrixBenchmarks.operation_matrix",
                "params": {"caseName": "clip_double", "side": "256"},
                "primaryMetric": {
                    "score": 2.0,
                    "scoreError": 0.1,
                    "scoreUnit": "us/op",
                    "scorePercentiles": {"50.0": 1.5},
                },
            }
        ]
        rows = comparison.read_jmh(payload)
        self.assertEqual(rows[("clip_double", 256)]["ravel_ns"], 1_500.0)


if __name__ == "__main__":
    unittest.main()
