from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import compare_common_ops as comparison
import numpy_common_ops as numpy_suite


class CommonOpsTest(unittest.TestCase):
    def test_registry_covers_required_dimensions(self) -> None:
        rows = numpy_suite.cases(numpy_suite.Fixture())
        names = {row["case"] for row in rows}
        self.assertEqual(len(names), len(rows))
        self.assertGreaterEqual(len(rows), 40)
        self.assertEqual(
            {row["input_dtype"] for row in rows},
            {
                "bool",
                "uint8",
                "uint16",
                "int8",
                "int16",
                "int32",
                "int64",
                "float32",
                "float64",
            },
        )
        self.assertTrue(
            {
                "empty_int_sum",
                "empty_bool_all",
                "rank1_int_add",
                "rank3_int_sum_axis0",
                "rank2_int_sum_keepdims",
                "inner_stride_uint8_min",
                "uint8_cast_float32",
                "uint16_less",
                "bool_any_keepdims",
                "double_is_nan",
                "empty_float_mean",
                "transpose_double_minimum",
                "scalar_add_int",
                "rank2_int_product",
                "rank2_int_arg_min",
                "rank2_int_sum_as_long",
                "byte_minimum",
                "short_arg_min",
                "rank2_long_add",
                "float_sum_as_double",
                "byte_convert_int32",
                "double_convert_int32_nearest_even",
                "double_convert_int32_floor",
                "int32_convert_int8_reject",
                "double_convert_int32_nan_reject",
                "reversed_int32_convert_int8_reject",
            }
            <= names
        )

    def test_self_compare_passes(self) -> None:
        payload = numpy_suite.run_suite()
        comparison.validate_signatures(
            comparison.index_results(payload),
            comparison.index_results(payload),
        )

    def test_value_mismatch_is_reported(self) -> None:
        payload = numpy_suite.run_suite()
        broken = comparison.index_results(payload)
        first = next(iter(broken))
        broken[first] = dict(broken[first])
        values = list(broken[first]["values"])
        if not values:
            self.skipTest("unexpected empty values")
        values[0] = values[0] + 1 if isinstance(values[0], int) else not values[0]
        broken[first]["values"] = values
        with self.assertRaises(ValueError):
            comparison.validate_signatures(
                comparison.index_results(payload),
                broken,
            )


if __name__ == "__main__":
    unittest.main()
