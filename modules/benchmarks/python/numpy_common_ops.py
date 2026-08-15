#!/usr/bin/env python3
"""Emit elementwise NumPy results for Ravel's common-ops 0.1 oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np


def encode_number(value: float) -> float | str:
    if np.isnan(value):
        return "nan"
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"
    return float(value)


def encode_values(result: Any) -> tuple[str, str, list[int], list[Any]]:
    array = np.asarray(result)
    if array.shape == () and not isinstance(result, np.ndarray):
        scalar = result.item() if hasattr(result, "item") else result
        if isinstance(scalar, (bool, np.bool_)):
            return "scalar", "scalar", [], [bool(scalar)]
        if isinstance(scalar, (int, np.integer)):
            return "scalar", "scalar", [], [int(scalar)]
        return "scalar", "scalar", [], [encode_number(float(scalar))]

    if array.shape == () and isinstance(result, np.ndarray):
        scalar = result.item()
        if array.dtype == np.bool_ or isinstance(scalar, (bool, np.bool_)):
            return "scalar", "scalar", [], [bool(scalar)]
        if np.issubdtype(array.dtype, np.integer):
            return "scalar", "scalar", [], [int(scalar)]
        return "scalar", "scalar", [], [encode_number(float(scalar))]

    layout = "c_contiguous" if array.flags["C_CONTIGUOUS"] else "strided_view"
    flat = np.ascontiguousarray(array).reshape(-1)
    if array.dtype == np.bool_:
        values: list[Any] = [bool(value) for value in flat]
    elif np.issubdtype(array.dtype, np.integer):
        values = [int(value) for value in flat]
    else:
        values = [encode_number(float(value)) for value in flat]
    return "array", layout, [int(dim) for dim in array.shape], values


class Fixture:
    def __init__(self) -> None:
        rows = np.arange(4, dtype=np.int64)[:, None]
        columns = np.arange(3, dtype=np.int64)[None, :]
        wide_columns = np.arange(6, dtype=np.int64)[None, :]

        self.empty_int = np.zeros((0,), dtype=np.int32)
        self.empty_bool = np.zeros((0,), dtype=bool)
        self.empty_double = np.zeros((0,), dtype=np.float64)

        index = np.arange(7, dtype=np.int64)
        self.int1_left = ((index * 17) % 23 - 11).astype(np.int32)
        self.int1_right = ((index * 7) % 11 - 5).astype(np.int32)

        self.int2_left = ((rows * 17 + columns * 5) % 23 - 11).astype(np.int32)
        self.int2_right = ((rows * 7 + columns * 3) % 11 - 5).astype(np.int32)
        self.int2_positive = ((rows * 3 + columns * 5) % 7 + 1).astype(np.int32)
        self.int_row = ((np.arange(3, dtype=np.int64) * 5) % 7 - 3).astype(np.int32)
        self.int3 = np.fromfunction(
            lambda i, j, k: i * 10 + j * 3 + k, (2, 3, 2), dtype=np.int32
        )
        int_wide = ((rows * 17 + wide_columns * 5) % 23 - 11).astype(np.int32)
        int_wide_right = ((rows * 7 + wide_columns * 3) % 11 - 5).astype(np.int32)
        self.int_inner_left = int_wide[:, 0:6:2]
        self.int_inner_right = int_wide_right[:, 1:6:2]
        self.int_reversed = self.int2_left[:, ::-1]
        self.int_transposed = self.int2_left.T
        self.int2_sign = np.where((rows + columns) % 3 == 0, -1, 1).astype(np.int32)

        self.byte_left = ((rows * 17 + columns * 5) % 23 - 11).astype(np.int8)
        self.byte_right = ((rows * 7 + columns * 3) % 11 - 5).astype(np.int8)

        self.short_left = ((rows * 31 + columns * 7) % 41 - 20).astype(np.int16)
        self.short_right = ((rows * 5 + columns * 11) % 19 - 9).astype(np.int16)

        self.long_left = ((rows * 17 + columns * 5) % 23 - 11).astype(np.int64)
        self.long_right = ((rows * 7 + columns * 3) % 11 - 5).astype(np.int64)
        self.long_positive = ((rows * 3 + columns * 5) % 7 + 1).astype(np.int64)
        self.long_sign = np.where((rows + columns) % 3 == 0, -1, 1).astype(np.int64)

        self.double_sign = np.where((rows + columns) % 3 == 0, -1.0, 1.0).astype(
            np.float64
        )

        self.bool2 = (rows + columns) % 2 == 0
        i, j, k = np.indices((2, 2, 2))
        self.bool3 = (i == 0) | ((j == 1) & (k == 1))

        self.uint8_left = ((rows * 17 + columns * 5) % 23 + 40).astype(np.uint8)
        self.uint8_right = ((rows * 7 + columns * 3) % 11 + 30).astype(np.uint8)
        uint8_wide = ((rows * 17 + wide_columns * 5) % 23 + 40).astype(np.uint8)
        uint8_wide_right = ((rows * 7 + wide_columns * 3) % 11 + 30).astype(np.uint8)
        self.uint8_inner_left = uint8_wide[:, 0:6:2]
        self.uint8_inner_right = uint8_wide_right[:, 1:6:2]

        self.uint16_left = ((rows * 31 + columns * 7) % 41 + 200).astype(np.uint16)
        self.uint16_right = ((rows * 5 + columns * 11) % 19 + 180).astype(np.uint16)

        self.double_left = ((rows * 131 + columns * 17) % 251 - 125).astype(
            np.float64
        ) / 16.0
        self.double_right = ((rows * 43 + columns * 19) % 257 - 128).astype(
            np.float64
        ) / 32.0
        self.double_positive = ((rows * 23 + columns * 11) % 97 + 1).astype(
            np.float64
        ) / 17.0
        self.double_row = ((np.arange(3, dtype=np.int64) * 29) % 127 - 63).astype(
            np.float64
        ) / 8.0
        double_wide = ((rows * 131 + wide_columns * 17) % 251 - 125).astype(
            np.float64
        ) / 16.0
        double_wide_right = ((rows * 43 + wide_columns * 19) % 257 - 128).astype(
            np.float64
        ) / 32.0
        self.double_inner_left = double_wide[:, 0:6:2]
        self.double_inner_right = double_wide_right[:, 1:6:2]
        self.double_reversed = self.double_left[:, ::-1]
        self.double_transposed = self.double_left.T
        self.double3 = np.fromfunction(
            lambda i, j, k: (i * 10 + j + k) / 4.0, (2, 3, 2), dtype=np.float64
        )
        self.double_trunc = np.fromfunction(
            lambda i, j: i + j / 2.0, (4, 3), dtype=np.float64
        )
        self.double_non_finite = self.double_left.copy()
        self.double_non_finite[0, 0] = np.nan
        self.double_non_finite[1, 1] = np.inf
        self.float_left = self.double_left.astype(np.float32)
        self.float_right = self.double_right.astype(np.float32)


def cases(fixture: Fixture) -> list[dict[str, Any]]:
    def row(
        name: str,
        family: str,
        input_dtype: str,
        result_dtype: str,
        input_layout: str,
        comparison: str,
        run: Callable[[], Any],
    ) -> dict[str, Any]:
        kind, layout, shape, values = encode_values(run())
        return {
            "case": name,
            "family": family,
            "input_dtype": input_dtype,
            "result_dtype": result_dtype,
            "input_layout": input_layout,
            "comparison": comparison,
            "result_kind": kind,
            "result_layout": layout,
            "shape": shape,
            "values": values,
        }

    f = fixture
    return [
        row(
            "empty_int_sum",
            "reduction",
            "int32",
            "int32",
            "contiguous",
            "exact",
            lambda: f.empty_int.sum(),
        ),
        row(
            "empty_bool_all",
            "reduction",
            "bool",
            "bool",
            "contiguous",
            "exact",
            lambda: bool(f.empty_bool.all()),
        ),
        row(
            "empty_bool_any",
            "reduction",
            "bool",
            "bool",
            "contiguous",
            "exact",
            lambda: bool(f.empty_bool.any()),
        ),
        row(
            "rank1_int_add",
            "binary",
            "int32",
            "int32",
            "contiguous",
            "exact",
            lambda: np.add(f.int1_left, f.int1_right, order="C"),
        ),
        row(
            "rank2_int_add",
            "binary",
            "int32",
            "int32",
            "contiguous",
            "exact",
            lambda: np.add(f.int2_left, f.int2_right, order="C"),
        ),
        row(
            "rank2_int_multiply",
            "binary",
            "int32",
            "int32",
            "contiguous",
            "exact",
            lambda: np.multiply(f.int2_left, f.int2_right, order="C"),
        ),
        row(
            "rank2_int_quot",
            "binary",
            "int32",
            "int32",
            "contiguous",
            "exact",
            lambda: np.trunc(np.divide(f.int2_left, f.int2_positive)).astype(
                np.int32, order="C"
            ),
        ),
        row(
            "broadcast_int_add",
            "binary",
            "int32",
            "int32",
            "broadcast",
            "exact",
            lambda: np.add(f.int2_left, f.int_row, order="C"),
        ),
        row(
            "rank2_int_sum",
            "reduction",
            "int32",
            "int32",
            "contiguous",
            "exact",
            lambda: int(f.int2_left.sum()),
        ),
        row(
            "rank2_int_sum_axis0",
            "reduction",
            "int32",
            "int32",
            "contiguous",
            "exact",
            lambda: f.int2_left.sum(axis=0),
        ),
        row(
            "rank2_int_sum_keepdims",
            "reduction",
            "int32",
            "int32",
            "contiguous",
            "exact",
            lambda: f.int2_left.sum(axis=1, keepdims=True),
        ),
        row(
            "rank3_int_sum_axis0",
            "reduction",
            "int32",
            "int32",
            "contiguous",
            "exact",
            lambda: f.int3.sum(axis=0),
        ),
        row(
            "rank2_int_min",
            "binary",
            "int32",
            "int32",
            "contiguous",
            "exact",
            lambda: np.minimum(f.int2_left, f.int2_right, order="C"),
        ),
        row(
            "rank2_int_less",
            "comparison",
            "int32",
            "bool",
            "contiguous",
            "exact",
            lambda: np.less(f.int2_left, f.int2_right, order="C"),
        ),
        row(
            "inner_stride_int_add",
            "binary",
            "int32",
            "int32",
            "inner_stride",
            "exact",
            lambda: np.add(f.int_inner_left, f.int_inner_right, order="C"),
        ),
        row(
            "reverse_int_maximum",
            "binary",
            "int32",
            "int32",
            "reversed",
            "exact",
            lambda: np.maximum(f.int_reversed, f.int2_right, order="C"),
        ),
        row(
            "transpose_int_equal",
            "comparison",
            "int32",
            "bool",
            "transposed",
            "exact",
            lambda: np.equal(f.int_transposed, f.int2_right.T, order="C"),
        ),
        row(
            "bool_all",
            "reduction",
            "bool",
            "bool",
            "contiguous",
            "exact",
            lambda: bool(f.bool2.all()),
        ),
        row(
            "bool_any",
            "reduction",
            "bool",
            "bool",
            "contiguous",
            "exact",
            lambda: bool(f.bool2.any()),
        ),
        row(
            "bool_all_axis0",
            "reduction",
            "bool",
            "bool",
            "contiguous",
            "exact",
            lambda: f.bool2.all(axis=0),
        ),
        row(
            "bool_any_keepdims",
            "reduction",
            "bool",
            "bool",
            "contiguous",
            "exact",
            lambda: f.bool2.any(axis=1, keepdims=True),
        ),
        row(
            "bool_count_true",
            "reduction",
            "bool",
            "int32",
            "contiguous",
            "exact",
            lambda: int(f.bool2.sum()),
        ),
        row(
            "rank3_bool_all_axes",
            "reduction",
            "bool",
            "bool",
            "contiguous",
            "exact",
            lambda: f.bool3.all(axis=(0, 1)),
        ),
        row(
            "uint8_minimum",
            "binary",
            "uint8",
            "uint8",
            "contiguous",
            "exact",
            lambda: np.minimum(f.uint8_left, f.uint8_right, order="C"),
        ),
        row(
            "uint8_clip",
            "unary",
            "uint8",
            "uint8",
            "contiguous",
            "exact",
            lambda: np.clip(f.uint8_left, 40, 80),
        ),
        row(
            "uint8_greater_equal",
            "comparison",
            "uint8",
            "bool",
            "contiguous",
            "exact",
            lambda: np.greater_equal(f.uint8_left, np.uint8(50), order="C"),
        ),
        row(
            "uint8_max_axis1",
            "reduction",
            "uint8",
            "uint8",
            "contiguous",
            "exact",
            lambda: f.uint8_left.max(axis=1),
        ),
        row(
            "inner_stride_uint8_min",
            "binary",
            "uint8",
            "uint8",
            "inner_stride",
            "exact",
            lambda: np.minimum(f.uint8_inner_left, f.uint8_inner_right, order="C"),
        ),
        row(
            "uint8_cast_float32",
            "cast",
            "uint8",
            "float32",
            "contiguous",
            "exact",
            lambda: f.uint8_left.astype(np.float32, order="C"),
        ),
        row(
            "uint16_maximum",
            "binary",
            "uint16",
            "uint16",
            "contiguous",
            "exact",
            lambda: np.maximum(f.uint16_left, f.uint16_right, order="C"),
        ),
        row(
            "uint16_less",
            "comparison",
            "uint16",
            "bool",
            "contiguous",
            "exact",
            lambda: np.less(f.uint16_left, f.uint16_right, order="C"),
        ),
        row(
            "uint16_cast_int32",
            "cast",
            "uint16",
            "int32",
            "contiguous",
            "exact",
            lambda: f.uint16_left.astype(np.int32, order="C"),
        ),
        row(
            "cast_float64_int32",
            "cast",
            "float64",
            "int32",
            "contiguous",
            "exact",
            lambda: f.double_trunc.astype(np.int32, order="C"),
        ),
        row(
            "rank2_double_add",
            "binary",
            "float64",
            "float64",
            "contiguous",
            "floating",
            lambda: np.add(f.double_left, f.double_right, order="C"),
        ),
        row(
            "broadcast_double_subtract",
            "binary",
            "float64",
            "float64",
            "broadcast",
            "floating",
            lambda: np.subtract(f.double_left, f.double_row, order="C"),
        ),
        row(
            "inner_stride_double_multiply",
            "binary",
            "float64",
            "float64",
            "inner_stride",
            "floating",
            lambda: np.multiply(f.double_inner_left, f.double_inner_right, order="C"),
        ),
        row(
            "reverse_double_divide",
            "binary",
            "float64",
            "float64",
            "reversed",
            "floating",
            lambda: np.divide(f.double_reversed, f.double_positive, order="C"),
        ),
        row(
            "transpose_double_minimum",
            "binary",
            "float64",
            "float64",
            "transposed",
            "floating",
            lambda: np.minimum(f.double_transposed, f.double_right.T, order="C"),
        ),
        row(
            "double_abs",
            "unary",
            "float64",
            "float64",
            "contiguous",
            "floating",
            lambda: np.absolute(f.double_left, order="C"),
        ),
        row(
            "double_clip",
            "unary",
            "float64",
            "float64",
            "contiguous",
            "floating",
            lambda: np.clip(f.double_left, -2.0, 3.0),
        ),
        row(
            "double_less_equal",
            "comparison",
            "float64",
            "bool",
            "contiguous",
            "exact",
            lambda: np.less_equal(f.double_left, f.double_right, order="C"),
        ),
        row(
            "double_is_nan",
            "predicate",
            "float64",
            "bool",
            "contiguous",
            "exact",
            lambda: np.isnan(f.double_non_finite, order="C"),
        ),
        row(
            "double_is_finite",
            "predicate",
            "float64",
            "bool",
            "contiguous",
            "exact",
            lambda: np.isfinite(f.double_non_finite, order="C"),
        ),
        row(
            "double_mean",
            "reduction",
            "float64",
            "float64",
            "contiguous",
            "floating",
            lambda: float(f.double_left.mean()),
        ),
        row(
            "double_mean_axis0",
            "reduction",
            "float64",
            "float64",
            "contiguous",
            "floating",
            lambda: f.double_left.mean(axis=0),
        ),
        row(
            "empty_float_mean",
            "reduction",
            "float64",
            "float64",
            "contiguous",
            "floating",
            lambda: float("nan"),
        ),
        row(
            "rank3_double_mean_axis2",
            "reduction",
            "float64",
            "float64",
            "contiguous",
            "floating",
            lambda: f.double3.mean(axis=2),
        ),
        row(
            "rank2_float32_add",
            "binary",
            "float32",
            "float32",
            "contiguous",
            "floating",
            lambda: np.add(f.float_left, f.float_right, order="C"),
        ),
        row(
            "scalar_add_int",
            "scalar",
            "int32",
            "int32",
            "contiguous",
            "exact",
            lambda: np.add(f.int2_left, np.int32(1), order="C"),
        ),
        row(
            "scalar_subtract_int",
            "scalar",
            "int32",
            "int32",
            "contiguous",
            "exact",
            lambda: np.subtract(f.int2_left, np.int32(2), order="C"),
        ),
        row(
            "empty_int_product",
            "reduction",
            "int32",
            "int32",
            "contiguous",
            "exact",
            lambda: int(f.empty_int.prod()),
        ),
        row(
            "rank2_int_product",
            "reduction",
            "int32",
            "int32",
            "contiguous",
            "exact",
            lambda: int(f.int2_sign.prod()),
        ),
        row(
            "rank2_int_arg_min",
            "reduction",
            "int32",
            "int32",
            "contiguous",
            "exact",
            lambda: int(f.int2_left.argmin()),
        ),
        row(
            "rank2_int_arg_max",
            "reduction",
            "int32",
            "int32",
            "contiguous",
            "exact",
            lambda: int(f.int2_left.argmax()),
        ),
        row(
            "rank2_int_arg_min_axis0",
            "reduction",
            "int32",
            "int32",
            "contiguous",
            "exact",
            lambda: f.int2_left.argmin(axis=0),
        ),
        row(
            "rank2_int_sum_as_long",
            "reduction",
            "int32",
            "int64",
            "contiguous",
            "exact",
            lambda: int(f.int2_left.astype(np.int64).sum()),
        ),
        row(
            "byte_minimum",
            "binary",
            "int8",
            "int8",
            "contiguous",
            "exact",
            lambda: np.minimum(f.byte_left, f.byte_right, order="C"),
        ),
        row(
            "byte_clip",
            "unary",
            "int8",
            "int8",
            "contiguous",
            "exact",
            lambda: np.clip(f.byte_left, np.int8(-4), np.int8(6)).astype(
                np.int8, order="C"
            ),
        ),
        row(
            "byte_less",
            "comparison",
            "int8",
            "bool",
            "contiguous",
            "exact",
            lambda: np.less(f.byte_left, f.byte_right, order="C"),
        ),
        row(
            "byte_min",
            "reduction",
            "int8",
            "int8",
            "contiguous",
            "exact",
            lambda: int(f.byte_left.min()),
        ),
        row(
            "byte_arg_max",
            "reduction",
            "int8",
            "int32",
            "contiguous",
            "exact",
            lambda: int(f.byte_left.argmax()),
        ),
        row(
            "byte_cast_int32",
            "cast",
            "int8",
            "int32",
            "contiguous",
            "exact",
            lambda: f.byte_left.astype(np.int32, order="C"),
        ),
        row(
            "short_maximum",
            "binary",
            "int16",
            "int16",
            "contiguous",
            "exact",
            lambda: np.maximum(f.short_left, f.short_right, order="C"),
        ),
        row(
            "short_greater_equal",
            "comparison",
            "int16",
            "bool",
            "contiguous",
            "exact",
            lambda: np.greater_equal(f.short_left, np.int16(0), order="C"),
        ),
        row(
            "short_arg_min",
            "reduction",
            "int16",
            "int32",
            "contiguous",
            "exact",
            lambda: int(f.short_left.argmin()),
        ),
        row(
            "short_max_axis0",
            "reduction",
            "int16",
            "int16",
            "contiguous",
            "exact",
            lambda: f.short_left.max(axis=0),
        ),
        row(
            "short_cast_float64",
            "cast",
            "int16",
            "float64",
            "contiguous",
            "exact",
            lambda: f.short_left.astype(np.float64, order="C"),
        ),
        row(
            "rank2_long_add",
            "binary",
            "int64",
            "int64",
            "contiguous",
            "exact",
            lambda: np.add(f.long_left, f.long_right, order="C"),
        ),
        row(
            "rank2_long_multiply",
            "binary",
            "int64",
            "int64",
            "contiguous",
            "exact",
            lambda: np.multiply(f.long_left, f.long_right, order="C"),
        ),
        row(
            "scalar_add_long",
            "scalar",
            "int64",
            "int64",
            "contiguous",
            "exact",
            lambda: np.add(f.long_left, np.int64(1), order="C"),
        ),
        row(
            "rank2_long_quot",
            "binary",
            "int64",
            "int64",
            "contiguous",
            "exact",
            lambda: np.trunc(np.divide(f.long_left, f.long_positive)).astype(
                np.int64, order="C"
            ),
        ),
        row(
            "rank2_long_sum",
            "reduction",
            "int64",
            "int64",
            "contiguous",
            "exact",
            lambda: int(f.long_left.sum()),
        ),
        row(
            "rank2_long_product",
            "reduction",
            "int64",
            "int64",
            "contiguous",
            "exact",
            lambda: int(f.long_sign.prod()),
        ),
        row(
            "long_arg_min_axis1",
            "reduction",
            "int64",
            "int32",
            "contiguous",
            "exact",
            lambda: f.long_left.argmin(axis=1).astype(np.int32, order="C"),
        ),
        row(
            "scalar_add_double",
            "scalar",
            "float64",
            "float64",
            "contiguous",
            "exact",
            lambda: np.add(f.double_left, np.float64(0.25), order="C"),
        ),
        row(
            "rank2_double_product",
            "reduction",
            "float64",
            "float64",
            "contiguous",
            "exact",
            lambda: float(f.double_sign.prod()),
        ),
        row(
            "scalar_divide_float",
            "scalar",
            "float32",
            "float32",
            "contiguous",
            "floating",
            lambda: np.divide(f.float_left, np.float32(2.0), order="C"),
        ),
        row(
            "float_sum_as_double",
            "reduction",
            "float32",
            "float64",
            "contiguous",
            "floating",
            lambda: float(f.float_left.astype(np.float64).sum()),
        ),
    ]


def run_suite() -> dict[str, Any]:
    rows = cases(Fixture())
    names = [row["case"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("duplicate common-ops case name")
    return {
        "schema": "ravel.common-ops.signatures.v1",
        "metadata": {
            "library": "numpy",
            "oracle": "elementwise",
            "release_target": "0.1",
            "numpy_version": np.__version__,
        },
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("target/common-ops/numpy-signatures.json"),
    )
    args = parser.parse_args()
    payload = run_suite()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(payload['results'])} common-ops signatures to {args.out}")


if __name__ == "__main__":
    main()
