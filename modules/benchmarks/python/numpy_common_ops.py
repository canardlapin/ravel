#!/usr/bin/env python3
"""Emit elementwise NumPy results for Ravel's common-ops 0.1 oracle."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np

RAVEL_DTYPE_NAMES = {
    "int8": "Byte",
    "uint8": "UInt8",
    "int16": "Short",
    "uint16": "UInt16",
    "int32": "Int",
    "int64": "Long",
    "float32": "Float",
    "float64": "Double",
}

INTEGRAL_RANGES = {
    "int8": (-128, 127),
    "uint8": (0, 255),
    "int16": (-32768, 32767),
    "uint16": (0, 65535),
    "int32": (-2147483648, 2147483647),
    "int64": (-9223372036854775808, 9223372036854775807),
}

FLOAT32_MAX = float(np.finfo(np.float32).max)


def encode_number(value: float) -> float | str:
    if np.isnan(value):
        return "nan"
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"
    return float(value)


def encode_values(result: Any) -> tuple[str, str, list[int], list[Any]]:
    if isinstance(result, dict) and result.get("kind") == "error":
        return "error", "none", [], list(result["values"])
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


def numpy_dtype_name(dtype: np.dtype) -> str:
    if dtype == np.bool_:
        return "bool"
    return np.dtype(dtype).name


def round_floating(values: np.ndarray, rounding: str) -> np.ndarray:
    if rounding == "nearest_even":
        return np.rint(values)
    if rounding == "toward_zero":
        return np.trunc(values)
    if rounding == "floor":
        return np.floor(values)
    if rounding == "ceiling":
        return np.ceil(values)
    raise ValueError(f"unsupported rounding: {rounding}")


def ravel_convert(
    source: np.ndarray,
    target_dtype: str,
    rounding: str = "nearest_even",
) -> Any:
    """Mirror Ravel convert(..., Overflow.Reject) against a NumPy fixture."""
    array = np.asarray(source)
    source_name = RAVEL_DTYPE_NAMES[numpy_dtype_name(array.dtype)]
    target_name = RAVEL_DTYPE_NAMES[target_dtype]
    flat = np.ascontiguousarray(array).reshape(-1)
    source_integral = np.issubdtype(array.dtype, np.integer)
    target_integral = target_dtype in INTEGRAL_RANGES

    for logical, value in enumerate(flat):
        if target_integral:
            if source_integral:
                candidate = int(value)
            else:
                number = float(value)
                if not np.isfinite(number):
                    return {
                        "kind": "error",
                        "values": ["non_finite", logical, source_name, target_name],
                    }
                candidate = int(round_floating(np.array(number), rounding).item())
            low, high = INTEGRAL_RANGES[target_dtype]
            if candidate < low or candidate > high:
                return {
                    "kind": "error",
                    "values": ["out_of_range", logical, source_name, target_name],
                }
        elif (
            target_dtype == "float32"
            and not source_integral
            and np.isfinite(float(value))
            and (float(value) > FLOAT32_MAX or float(value) < -FLOAT32_MAX)
        ):
            return {
                "kind": "error",
                "values": ["out_of_range", logical, source_name, target_name],
            }

    if target_integral and not source_integral:
        return np.ascontiguousarray(
            round_floating(array, rounding).astype(target_dtype, order="C")
        )
    return np.ascontiguousarray(array.astype(target_dtype, order="C"))


def empty_reduction_error(operation: str) -> dict[str, Any]:
    return {"kind": "error", "values": ["empty_reduction", operation]}


def ravel_full_reduce(array: np.ndarray, operation: str) -> Any:
    values = np.asarray(array)
    if operation in {"min", "max", "argMin", "argMax"} and values.size == 0:
        return empty_reduction_error(operation)
    if operation == "min":
        return values.min()
    if operation == "max":
        return values.max()
    if operation == "argMin":
        return int(values.argmin())
    if operation == "argMax":
        return int(values.argmax())
    raise ValueError(f"unsupported full reduction: {operation}")


def ravel_axis_reduce(
    array: np.ndarray,
    operation: str,
    axis: int,
    keepdims: bool = False,
) -> Any:
    values = np.asarray(array)
    rank = values.ndim
    normalized = axis if axis >= 0 else axis + rank
    reduced_length = int(values.shape[normalized])
    output_size = int(
        np.prod(
            [dim for index, dim in enumerate(values.shape) if index != normalized],
            initial=1,
        )
    )
    if (
        operation in {"min", "max", "argMin", "argMax"}
        and reduced_length == 0
        and output_size > 0
    ):
        return empty_reduction_error(operation)

    kwargs: dict[str, Any] = {"axis": axis, "keepdims": keepdims}
    if operation == "sum":
        return values.sum(**kwargs)
    if operation == "product":
        return values.prod(**kwargs)
    if operation == "mean":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with np.errstate(invalid="ignore", divide="ignore"):
                return values.mean(**kwargs)
    if operation == "min":
        return values.min(**kwargs)
    if operation == "max":
        return values.max(**kwargs)
    if operation == "argMin":
        return values.argmin(**kwargs).astype(np.int32, order="C")
    if operation == "argMax":
        return values.argmax(**kwargs).astype(np.int32, order="C")
    if operation == "all":
        return values.all(**kwargs)
    if operation == "any":
        return values.any(**kwargs)
    raise ValueError(f"unsupported axis reduction: {operation}")


class Fixture:
    def __init__(self) -> None:
        rows = np.arange(4, dtype=np.int64)[:, None]
        columns = np.arange(3, dtype=np.int64)[None, :]
        wide_columns = np.arange(6, dtype=np.int64)[None, :]

        self.empty_int = np.zeros((0,), dtype=np.int32)
        self.empty_bool = np.zeros((0,), dtype=bool)
        self.empty_double = np.zeros((0,), dtype=np.float64)
        self.empty_fiber_int = np.zeros((3, 0), dtype=np.int32)
        self.empty_fiber_double = np.zeros((3, 0), dtype=np.float64)
        self.empty_fiber_rank3 = np.zeros((2, 0, 3), dtype=np.int32)
        self.empty_rows_int = np.zeros((0, 5), dtype=np.int32)
        self.empty_fiber_bool = np.zeros((2, 0, 3), dtype=bool)

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

        self.convert_round = np.array([2.5, 3.5, -2.5, -3.5], dtype=np.float64)
        self.convert_toward = np.array([1.9, -1.9], dtype=np.float64)
        self.convert_inrange_uint8 = np.array([0, 128, 255], dtype=np.int32)
        self.convert_inrange_uint8_float = np.array(
            [0.4, 200.4, 255.0], dtype=np.float64
        )
        self.convert_wide_int = np.array([127, 128, 0], dtype=np.int32)
        self.convert_neg_int = np.array([1, -1, 2], dtype=np.int32)
        self.convert_wide_uint16 = np.array([200, 300], dtype=np.uint16)
        self.convert_wide_long = np.array([1, np.int64(2**31)], dtype=np.int64)
        self.convert_overflow_byte = np.array([1.0, 128.0, 2.0], dtype=np.float64)
        self.convert_nan = np.array([1.0, np.nan, 3.0], dtype=np.float64)
        self.convert_inf = np.array([1.0, np.inf], dtype=np.float64)
        self.convert_huge_double = np.array([1.0, 1.0e40], dtype=np.float64)
        self.convert_overflow_float = np.array([1.0, 200.0], dtype=np.float32)
        self.convert_int_overflow = np.array([[1, 2, 200], [3, 4, 5]], dtype=np.int32)
        self.convert_int_overflow_reversed = self.convert_int_overflow[:, ::-1]


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
        row(
            "byte_convert_int32",
            "convert",
            "int8",
            "int32",
            "contiguous",
            "exact",
            lambda: ravel_convert(f.byte_left, "int32"),
        ),
        row(
            "byte_convert_int64",
            "convert",
            "int8",
            "int64",
            "contiguous",
            "exact",
            lambda: ravel_convert(f.byte_left, "int64"),
        ),
        row(
            "uint8_convert_int32",
            "convert",
            "uint8",
            "int32",
            "contiguous",
            "exact",
            lambda: ravel_convert(f.uint8_left, "int32"),
        ),
        row(
            "uint16_convert_int64",
            "convert",
            "uint16",
            "int64",
            "contiguous",
            "exact",
            lambda: ravel_convert(f.uint16_left, "int64"),
        ),
        row(
            "int32_convert_int64",
            "convert",
            "int32",
            "int64",
            "contiguous",
            "exact",
            lambda: ravel_convert(f.int2_left, "int64"),
        ),
        row(
            "int32_convert_float64",
            "convert",
            "int32",
            "float64",
            "contiguous",
            "exact",
            lambda: ravel_convert(f.int2_left, "float64"),
        ),
        row(
            "uint8_convert_float32",
            "convert",
            "uint8",
            "float32",
            "contiguous",
            "exact",
            lambda: ravel_convert(f.uint8_left, "float32"),
        ),
        row(
            "short_convert_int8",
            "convert",
            "int16",
            "int8",
            "contiguous",
            "exact",
            lambda: ravel_convert(f.short_left, "int8"),
        ),
        row(
            "uint8_convert_int8",
            "convert",
            "uint8",
            "int8",
            "contiguous",
            "exact",
            lambda: ravel_convert(f.uint8_left, "int8"),
        ),
        row(
            "uint16_convert_uint8",
            "convert",
            "uint16",
            "uint8",
            "contiguous",
            "exact",
            lambda: ravel_convert(f.uint16_left, "uint8"),
        ),
        row(
            "int32_convert_uint8_inrange",
            "convert",
            "int32",
            "uint8",
            "contiguous",
            "exact",
            lambda: ravel_convert(f.convert_inrange_uint8, "uint8"),
        ),
        row(
            "long_convert_int32",
            "convert",
            "int64",
            "int32",
            "contiguous",
            "exact",
            lambda: ravel_convert(f.long_left, "int32"),
        ),
        row(
            "float32_convert_float64",
            "convert",
            "float32",
            "float64",
            "contiguous",
            "exact",
            lambda: ravel_convert(f.float_left, "float64"),
        ),
        row(
            "float64_convert_float32_inrange",
            "convert",
            "float64",
            "float32",
            "contiguous",
            "exact",
            lambda: ravel_convert(f.double_left, "float32"),
        ),
        row(
            "reversed_int32_convert_int8",
            "convert",
            "int32",
            "int8",
            "reversed",
            "exact",
            lambda: ravel_convert(f.int_reversed, "int8"),
        ),
        row(
            "double_convert_int32_nearest_even",
            "convert",
            "float64",
            "int32",
            "contiguous",
            "exact",
            lambda: ravel_convert(f.convert_round, "int32"),
        ),
        row(
            "double_convert_int32_toward_zero",
            "convert",
            "float64",
            "int32",
            "contiguous",
            "exact",
            lambda: ravel_convert(f.convert_toward, "int32", rounding="toward_zero"),
        ),
        row(
            "double_convert_int32_floor",
            "convert",
            "float64",
            "int32",
            "contiguous",
            "exact",
            lambda: ravel_convert(f.convert_toward, "int32", rounding="floor"),
        ),
        row(
            "double_convert_int32_ceiling",
            "convert",
            "float64",
            "int32",
            "contiguous",
            "exact",
            lambda: ravel_convert(f.convert_toward, "int32", rounding="ceiling"),
        ),
        row(
            "double_convert_uint8_inrange",
            "convert",
            "float64",
            "uint8",
            "contiguous",
            "exact",
            lambda: ravel_convert(f.convert_inrange_uint8_float, "uint8"),
        ),
        row(
            "int32_convert_int8_reject",
            "convert",
            "int32",
            "int8",
            "contiguous",
            "exact",
            lambda: ravel_convert(f.convert_wide_int, "int8"),
        ),
        row(
            "int32_convert_uint8_reject",
            "convert",
            "int32",
            "uint8",
            "contiguous",
            "exact",
            lambda: ravel_convert(f.convert_neg_int, "uint8"),
        ),
        row(
            "uint16_convert_uint8_reject",
            "convert",
            "uint16",
            "uint8",
            "contiguous",
            "exact",
            lambda: ravel_convert(f.convert_wide_uint16, "uint8"),
        ),
        row(
            "long_convert_int32_reject",
            "convert",
            "int64",
            "int32",
            "contiguous",
            "exact",
            lambda: ravel_convert(f.convert_wide_long, "int32"),
        ),
        row(
            "double_convert_int8_reject",
            "convert",
            "float64",
            "int8",
            "contiguous",
            "exact",
            lambda: ravel_convert(f.convert_overflow_byte, "int8"),
        ),
        row(
            "double_convert_int32_nan_reject",
            "convert",
            "float64",
            "int32",
            "contiguous",
            "exact",
            lambda: ravel_convert(f.convert_nan, "int32"),
        ),
        row(
            "double_convert_int32_inf_reject",
            "convert",
            "float64",
            "int32",
            "contiguous",
            "exact",
            lambda: ravel_convert(f.convert_inf, "int32"),
        ),
        row(
            "double_convert_float32_reject",
            "convert",
            "float64",
            "float32",
            "contiguous",
            "exact",
            lambda: ravel_convert(f.convert_huge_double, "float32"),
        ),
        row(
            "float32_convert_int8_reject",
            "convert",
            "float32",
            "int8",
            "contiguous",
            "exact",
            lambda: ravel_convert(f.convert_overflow_float, "int8"),
        ),
        row(
            "reversed_int32_convert_int8_reject",
            "convert",
            "int32",
            "int8",
            "reversed",
            "exact",
            lambda: ravel_convert(f.convert_int_overflow_reversed, "int8"),
        ),
        row(
            "empty_int_min_reject",
            "reduction",
            "int32",
            "int32",
            "contiguous",
            "exact",
            lambda: ravel_full_reduce(f.empty_int, "min"),
        ),
        row(
            "empty_int_max_reject",
            "reduction",
            "int32",
            "int32",
            "contiguous",
            "exact",
            lambda: ravel_full_reduce(f.empty_int, "max"),
        ),
        row(
            "empty_int_arg_min_reject",
            "reduction",
            "int32",
            "int32",
            "contiguous",
            "exact",
            lambda: ravel_full_reduce(f.empty_int, "argMin"),
        ),
        row(
            "empty_int_arg_max_reject",
            "reduction",
            "int32",
            "int32",
            "contiguous",
            "exact",
            lambda: ravel_full_reduce(f.empty_int, "argMax"),
        ),
        row(
            "empty_fiber_int_sum",
            "reduction",
            "int32",
            "int32",
            "contiguous",
            "exact",
            lambda: ravel_axis_reduce(f.empty_fiber_int, "sum", 1),
        ),
        row(
            "empty_fiber_int_product",
            "reduction",
            "int32",
            "int32",
            "contiguous",
            "exact",
            lambda: ravel_axis_reduce(f.empty_fiber_int, "product", 1),
        ),
        row(
            "empty_fiber_int_min_reject",
            "reduction",
            "int32",
            "int32",
            "contiguous",
            "exact",
            lambda: ravel_axis_reduce(f.empty_fiber_int, "min", 1),
        ),
        row(
            "empty_fiber_int_arg_max_reject",
            "reduction",
            "int32",
            "int32",
            "contiguous",
            "exact",
            lambda: ravel_axis_reduce(f.empty_fiber_int, "argMax", 1),
        ),
        row(
            "empty_fiber_double_mean",
            "reduction",
            "float64",
            "float64",
            "contiguous",
            "floating",
            lambda: ravel_axis_reduce(f.empty_fiber_double, "mean", 1),
        ),
        row(
            "empty_fiber_rank3_sum",
            "reduction",
            "int32",
            "int32",
            "contiguous",
            "exact",
            lambda: ravel_axis_reduce(f.empty_fiber_rank3, "sum", 1),
        ),
        row(
            "empty_fiber_rank3_min_reject",
            "reduction",
            "int32",
            "int32",
            "contiguous",
            "exact",
            lambda: ravel_axis_reduce(f.empty_fiber_rank3, "min", 1),
        ),
        row(
            "empty_output_int_sum_axis0",
            "reduction",
            "int32",
            "int32",
            "contiguous",
            "exact",
            lambda: ravel_axis_reduce(f.empty_rows_int, "sum", 0),
        ),
        row(
            "empty_output_int_sum_axis1",
            "reduction",
            "int32",
            "int32",
            "contiguous",
            "exact",
            lambda: ravel_axis_reduce(f.empty_rows_int, "sum", 1),
        ),
        row(
            "empty_output_int_min_axis1",
            "reduction",
            "int32",
            "int32",
            "contiguous",
            "exact",
            lambda: ravel_axis_reduce(f.empty_rows_int, "min", 1),
        ),
        row(
            "empty_output_rank3_min_axis0",
            "reduction",
            "int32",
            "int32",
            "contiguous",
            "exact",
            lambda: ravel_axis_reduce(f.empty_fiber_rank3, "min", 0),
        ),
        row(
            "empty_fiber_bool_all",
            "reduction",
            "bool",
            "bool",
            "contiguous",
            "exact",
            lambda: ravel_axis_reduce(f.empty_fiber_bool, "all", 1),
        ),
        row(
            "empty_fiber_bool_any",
            "reduction",
            "bool",
            "bool",
            "contiguous",
            "exact",
            lambda: ravel_axis_reduce(f.empty_fiber_bool, "any", 1),
        ),
        row(
            "rank2_int_sum_axis_neg1",
            "reduction",
            "int32",
            "int32",
            "contiguous",
            "exact",
            lambda: ravel_axis_reduce(f.int2_left, "sum", -1),
        ),
        row(
            "rank2_int_sum_axis_neg2",
            "reduction",
            "int32",
            "int32",
            "contiguous",
            "exact",
            lambda: ravel_axis_reduce(f.int2_left, "sum", -2),
        ),
        row(
            "rank2_int_product_axis_neg1",
            "reduction",
            "int32",
            "int32",
            "contiguous",
            "exact",
            lambda: ravel_axis_reduce(f.int2_sign, "product", -1),
        ),
        row(
            "rank2_int_min_axis_neg1",
            "reduction",
            "int32",
            "int32",
            "contiguous",
            "exact",
            lambda: ravel_axis_reduce(f.int2_left, "min", -1),
        ),
        row(
            "rank2_int_max_axis_neg1",
            "reduction",
            "int32",
            "int32",
            "contiguous",
            "exact",
            lambda: ravel_axis_reduce(f.int2_left, "max", -1),
        ),
        row(
            "rank2_int_arg_min_axis_neg1",
            "reduction",
            "int32",
            "int32",
            "contiguous",
            "exact",
            lambda: ravel_axis_reduce(f.int2_left, "argMin", -1),
        ),
        row(
            "rank2_int_arg_max_axis_neg1",
            "reduction",
            "int32",
            "int32",
            "contiguous",
            "exact",
            lambda: ravel_axis_reduce(f.int2_left, "argMax", -1),
        ),
        row(
            "rank2_int_sum_keepdims_neg1",
            "reduction",
            "int32",
            "int32",
            "contiguous",
            "exact",
            lambda: ravel_axis_reduce(f.int2_left, "sum", -1, keepdims=True),
        ),
        row(
            "rank3_int_sum_axis_neg1",
            "reduction",
            "int32",
            "int32",
            "contiguous",
            "exact",
            lambda: ravel_axis_reduce(f.int3, "sum", -1),
        ),
        row(
            "rank2_double_mean_axis_neg1",
            "reduction",
            "float64",
            "float64",
            "contiguous",
            "floating",
            lambda: ravel_axis_reduce(f.double_left, "mean", -1),
        ),
        row(
            "uint8_max_axis_neg1",
            "reduction",
            "uint8",
            "uint8",
            "contiguous",
            "exact",
            lambda: ravel_axis_reduce(f.uint8_left, "max", -1),
        ),
        row(
            "long_arg_min_axis_neg1",
            "reduction",
            "int64",
            "int32",
            "contiguous",
            "exact",
            lambda: ravel_axis_reduce(f.long_left, "argMin", -1),
        ),
        row(
            "empty_fiber_int_sum_axis_neg1",
            "reduction",
            "int32",
            "int32",
            "contiguous",
            "exact",
            lambda: ravel_axis_reduce(f.empty_fiber_int, "sum", -1),
        ),
        row(
            "empty_fiber_bool_all_axis_neg2",
            "reduction",
            "bool",
            "bool",
            "contiguous",
            "exact",
            lambda: ravel_axis_reduce(f.empty_fiber_bool, "all", -2),
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
