#!/usr/bin/env python3
"""Benchmark NumPy operations matched to Ravel's broad public-operation matrix."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    family: str
    input_dtype: str
    result_dtype: str
    input_layout: str
    logical_work_units: int
    work_unit: str
    comparison: str
    run: Callable[[], object]
    signature_run: Callable[[], object]


def benchmark_case(
    name: str,
    family: str,
    input_dtype: str,
    result_dtype: str,
    input_layout: str,
    logical_work_units: int,
    run: Callable[[], object],
    *,
    comparison: str = "floating",
    signature_run: Callable[[], object] | None = None,
) -> BenchmarkCase:
    return BenchmarkCase(
        name,
        family,
        input_dtype,
        result_dtype,
        input_layout,
        logical_work_units,
        "element",
        comparison,
        run,
        run if signature_run is None else signature_run,
    )


class Fixture:
    def __init__(self, side: int) -> None:
        if side <= 0 or side % 2 != 0:
            raise ValueError(f"side must be a positive even integer, got {side}")
        rows = np.arange(side, dtype=np.int64)[:, None]
        columns = np.arange(side, dtype=np.int64)[None, :]
        self.side = side
        self.full_size = side * side
        self.half_size = self.full_size // 2

        self.double_left = (
            ((rows * 131 + columns * 17) % 251 - 125).astype(np.float64) / 16.0
        )
        self.double_right = (
            ((rows * 43 + columns * 19) % 257 - 128).astype(np.float64) / 32.0
        )
        self.double_positive = (
            ((rows * 23 + columns * 11) % 97 + 1).astype(np.float64) / 17.0
        )
        self.double_product = (
            1.0
            + (((rows * 13 + columns * 7) % 17) - 8).astype(np.float64) * 1.0e-7
        )
        self.double_non_finite = self.double_left.copy()
        flat_non_finite = self.double_non_finite.ravel()
        flat_non_finite[::263] = np.inf
        flat_non_finite[::257] = np.nan
        self.double_row = (
            ((np.arange(side, dtype=np.int64) * 29) % 127 - 63).astype(np.float64)
            / 8.0
        )

        self.float_left = self.double_left.astype(np.float32)
        self.float_right = self.double_right.astype(np.float32)
        self.float_product = self.double_product.astype(np.float32)

        self.int_left = ((rows * 17 + columns * 5) % 23 - 11).astype(np.int32)
        self.int_right = ((rows * 7 + columns * 3) % 11 - 5).astype(np.int32)
        self.int_positive = ((rows * 3 + columns * 5) % 7 + 1).astype(np.int32)
        self.int_product = np.where((rows + columns) % 3 == 0, -1, 1).astype(
            np.int32
        )

        self.long_left = ((rows * 31 + columns * 7) % 41 - 20).astype(np.int64)
        self.long_right = ((rows * 5 + columns * 11) % 19 - 9).astype(np.int64)
        self.long_positive = ((rows * 7 + columns * 3) % 9 + 1).astype(np.int64)
        self.long_product = np.where((rows + columns) % 5 == 0, -1, 1).astype(
            np.int64
        )

        self.inner_double_left = self.double_left[:, 0:side:2]
        self.inner_double_right = self.double_right[:, 1:side:2]
        self.outer_double_left = self.double_left[0:side:2, :]
        self.outer_double_positive = self.double_positive[1:side:2, :]
        self.reversed_double = self.double_left[:, ::-1]
        self.transposed_double = self.double_left.T
        self.transposed_right = self.double_right.T

        self.mutable_add_double = self.double_left.copy()
        self.mutable_subtract_double = self.double_left.copy()
        self.mutable_multiply_double = self.double_left.copy()
        self.mutable_divide_double = self.double_positive.copy()
        self.mutable_add_float = self.float_left.copy()
        self.mutable_add_int = self.int_left.copy()
        self.mutable_add_long = self.long_left.copy()
        mutable_reverse_owner = self.double_left.copy()
        self.mutable_reverse_double = mutable_reverse_owner[:, ::-1]
        mutable_inner_owner = self.double_product.copy()
        self.mutable_inner_double = mutable_inner_owner[:, 0:side:2]

    def inplace_add_double(self) -> np.ndarray:
        return np.add(
            self.mutable_add_double, np.float64(0.25), out=self.mutable_add_double
        )

    def inplace_subtract_double(self) -> np.ndarray:
        return np.subtract(
            self.mutable_subtract_double,
            np.float64(0.125),
            out=self.mutable_subtract_double,
        )

    def inplace_multiply_double(self) -> np.ndarray:
        return np.multiply(
            self.mutable_multiply_double,
            np.float64(-1.0),
            out=self.mutable_multiply_double,
        )

    def inplace_divide_double(self) -> np.ndarray:
        return np.divide(
            self.mutable_divide_double,
            np.float64(-1.0),
            out=self.mutable_divide_double,
        )

    def inplace_add_float(self) -> np.ndarray:
        return np.add(
            self.mutable_add_float, np.float32(0.25), out=self.mutable_add_float
        )

    def inplace_add_int(self) -> np.ndarray:
        return np.add(self.mutable_add_int, np.int32(1), out=self.mutable_add_int)

    def inplace_add_long(self) -> np.ndarray:
        return np.add(self.mutable_add_long, np.int64(1), out=self.mutable_add_long)

    def inplace_reverse_add_double(self) -> np.ndarray:
        return np.add(
            self.mutable_reverse_double,
            np.float64(0.25),
            out=self.mutable_reverse_double,
        )

    def inplace_inner_multiply_double(self) -> np.ndarray:
        return np.multiply(
            self.mutable_inner_double,
            np.float64(-1.0),
            out=self.mutable_inner_double,
        )


def benchmark_cases(fixture: Fixture) -> list[BenchmarkCase]:
    f = fixture
    full = f.full_size
    half = f.half_size
    floating = benchmark_case

    def exact(*args: object, **kwargs: object) -> BenchmarkCase:
        kwargs["comparison"] = "exact"
        return benchmark_case(*args, **kwargs)  # type: ignore[arg-type]

    return [
        floating(
            "contiguous_subtract_double",
            "binary",
            "float64",
            "float64",
            "contiguous",
            full,
            lambda: np.subtract(f.double_left, f.double_right, order="C"),
        ),
        floating(
            "inner_stride_multiply_double",
            "binary",
            "float64",
            "float64",
            "inner_stride",
            half,
            lambda: np.multiply(
                f.inner_double_left, f.inner_double_right, order="C"
            ),
        ),
        floating(
            "outer_stride_divide_double",
            "binary",
            "float64",
            "float64",
            "outer_stride",
            half,
            lambda: np.divide(
                f.outer_double_left, f.outer_double_positive, order="C"
            ),
        ),
        floating(
            "reverse_minimum_double",
            "binary",
            "float64",
            "float64",
            "reversed",
            full,
            lambda: np.minimum(f.reversed_double, f.double_right, order="C"),
        ),
        floating(
            "transpose_maximum_double",
            "binary",
            "float64",
            "float64",
            "transposed",
            full,
            lambda: np.maximum(
                f.transposed_double, f.transposed_right, order="C"
            ),
        ),
        floating(
            "broadcast_subtract_double",
            "binary",
            "float64",
            "float64",
            "broadcast",
            full,
            lambda: np.subtract(f.double_left, f.double_row, order="C"),
        ),
        floating(
            "scalar_add_double",
            "scalar",
            "float64",
            "float64",
            "contiguous",
            full,
            lambda: np.add(f.double_left, np.float64(1.25), order="C"),
        ),
        floating(
            "scalar_subtract_double",
            "scalar",
            "float64",
            "float64",
            "contiguous",
            full,
            lambda: np.subtract(f.double_left, np.float64(0.75), order="C"),
        ),
        floating(
            "scalar_multiply_double",
            "scalar",
            "float64",
            "float64",
            "contiguous",
            full,
            lambda: np.multiply(f.double_left, np.float64(-1.5), order="C"),
        ),
        floating(
            "scalar_divide_double",
            "scalar",
            "float64",
            "float64",
            "contiguous",
            full,
            lambda: np.divide(f.double_left, np.float64(3.0), order="C"),
        ),
        floating(
            "scalar_minimum_double",
            "scalar",
            "float64",
            "float64",
            "contiguous",
            full,
            lambda: np.minimum(f.double_left, np.float64(0.5), order="C"),
        ),
        floating(
            "scalar_maximum_double",
            "scalar",
            "float64",
            "float64",
            "contiguous",
            full,
            lambda: np.maximum(f.double_left, np.float64(-0.5), order="C"),
        ),
        floating(
            "clip_double",
            "unary",
            "float64",
            "float64",
            "contiguous",
            full,
            lambda: np.clip(f.double_left, -2.0, 3.0),
        ),
        floating(
            "negate_double",
            "unary",
            "float64",
            "float64",
            "contiguous",
            full,
            lambda: np.negative(f.double_left, order="C"),
        ),
        floating(
            "abs_double",
            "unary",
            "float64",
            "float64",
            "contiguous",
            full,
            lambda: np.absolute(f.double_left, order="C"),
        ),
        floating(
            "sqrt_double",
            "unary",
            "float64",
            "float64",
            "contiguous",
            full,
            lambda: np.sqrt(f.double_positive, order="C"),
        ),
        floating(
            "exp_double",
            "unary",
            "float64",
            "float64",
            "contiguous",
            full,
            lambda: np.exp(f.double_left, order="C"),
        ),
        floating(
            "log_double",
            "unary",
            "float64",
            "float64",
            "contiguous",
            full,
            lambda: np.log(f.double_positive, order="C"),
        ),
        floating(
            "sin_double",
            "unary",
            "float64",
            "float64",
            "contiguous",
            full,
            lambda: np.sin(f.double_left, order="C"),
        ),
        floating(
            "cos_double",
            "unary",
            "float64",
            "float64",
            "contiguous",
            full,
            lambda: np.cos(f.double_left, order="C"),
        ),
        floating(
            "tan_double",
            "unary",
            "float64",
            "float64",
            "contiguous",
            full,
            lambda: np.tan(f.double_left, order="C"),
        ),
        floating(
            "floor_double",
            "unary",
            "float64",
            "float64",
            "contiguous",
            full,
            lambda: np.floor(f.double_left, order="C"),
        ),
        floating(
            "ceil_double",
            "unary",
            "float64",
            "float64",
            "contiguous",
            full,
            lambda: np.ceil(f.double_left, order="C"),
        ),
        exact(
            "equal_double",
            "comparison",
            "float64",
            "bool",
            "contiguous",
            full,
            lambda: np.equal(f.double_left, f.double_right, order="C"),
        ),
        exact(
            "not_equal_double",
            "comparison",
            "float64",
            "bool",
            "contiguous",
            full,
            lambda: np.not_equal(f.double_left, f.double_right, order="C"),
        ),
        exact(
            "less_inner_stride_double",
            "comparison",
            "float64",
            "bool",
            "inner_stride",
            half,
            lambda: np.less(
                f.inner_double_left, f.inner_double_right, order="C"
            ),
        ),
        exact(
            "less_equal_outer_stride_double",
            "comparison",
            "float64",
            "bool",
            "outer_stride",
            half,
            lambda: np.less_equal(
                f.outer_double_left, f.outer_double_positive, order="C"
            ),
        ),
        exact(
            "greater_broadcast_double",
            "comparison",
            "float64",
            "bool",
            "broadcast",
            full,
            lambda: np.greater(f.double_left, f.double_row, order="C"),
        ),
        exact(
            "greater_equal_scalar_double",
            "comparison",
            "float64",
            "bool",
            "contiguous",
            full,
            lambda: np.greater_equal(f.double_left, np.float64(0.0), order="C"),
        ),
        exact(
            "is_nan_double",
            "predicate",
            "float64",
            "bool",
            "contiguous",
            full,
            lambda: np.isnan(f.double_non_finite, order="C"),
        ),
        exact(
            "is_finite_double",
            "predicate",
            "float64",
            "bool",
            "contiguous",
            full,
            lambda: np.isfinite(f.double_non_finite, order="C"),
        ),
        exact(
            "cast_double_int",
            "cast",
            "float64",
            "int32",
            "contiguous",
            full,
            lambda: f.double_left.astype(np.int32, order="C", copy=True),
        ),
        floating(
            "cast_float_double",
            "cast",
            "float32",
            "float64",
            "contiguous",
            full,
            lambda: f.float_left.astype(np.float64, order="C", copy=True),
        ),
        exact(
            "cast_int_long",
            "cast",
            "int32",
            "int64",
            "contiguous",
            full,
            lambda: f.int_left.astype(np.int64, order="C", copy=True),
        ),
        floating(
            "cast_long_float",
            "cast",
            "int64",
            "float32",
            "contiguous",
            full,
            lambda: f.long_left.astype(np.float32, order="C", copy=True),
        ),
        floating(
            "contiguous_add_float",
            "binary",
            "float32",
            "float32",
            "contiguous",
            full,
            lambda: np.add(f.float_left, f.float_right, order="C"),
        ),
        floating(
            "contiguous_multiply_float",
            "binary",
            "float32",
            "float32",
            "contiguous",
            full,
            lambda: np.multiply(f.float_left, f.float_right, order="C"),
        ),
        floating(
            "scalar_divide_float",
            "scalar",
            "float32",
            "float32",
            "contiguous",
            full,
            lambda: np.divide(f.float_left, np.float32(3.0), order="C"),
        ),
        exact(
            "contiguous_add_int",
            "binary",
            "int32",
            "int32",
            "contiguous",
            full,
            lambda: np.add(f.int_left, f.int_right, order="C"),
        ),
        exact(
            "contiguous_multiply_int",
            "binary",
            "int32",
            "int32",
            "contiguous",
            full,
            lambda: np.multiply(f.int_left, f.int_right, order="C"),
        ),
        exact(
            "scalar_quot_int",
            "scalar",
            "int32",
            "int32",
            "contiguous",
            full,
            lambda: np.floor_divide(f.int_positive, np.int32(3)),
        ),
        exact(
            "contiguous_add_long",
            "binary",
            "int64",
            "int64",
            "contiguous",
            full,
            lambda: np.add(f.long_left, f.long_right, order="C"),
        ),
        exact(
            "contiguous_multiply_long",
            "binary",
            "int64",
            "int64",
            "contiguous",
            full,
            lambda: np.multiply(f.long_left, f.long_right, order="C"),
        ),
        exact(
            "scalar_quot_long",
            "scalar",
            "int64",
            "int64",
            "contiguous",
            full,
            lambda: np.floor_divide(f.long_positive, np.int64(3)),
        ),
        floating(
            "full_product_double",
            "reduction",
            "float64",
            "float64",
            "contiguous",
            full,
            lambda: np.prod(f.double_product, dtype=np.float64),
        ),
        floating(
            "full_min_double",
            "reduction",
            "float64",
            "float64",
            "contiguous",
            full,
            lambda: np.min(f.double_left),
        ),
        floating(
            "full_max_double",
            "reduction",
            "float64",
            "float64",
            "contiguous",
            full,
            lambda: np.max(f.double_left),
        ),
        exact(
            "full_arg_min_double",
            "reduction",
            "float64",
            "int32",
            "contiguous",
            full,
            lambda: np.argmin(f.double_left),
        ),
        exact(
            "full_arg_max_double",
            "reduction",
            "float64",
            "int32",
            "contiguous",
            full,
            lambda: np.argmax(f.double_left),
        ),
        floating(
            "full_mean_double",
            "reduction",
            "float64",
            "float64",
            "contiguous",
            full,
            lambda: np.mean(f.double_left, dtype=np.float64),
        ),
        floating(
            "axis0_product_double",
            "reduction",
            "float64",
            "float64",
            "contiguous",
            full,
            lambda: np.prod(f.double_product, axis=0, dtype=np.float64),
        ),
        floating(
            "axis1_product_double",
            "reduction",
            "float64",
            "float64",
            "contiguous",
            full,
            lambda: np.prod(f.double_product, axis=1, dtype=np.float64),
        ),
        floating(
            "axis0_min_double",
            "reduction",
            "float64",
            "float64",
            "contiguous",
            full,
            lambda: np.min(f.double_left, axis=0),
        ),
        floating(
            "axis1_max_double",
            "reduction",
            "float64",
            "float64",
            "contiguous",
            full,
            lambda: np.max(f.double_left, axis=1),
        ),
        exact(
            "axis0_arg_min_double",
            "reduction",
            "float64",
            "int32",
            "contiguous",
            full,
            lambda: np.argmin(f.double_left, axis=0),
        ),
        exact(
            "axis1_arg_max_double",
            "reduction",
            "float64",
            "int32",
            "contiguous",
            full,
            lambda: np.argmax(f.double_left, axis=1),
        ),
        floating(
            "axis0_mean_double",
            "reduction",
            "float64",
            "float64",
            "contiguous",
            full,
            lambda: np.mean(f.double_left, axis=0, dtype=np.float64),
        ),
        floating(
            "axis1_mean_double",
            "reduction",
            "float64",
            "float64",
            "contiguous",
            full,
            lambda: np.mean(f.double_left, axis=1, dtype=np.float64),
        ),
        floating(
            "inner_stride_product_double",
            "reduction",
            "float64",
            "float64",
            "inner_stride",
            half,
            lambda: np.prod(
                f.double_product[:, 0 : f.side : 2], dtype=np.float64
            ),
        ),
        floating(
            "reverse_min_double",
            "reduction",
            "float64",
            "float64",
            "reversed",
            full,
            lambda: np.min(f.reversed_double),
        ),
        floating(
            "transpose_max_double",
            "reduction",
            "float64",
            "float64",
            "transposed",
            full,
            lambda: np.max(f.transposed_double),
        ),
        floating(
            "full_sum_float",
            "reduction",
            "float32",
            "float32",
            "contiguous",
            full,
            lambda: np.sum(f.float_left, dtype=np.float32),
        ),
        floating(
            "full_product_float",
            "reduction",
            "float32",
            "float32",
            "contiguous",
            full,
            lambda: np.prod(f.float_product, dtype=np.float32),
        ),
        floating(
            "full_mean_float",
            "reduction",
            "float32",
            "float32",
            "contiguous",
            full,
            lambda: np.mean(f.float_left, dtype=np.float32),
        ),
        exact(
            "full_sum_int",
            "reduction",
            "int32",
            "int32",
            "contiguous",
            full,
            lambda: np.sum(f.int_left, dtype=np.int32),
        ),
        exact(
            "full_product_int",
            "reduction",
            "int32",
            "int32",
            "contiguous",
            full,
            lambda: np.prod(f.int_product, dtype=np.int32),
        ),
        exact(
            "full_sum_long",
            "reduction",
            "int64",
            "int64",
            "contiguous",
            full,
            lambda: np.sum(f.long_left, dtype=np.int64),
        ),
        exact(
            "full_product_long",
            "reduction",
            "int64",
            "int64",
            "contiguous",
            full,
            lambda: np.prod(f.long_product, dtype=np.int64),
        ),
        exact(
            "sum_as_long_int",
            "reduction",
            "int32",
            "int64",
            "contiguous",
            full,
            lambda: np.sum(f.int_left, dtype=np.int64),
        ),
        floating(
            "sum_as_double_float",
            "reduction",
            "float32",
            "float64",
            "contiguous",
            full,
            lambda: np.sum(f.float_left, dtype=np.float64),
        ),
        floating(
            "inplace_add_double",
            "in-place",
            "float64",
            "float64",
            "contiguous",
            full,
            f.inplace_add_double,
        ),
        floating(
            "inplace_subtract_double",
            "in-place",
            "float64",
            "float64",
            "contiguous",
            full,
            f.inplace_subtract_double,
        ),
        floating(
            "inplace_multiply_double",
            "in-place",
            "float64",
            "float64",
            "contiguous",
            full,
            f.inplace_multiply_double,
        ),
        floating(
            "inplace_divide_double",
            "in-place",
            "float64",
            "float64",
            "contiguous",
            full,
            f.inplace_divide_double,
        ),
        floating(
            "inplace_add_float",
            "in-place",
            "float32",
            "float32",
            "contiguous",
            full,
            f.inplace_add_float,
        ),
        exact(
            "inplace_add_int",
            "in-place",
            "int32",
            "int32",
            "contiguous",
            full,
            f.inplace_add_int,
        ),
        exact(
            "inplace_add_long",
            "in-place",
            "int64",
            "int64",
            "contiguous",
            full,
            f.inplace_add_long,
        ),
        floating(
            "inplace_reverse_add_double",
            "in-place",
            "float64",
            "float64",
            "reversed",
            full,
            f.inplace_reverse_add_double,
            signature_run=lambda: f.inplace_reverse_add_double().copy(order="C"),
        ),
        floating(
            "inplace_inner_stride_multiply_double",
            "in-place",
            "float64",
            "float64",
            "inner_stride",
            half,
            f.inplace_inner_multiply_double,
            signature_run=lambda: f.inplace_inner_multiply_double().copy(order="C"),
        ),
    ]


def signature(value: object) -> dict[str, int | float | str]:
    if isinstance(value, np.ndarray):
        flat = value.ravel(order="C")
        total = 0.0
        weighted_total = 0.0
        for index, element in enumerate(flat, start=1):
            scalar = float(element)
            total += scalar
            weighted_total += scalar * index
        return {
            "result_size": int(flat.size),
            "result_layout": (
                "c_contiguous" if value.flags.c_contiguous else "strided_view"
            ),
            "sum": total,
            "weighted_sum": weighted_total,
        }
    scalar = float(value)
    return {
        "result_size": 1,
        "result_layout": "scalar",
        "sum": scalar,
        "weighted_sum": scalar,
    }


def measure(
    run: Callable[[], object],
    warmups: int,
    samples: int,
    target_ns: int,
) -> tuple[float, float, int]:
    for _ in range(warmups):
        run()

    loops = 1
    while True:
        started = time.perf_counter_ns()
        for _ in range(loops):
            value = run()
        elapsed = time.perf_counter_ns() - started
        if elapsed >= target_ns or loops >= 1_048_576:
            break
        if elapsed <= 0:
            loops *= 2
        else:
            loops = min(
                1_048_576,
                max(loops + 1, math.ceil(loops * target_ns / elapsed)),
            )

    timings: list[float] = []
    for _ in range(samples):
        gc.collect()
        started = time.perf_counter_ns()
        for _ in range(loops):
            value = run()
        elapsed = time.perf_counter_ns() - started
        timings.append(elapsed / loops)

    if value is None:
        raise AssertionError("benchmark workload returned None")
    median = statistics.median(timings)
    deviations = [abs(timing - median) for timing in timings]
    return median, statistics.median(deviations), loops


def run_suite(
    sides: list[int],
    warmups: int,
    samples: int,
    target_ms: float,
    *,
    signatures_only: bool,
) -> dict[str, object]:
    results: list[dict[str, object]] = []
    for side in sides:
        fixture = Fixture(side)
        cases = benchmark_cases(fixture)
        names = [case.name for case in cases]
        if len(names) != len(set(names)):
            raise AssertionError(f"duplicate operation-matrix case at side {side}")
        for case in cases:
            semantic_signature = signature(case.signature_run())
            row: dict[str, object] = {
                "case": case.name,
                "family": case.family,
                "side": side,
                "input_dtype": case.input_dtype,
                "result_dtype": case.result_dtype,
                "input_layout": case.input_layout,
                "logical_work_units": case.logical_work_units,
                "work_unit": case.work_unit,
                "comparison": case.comparison,
                **semantic_signature,
            }
            if not signatures_only:
                median_ns, mad_ns, loops = measure(
                    case.run,
                    warmups=warmups,
                    samples=samples,
                    target_ns=max(1, round(target_ms * 1_000_000)),
                )
                row["median_ns"] = median_ns
                row["mad_ns"] = mad_ns
                row["loops_per_sample"] = loops
            results.append(row)
    metadata: dict[str, object] = {
        "library": "numpy",
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "available_processors": os.cpu_count(),
        "allocation_policy": "public allocating operations; in-place cases reuse destination",
        "mode": "signatures_only" if signatures_only else "timed",
    }
    if not signatures_only:
        metadata.update(
            {
                "timer": "time.perf_counter_ns",
                "warmups": warmups,
                "samples": samples,
                "target_ms_per_sample": target_ms,
            }
        )
    return {
        "schema": "ravel.operation-matrix.numpy.v1",
        "metadata": metadata,
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--side",
        action="append",
        type=int,
        dest="sides",
        help="even matrix side length; repeat for multiple sizes",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--target-ms", type=float, default=150.0)
    parser.add_argument(
        "--signatures-only",
        action="store_true",
        help="emit semantic signatures without timing (CI correctness gate)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("target/operation-matrix/numpy.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sides = args.sides or [64, 256, 1024]
    if args.warmup < 0 or args.samples < 1 or args.target_ms <= 0:
        raise ValueError(
            "warmup must be non-negative; samples and target-ms must be positive"
        )
    result = run_suite(
        sides,
        args.warmup,
        args.samples,
        args.target_ms,
        signatures_only=args.signatures_only,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(result['results'])} results to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
