#!/usr/bin/env python3
"""Benchmark NumPy operations matched to Ravel's public access-pattern suite."""

from __future__ import annotations

import argparse
import gc
import json
import math
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
    logical_work_units: int
    work_unit: str
    run: Callable[[], object]


class Fixture:
    def __init__(self, side: int) -> None:
        if side <= 0 or side % 2 != 0:
            raise ValueError(f"side must be a positive even integer, got {side}")
        rows = np.arange(side, dtype=np.int64)[:, None]
        columns = np.arange(side, dtype=np.int64)[None, :]
        self.side = side
        self.left = ((rows * 131 + columns * 17) % 251 - 125).astype(np.float64) / 16.0
        self.right = ((rows * 43 + columns * 19) % 257 - 128).astype(np.float64) / 32.0
        self.row = ((np.arange(side) * 29) % 127 - 63).astype(np.float64) / 8.0

        self.inner_stride_left = self.left[:, 0:side:2]
        self.inner_stride_right = self.right[:, 1:side:2]
        self.outer_stride_left = self.left[0:side:2, :]
        self.outer_stride_right = self.right[1:side:2, :]
        self.reversed_left = self.left[:, ::-1]
        self.transposed_left = self.left.T
        self.transposed_right = self.right.T

    def scalar_read_row_major(self) -> float:
        total = 0.0
        for row in range(self.side):
            for column in range(self.side):
                total += float(self.left[row, column])
        return total

    def scalar_read_column_major(self) -> float:
        total = 0.0
        for column in range(self.side):
            for row in range(self.side):
                total += float(self.left[row, column])
        return total


def benchmark_cases(fixture: Fixture) -> list[BenchmarkCase]:
    side = fixture.side
    full = side * side
    half = full // 2
    return [
        BenchmarkCase(
            "contiguous_add",
            "elementwise",
            full,
            "element",
            lambda: np.add(fixture.left, fixture.right, order="C"),
        ),
        BenchmarkCase(
            "inner_stride_add",
            "elementwise",
            half,
            "element",
            lambda: np.add(
                fixture.inner_stride_left, fixture.inner_stride_right, order="C"
            ),
        ),
        BenchmarkCase(
            "outer_stride_add",
            "elementwise",
            half,
            "element",
            lambda: np.add(
                fixture.outer_stride_left, fixture.outer_stride_right, order="C"
            ),
        ),
        BenchmarkCase(
            "reverse_add",
            "elementwise",
            full,
            "element",
            lambda: np.add(fixture.reversed_left, fixture.right, order="C"),
        ),
        BenchmarkCase(
            "transpose_add",
            "elementwise",
            full,
            "element",
            lambda: np.add(
                fixture.transposed_left, fixture.transposed_right, order="C"
            ),
        ),
        BenchmarkCase(
            "broadcast_row_add",
            "broadcast",
            full,
            "element",
            lambda: np.add(fixture.left, fixture.row, order="C"),
        ),
        BenchmarkCase(
            "full_sum_contiguous",
            "reduction",
            full,
            "element",
            lambda: float(fixture.left.sum()),
        ),
        BenchmarkCase(
            "full_sum_inner_stride",
            "reduction",
            half,
            "element",
            lambda: float(fixture.inner_stride_left.sum()),
        ),
        BenchmarkCase(
            "axis0_sum",
            "reduction",
            full,
            "element",
            lambda: fixture.left.sum(axis=0),
        ),
        BenchmarkCase(
            "axis1_sum",
            "reduction",
            full,
            "element",
            lambda: fixture.left.sum(axis=1),
        ),
        BenchmarkCase(
            "copy_inner_stride",
            "copy",
            half,
            "element",
            lambda: fixture.inner_stride_left.copy(order="C"),
        ),
        BenchmarkCase(
            "copy_transpose",
            "copy",
            full,
            "element",
            lambda: fixture.transposed_left.copy(order="C"),
        ),
        BenchmarkCase(
            "scalar_read_row_major",
            "scalar access",
            full,
            "element access",
            fixture.scalar_read_row_major,
        ),
        BenchmarkCase(
            "scalar_read_column_major",
            "scalar access",
            full,
            "element access",
            fixture.scalar_read_column_major,
        ),
        BenchmarkCase(
            "view_inner_stride_create",
            "view creation",
            1,
            "view",
            lambda: fixture.left[:, 0:side:2],
        ),
        BenchmarkCase(
            "view_transpose_create",
            "view creation",
            1,
            "view",
            lambda: fixture.left.T,
        ),
    ]


def signature(value: object) -> dict[str, int | float]:
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

    # Keep the final result live through the measurement loop.
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
        for case in benchmark_cases(fixture):
            semantic_signature = signature(case.run())
            row: dict[str, object] = {
                "case": case.name,
                "family": case.family,
                "side": side,
                "logical_work_units": case.logical_work_units,
                "work_unit": case.work_unit,
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
        "dtype": "float64",
        "allocation_policy": "public allocating operation; views remain zero-copy",
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
        "schema": "ravel.access-pattern.numpy.v1",
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
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--target-ms", type=float, default=200.0)
    parser.add_argument(
        "--signatures-only",
        action="store_true",
        help="emit semantic signatures without timing (CI correctness gate)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("target/access-patterns/numpy.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sides = args.sides or [256, 1024]
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
