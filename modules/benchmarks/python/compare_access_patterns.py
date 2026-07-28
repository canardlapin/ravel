#!/usr/bin/env python3
"""Validate Ravel/NumPy parity and render a comparative benchmark report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


CASE_ORDER = [
    "contiguous_add",
    "inner_stride_add",
    "outer_stride_add",
    "reverse_add",
    "transpose_add",
    "broadcast_row_add",
    "full_sum_contiguous",
    "full_sum_inner_stride",
    "axis0_sum",
    "axis1_sum",
    "copy_inner_stride",
    "copy_transpose",
    "scalar_read_row_major",
    "scalar_read_column_major",
    "view_inner_stride_create",
    "view_transpose_create",
]
CASE_INDEX = {name: index for index, name in enumerate(CASE_ORDER)}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def score_to_ns(score: float, unit: str) -> float:
    factors = {
        "ns/op": 1.0,
        "us/op": 1_000.0,
        "ms/op": 1_000_000.0,
        "s/op": 1_000_000_000.0,
    }
    try:
        return score * factors[unit]
    except KeyError as error:
        raise ValueError(f"unsupported JMH score unit: {unit}") from error


def read_jmh(payload: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    results: dict[tuple[str, int], dict[str, Any]] = {}
    for entry in payload:
        name = entry["benchmark"].rsplit(".", 1)[-1]
        if name not in CASE_INDEX:
            continue
        side = int(entry["params"]["side"])
        metric = entry["primaryMetric"]
        key = (name, side)
        if key in results:
            raise ValueError(f"duplicate JMH result for {key}")
        median_score = metric.get("scorePercentiles", {}).get("50.0", metric["score"])
        results[key] = {
            "ravel_ns": score_to_ns(float(median_score), metric["scoreUnit"]),
            "ravel_error_ns": score_to_ns(
                float(metric.get("scoreError", 0.0)), metric["scoreUnit"]
            ),
        }
    return results


def index_results(payload: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(entry["case"], int(entry["side"])): entry for entry in payload["results"]}


def validate_signatures(
    ravel: dict[tuple[str, int], dict[str, Any]],
    numpy: dict[tuple[str, int], dict[str, Any]],
) -> None:
    if set(ravel) != set(numpy):
        missing_ravel = sorted(set(numpy) - set(ravel))
        missing_numpy = sorted(set(ravel) - set(numpy))
        raise ValueError(
            f"signature case mismatch; missing Ravel={missing_ravel}, "
            f"missing NumPy={missing_numpy}"
        )
    for key in sorted(ravel):
        left = ravel[key]
        right = numpy[key]
        for field in (
            "logical_work_units",
            "work_unit",
            "result_size",
            "result_layout",
        ):
            if left[field] != right[field]:
                raise ValueError(
                    f"parity failed for {key}: {field} "
                    f"Ravel={left[field]!r}, NumPy={right[field]!r}"
                )
        for field in ("sum", "weighted_sum"):
            if not math.isclose(
                float(left[field]),
                float(right[field]),
                rel_tol=1e-12,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"parity failed for {key}: {field} "
                    f"Ravel={left[field]!r}, NumPy={right[field]!r}"
                )


def report(
    jmh: dict[tuple[str, int], dict[str, Any]],
    signatures: dict[tuple[str, int], dict[str, Any]],
    numpy: dict[tuple[str, int], dict[str, Any]],
    numpy_metadata: dict[str, Any],
    allow_partial: bool,
) -> str:
    expected = set(signatures)
    measured = set(jmh)
    if not allow_partial and measured != expected:
        missing = sorted(expected - measured)
        extra = sorted(measured - expected)
        raise ValueError(f"JMH case mismatch; missing={missing}, extra={extra}")
    keys = measured & expected
    if not keys:
        raise ValueError("no matching JMH access-pattern results found")

    lines = [
        "# Ravel JVM vs NumPy access-pattern benchmark",
        "",
        "Semantic parity passed for every reported row. Timings compare public, "
        "allocating operations except for zero-copy view creation and scalar reads.",
        "",
        f"NumPy {numpy_metadata.get('numpy_version', 'unknown')} on Python "
        f"{numpy_metadata.get('python_version', 'unknown')}; "
        "Ravel timings are from the supplied JMH result.",
        "",
        "| side | family | case | work | Ravel median ns/op | NumPy median ns/op | "
        "Ravel speed vs NumPy | Ravel ns/unit | NumPy ns/unit |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    ordered = sorted(keys, key=lambda key: (key[1], CASE_INDEX[key[0]]))
    for key in ordered:
        numpy_row = numpy[key]
        units = int(signatures[key]["logical_work_units"])
        ravel_ns = float(jmh[key]["ravel_ns"])
        numpy_ns = float(numpy_row["median_ns"])
        ravel_speed = numpy_ns / ravel_ns
        lines.append(
            f"| {key[1]} | {numpy_row['family']} | `{key[0]}` | "
            f"{units:,} {numpy_row['work_unit']} | {ravel_ns:,.1f} | "
            f"{numpy_ns:,.1f} | {ravel_speed:.3f}x | "
            f"{ravel_ns / units:,.3f} | {numpy_ns / units:,.3f} |"
        )
    lines.extend(
        [
            "",
            "A speed value above 1.0x favors Ravel; below 1.0x favors NumPy. "
            "Scalar-access rows include each host language's call/indexing overhead "
            "and are not native-kernel comparisons.",
            "",
            "These cross-process timings are descriptive. Treat changes as regressions "
            "only against same-host baselines collected with the same runtimes and "
            "benchmark settings.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jmh",
        type=Path,
        help="JMH JSON results; required unless --parity-only",
    )
    parser.add_argument("--numpy", required=True, type=Path)
    parser.add_argument("--signatures", required=True, type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        help="comparison markdown path; required unless --parity-only",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="render the intersection of JMH and parity cases (for smoke runs)",
    )
    parser.add_argument(
        "--parity-only",
        action="store_true",
        help="validate Ravel/NumPy semantic signatures without JMH timings",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    numpy_payload = load_json(args.numpy)
    signatures_payload = load_json(args.signatures)
    numpy = index_results(numpy_payload)
    signatures = index_results(signatures_payload)
    validate_signatures(signatures, numpy)
    if args.parity_only:
        print(
            f"Parity OK: {len(signatures)} matched cases "
            f"({sorted({side for _, side in signatures})})"
        )
        return
    if args.jmh is None or args.out is None:
        raise SystemExit("--jmh and --out are required unless --parity-only")
    jmh = read_jmh(load_json(args.jmh))
    rendered = report(
        jmh,
        signatures,
        numpy,
        numpy_payload["metadata"],
        args.allow_partial,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered, encoding="utf-8")
    print(f"Wrote comparison report to {args.out}")


if __name__ == "__main__":
    main()
