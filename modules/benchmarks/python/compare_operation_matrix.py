#!/usr/bin/env python3
"""Validate operation-matrix parity and render Ravel/NumPy timing summaries."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


METADATA_FIELDS = (
    "family",
    "input_dtype",
    "result_dtype",
    "input_layout",
    "logical_work_units",
    "work_unit",
    "comparison",
    "result_size",
    "result_layout",
)


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
        benchmark = entry["benchmark"].rsplit(".", 1)[-1]
        if benchmark != "operation_matrix":
            continue
        params = entry["params"]
        name = str(params["caseName"])
        side = int(params["side"])
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
    results: dict[tuple[str, int], dict[str, Any]] = {}
    for entry in payload["results"]:
        key = (str(entry["case"]), int(entry["side"]))
        if key in results:
            raise ValueError(f"duplicate semantic result for {key}")
        results[key] = entry
    return results


def validate_signatures(
    ravel: dict[tuple[str, int], dict[str, Any]],
    numpy: dict[tuple[str, int], dict[str, Any]],
) -> None:
    if set(ravel) != set(numpy):
        missing_ravel = sorted(set(numpy) - set(ravel))
        missing_numpy = sorted(set(ravel) - set(numpy))
        raise ValueError(
            f"operation-matrix case mismatch; missing Ravel={missing_ravel}, "
            f"missing NumPy={missing_numpy}"
        )
    for key in sorted(ravel):
        left = ravel[key]
        right = numpy[key]
        for field in METADATA_FIELDS:
            if left[field] != right[field]:
                raise ValueError(
                    f"parity failed for {key}: {field} "
                    f"Ravel={left[field]!r}, NumPy={right[field]!r}"
                )
        for field in ("sum", "weighted_sum"):
            left_value = float(left[field])
            right_value = float(right[field])
            if left["comparison"] == "exact":
                matched = left_value == right_value
            elif left["result_dtype"] == "float32":
                matched = math.isclose(
                    left_value,
                    right_value,
                    rel_tol=2e-6,
                    abs_tol=2e-3,
                )
            else:
                matched = math.isclose(
                    left_value,
                    right_value,
                    rel_tol=2e-10,
                    abs_tol=2e-7,
                )
            if not matched:
                raise ValueError(
                    f"parity failed for {key}: {field} "
                    f"Ravel={left[field]!r}, NumPy={right[field]!r}"
                )


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def summary_table(
    rows: list[dict[str, Any]],
    field: str,
    title: str,
) -> list[str]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(float(row["speed"]))
    lines = [
        f"## {title}",
        "",
        "| group | rows | geometric mean | minimum | maximum |",
        "|---|---:|---:|---:|---:|",
    ]
    for group in sorted(grouped):
        speeds = grouped[group]
        lines.append(
            f"| `{group}` | {len(speeds)} | {geometric_mean(speeds):.3f}x | "
            f"{min(speeds):.3f}x | {max(speeds):.3f}x |"
        )
    lines.append("")
    return lines


def report(
    jmh: dict[tuple[str, int], dict[str, Any]],
    signatures: dict[tuple[str, int], dict[str, Any]],
    signature_order: list[tuple[str, int]],
    numpy: dict[tuple[str, int], dict[str, Any]],
    numpy_metadata: dict[str, Any],
    ravel_metadata: dict[str, Any],
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
        raise ValueError("no matching JMH operation-matrix results found")

    rows: list[dict[str, Any]] = []
    for key in signature_order:
        if key not in keys:
            continue
        metadata = signatures[key]
        ravel_ns = float(jmh[key]["ravel_ns"])
        numpy_ns = float(numpy[key]["median_ns"])
        rows.append(
            {
                **metadata,
                "ravel_ns": ravel_ns,
                "numpy_ns": numpy_ns,
                "speed": numpy_ns / ravel_ns,
            }
        )

    lines = [
        "# Ravel JVM vs NumPy public-operation matrix",
        "",
        "Semantic parity passed for every reported row before timing comparison. "
        "This broad matrix locates follow-up targets; focused JMH controls remain "
        "the authority for optimization claims.",
        "",
        f"Ravel: Java {ravel_metadata.get('java_version', 'unknown')} "
        f"({ravel_metadata.get('java_vendor', 'unknown')}), "
        f"{ravel_metadata.get('os_name', 'unknown')} "
        f"{ravel_metadata.get('os_arch', 'unknown')}. NumPy "
        f"{numpy_metadata.get('numpy_version', 'unknown')} on Python "
        f"{numpy_metadata.get('python_version', 'unknown')}, "
        f"{numpy_metadata.get('platform', 'unknown')}.",
        "",
    ]
    lines.extend(summary_table(rows, "family", "Family summary"))
    lines.extend(summary_table(rows, "input_dtype", "Input dtype summary"))
    lines.extend(summary_table(rows, "input_layout", "Input layout summary"))
    lines.extend(
        [
            "## Detailed results",
            "",
            "| side | family | dtype | layout | case | work | Ravel ns/op | "
            "NumPy ns/op | Ravel speed |",
            "|---:|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        units = int(row["logical_work_units"])
        lines.append(
            f"| {row['side']} | {row['family']} | `{row['input_dtype']}` | "
            f"`{row['input_layout']}` | `{row['case']}` | {units:,} "
            f"{row['work_unit']} | {row['ravel_ns']:,.1f} | "
            f"{row['numpy_ns']:,.1f} | {row['speed']:.3f}x |"
        )
    lines.extend(
        [
            "",
            "A speed value above 1.0x favors Ravel. In-place rows reuse their "
            "destination; allocating array operations include result allocation. "
            "All other interpretation cautions from the access-pattern suite apply.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jmh", type=Path, help="JMH JSON; required unless parity-only")
    parser.add_argument("--numpy", required=True, type=Path)
    parser.add_argument("--signatures", required=True, type=Path)
    parser.add_argument("--out", type=Path, help="report path; required unless parity-only")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="render the measured intersection for smoke runs",
    )
    parser.add_argument(
        "--parity-only",
        action="store_true",
        help="validate semantic signatures without JMH timings",
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
            f"Operation matrix parity OK: {len(signatures)} matched cases "
            f"({sorted({side for _, side in signatures})})"
        )
        return
    if args.jmh is None or args.out is None:
        raise SystemExit("--jmh and --out are required unless --parity-only")
    jmh = read_jmh(load_json(args.jmh))
    signature_order = [
        (str(row["case"]), int(row["side"])) for row in signatures_payload["results"]
    ]
    rendered = report(
        jmh,
        signatures,
        signature_order,
        numpy,
        numpy_payload["metadata"],
        signatures_payload["metadata"],
        args.allow_partial,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered, encoding="utf-8")
    print(f"Wrote operation-matrix report to {args.out}")


if __name__ == "__main__":
    main()
