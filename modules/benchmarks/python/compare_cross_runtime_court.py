#!/usr/bin/env python3
"""Validate JVM/Scala.js court parity and compare same-host Node receipts."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROW_KEY_FIELDS = ("case", "side")
SEMANTIC_METADATA_FIELDS = (
    "family",
    "input_dtype",
    "result_dtype",
    "input_layout",
    "logical_work_units",
    "work_unit",
    "comparison",
    "timing_scope",
    "result_size",
    "result_layout",
)
EXACT_SIGNATURE_FIELDS = (
    "nan_count",
    "positive_infinity_count",
    "negative_infinity_count",
    "negative_zero_count",
)
PINNED_RUNTIME_FIELDS = (
    "runtime_name",
    "runtime_version",
    "vm_name",
    "scala_version",
    "os_name",
    "os_arch",
    "available_processors",
    "timer",
    "explicit_gc_available",
    "sides",
    "warmup_ms",
    "sample_ms",
    "samples",
    "case_filter",
)
MINIMUM_TIMING_SAMPLES = 5


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "ravel.cross-runtime-court.v1":
        raise ValueError(
            f"unsupported court schema in {path}: {payload.get('schema')!r}"
        )
    return payload


def row_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row[ROW_KEY_FIELDS[0]]), int(row[ROW_KEY_FIELDS[1]])


def index_results(payload: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in payload["results"]:
        key = row_key(row)
        if key in indexed:
            raise ValueError(f"duplicate court row: {key}")
        indexed[key] = row
    return indexed


def signatures_match(left: dict[str, Any], right: dict[str, Any], field: str) -> bool:
    left_value = float(left[field])
    right_value = float(right[field])
    if left["comparison"] == "exact":
        return left_value == right_value
    if left["result_dtype"] == "float32":
        return math.isclose(left_value, right_value, rel_tol=2e-6, abs_tol=2e-3)
    return math.isclose(left_value, right_value, rel_tol=2e-10, abs_tol=2e-7)


def validate_semantic_parity(
    jvm_payload: dict[str, Any], js_payload: dict[str, Any]
) -> None:
    jvm = index_results(jvm_payload)
    js = index_results(js_payload)
    if set(jvm) != set(js):
        missing_jvm = sorted(set(js) - set(jvm))
        missing_js = sorted(set(jvm) - set(js))
        raise ValueError(
            f"cross-runtime case mismatch; missing JVM={missing_jvm}, "
            f"missing Scala.js={missing_js}"
        )
    for key in sorted(jvm):
        left = jvm[key]
        right = js[key]
        for field in SEMANTIC_METADATA_FIELDS:
            if left[field] != right[field]:
                raise ValueError(
                    f"cross-runtime parity failed for {key}: {field} "
                    f"JVM={left[field]!r}, Scala.js={right[field]!r}"
                )
        for field in EXACT_SIGNATURE_FIELDS:
            if int(left[field]) != int(right[field]):
                raise ValueError(
                    f"cross-runtime parity failed for {key}: {field} "
                    f"JVM={left[field]!r}, Scala.js={right[field]!r}"
                )
        for field in ("sum", "weighted_sum"):
            if not signatures_match(left, right, field):
                raise ValueError(
                    f"cross-runtime parity failed for {key}: {field} "
                    f"JVM={left[field]!r}, Scala.js={right[field]!r}"
                )


def validate_same_host(
    baseline_payload: dict[str, Any], candidate_payload: dict[str, Any]
) -> None:
    validate_timed_node_receipt(baseline_payload)
    validate_timed_node_receipt(candidate_payload)
    baseline = baseline_payload["metadata"]
    candidate = candidate_payload["metadata"]
    for field in PINNED_RUNTIME_FIELDS:
        if baseline.get(field) != candidate.get(field):
            raise ValueError(
                f"Node receipts are not comparable: {field} "
                f"baseline={baseline.get(field)!r}, candidate={candidate.get(field)!r}"
            )


def validate_timed_node_receipt(
    payload: dict[str, Any],
    minimum_samples: int = MINIMUM_TIMING_SAMPLES,
) -> None:
    metadata = payload["metadata"]
    if metadata.get("runtime_name") != "node":
        raise ValueError("timing evidence accepts a Node receipt only")
    if metadata.get("mode") != "timed":
        raise ValueError("timing evidence requires a timed Node receipt")
    configured_samples = int(metadata.get("samples", 0))
    if configured_samples < minimum_samples:
        raise ValueError(
            f"Node receipt has {configured_samples} samples; "
            f"at least {minimum_samples} are required"
        )
    measured = 0
    for key, row in index_results(payload).items():
        if row.get("timing_status") != "measured":
            continue
        measured += 1
        samples = row.get("samples_ns")
        if not isinstance(samples, list) or len(samples) != configured_samples:
            raise ValueError(
                f"Node sample-count mismatch for {key}: "
                f"metadata={configured_samples}, row="
                f"{len(samples) if isinstance(samples, list) else 'missing'}"
            )
        if any(float(sample) <= 0.0 for sample in samples):
            raise ValueError(f"non-positive Node timing sample for {key}")
        numeric_samples = [float(sample) for sample in samples]
        expected_median = statistics.median(numeric_samples)
        reported_median = float(row["median_ns"])
        if not math.isclose(
            reported_median,
            expected_median,
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise ValueError(
                f"Node median mismatch for {key}: "
                f"reported={reported_median}, samples={expected_median}"
            )
        mean = statistics.fmean(numeric_samples)
        expected_rsd = statistics.pstdev(numeric_samples) / mean
        reported_rsd = float(row["relative_standard_deviation"])
        if not math.isclose(
            reported_rsd,
            expected_rsd,
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"Node RSD mismatch for {key}: "
                f"reported={reported_rsd}, samples={expected_rsd}"
            )
    if measured == 0:
        raise ValueError("Node receipt contains no measured rows")


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def summarize_node_receipt(
    payload: dict[str, Any],
    instability_limit: float,
    minimum_samples: int = MINIMUM_TIMING_SAMPLES,
) -> tuple[list[dict[str, Any]], str]:
    validate_timed_node_receipt(payload, minimum_samples)
    metadata = payload["metadata"]

    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key, source in sorted(index_results(payload).items()):
        if source.get("timing_status") != "measured":
            continue
        median_ns = float(source["median_ns"])
        work_units = int(source["logical_work_units"])
        relative_standard_deviation = float(source["relative_standard_deviation"])
        if median_ns <= 0.0 or work_units <= 0:
            raise ValueError(f"non-positive Node timing or work count for {key}")
        if relative_standard_deviation < 0.0:
            raise ValueError(f"negative Node timing variability for {key}")
        row = {
            "case": key[0],
            "side": key[1],
            "family": source["family"],
            "median_ns": median_ns,
            "ns_per_work_unit": median_ns / work_units,
            "relative_standard_deviation": relative_standard_deviation,
            "unstable": relative_standard_deviation > instability_limit,
        }
        rows.append(row)
        grouped[str(row["family"])].append(row)
    lines = [
        "# Ravel Scala.js same-host baseline",
        "",
        "This is an absolute full-linked Node baseline, not a comparison with JVM "
        "or NumPy. A future candidate is comparable only when the pinned runtime "
        "metadata matches.",
        "",
        f"Each row contains {metadata['samples']} measured samples.",
        "",
        "## Family summary",
        "",
        "| family | rows | geometric mean ns/work | maximum RSD | unstable rows |",
        "|---|---:|---:|---:|---:|",
    ]
    for family in sorted(grouped):
        family_rows = grouped[family]
        normalized = [float(row["ns_per_work_unit"]) for row in family_rows]
        maximum_rsd = max(
            float(row["relative_standard_deviation"]) for row in family_rows
        )
        unstable = sum(bool(row["unstable"]) for row in family_rows)
        lines.append(
            f"| `{family}` | {len(family_rows)} | "
            f"{geometric_mean(normalized):,.3f} | {maximum_rsd:.1%} | {unstable} |"
        )
    lines.extend(
        [
            "",
            "## Every measured row",
            "",
            "| side | family | case | median ns/op | ns/work | RSD | status |",
            "|---:|---|---|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        status = "UNSTABLE" if row["unstable"] else "ok"
        lines.append(
            f"| {row['side']} | `{row['family']}` | `{row['case']}` | "
            f"{row['median_ns']:,.1f} | {row['ns_per_work_unit']:,.3f} | "
            f"{row['relative_standard_deviation']:.1%} | {status} |"
        )
    lines.extend(
        [
            "",
            f"Rows with relative standard deviation above {instability_limit:.0%} "
            "are marked unstable and cannot support a performance claim.",
            "",
        ]
    )
    return rows, "\n".join(lines)


def compare_node_timings(
    baseline_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
    regression_limit: float,
    instability_limit: float = 0.10,
) -> tuple[list[dict[str, Any]], str]:
    validate_same_host(baseline_payload, candidate_payload)
    baseline = {
        key: row
        for key, row in index_results(baseline_payload).items()
        if row.get("timing_status") == "measured"
    }
    candidate = {
        key: row
        for key, row in index_results(candidate_payload).items()
        if row.get("timing_status") == "measured"
    }
    if set(baseline) != set(candidate):
        missing_baseline = sorted(set(candidate) - set(baseline))
        missing_candidate = sorted(set(baseline) - set(candidate))
        raise ValueError(
            f"measured Node case mismatch; missing baseline={missing_baseline}, "
            f"missing candidate={missing_candidate}"
        )

    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key in sorted(baseline):
        baseline_ns = float(baseline[key]["median_ns"])
        candidate_ns = float(candidate[key]["median_ns"])
        if baseline_ns <= 0.0 or candidate_ns <= 0.0:
            raise ValueError(f"non-positive Node timing for {key}")
        speedup = baseline_ns / candidate_ns
        row = {
            "case": key[0],
            "side": key[1],
            "family": candidate[key]["family"],
            "baseline_ns": baseline_ns,
            "candidate_ns": candidate_ns,
            "baseline_rsd": float(baseline[key]["relative_standard_deviation"]),
            "candidate_rsd": float(candidate[key]["relative_standard_deviation"]),
            "speedup": speedup,
            "regression": speedup < 1.0 - regression_limit,
        }
        row["unstable"] = (
            row["baseline_rsd"] > instability_limit
            or row["candidate_rsd"] > instability_limit
        )
        rows.append(row)
        grouped[str(row["family"])].append(row)

    lines = [
        "# Ravel Scala.js same-host performance court",
        "",
        "JVM/Scala.js semantic parity is a separate mandatory gate. This report "
        "compares full-linked Node receipts from an identical runtime and host.",
        "",
        "## Family summary",
        "",
        "| family | rows | geometric mean | minimum | maximum | unstable rows |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for family in sorted(grouped):
        family_rows = grouped[family]
        values = [float(row["speedup"]) for row in family_rows]
        unstable = sum(bool(row["unstable"]) for row in family_rows)
        lines.append(
            f"| `{family}` | {len(values)} | {geometric_mean(values):.3f}x | "
            f"{min(values):.3f}x | {max(values):.3f}x | {unstable} |"
        )
    lines.extend(
        [
            "",
            "## Every measured row",
            "",
            "| side | family | case | baseline ns/op | candidate ns/op | "
            "baseline RSD | candidate RSD | speedup | status |",
            "|---:|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        statuses = []
        if row["regression"]:
            statuses.append("REGRESSION")
        if row["unstable"]:
            statuses.append("UNSTABLE")
        status = "; ".join(statuses) if statuses else "ok"
        lines.append(
            f"| {row['side']} | `{row['family']}` | `{row['case']}` | "
            f"{row['baseline_ns']:,.1f} | {row['candidate_ns']:,.1f} | "
            f"{row['baseline_rsd']:.1%} | {row['candidate_rsd']:.1%} | "
            f"{row['speedup']:.3f}x | {status} |"
        )
    lines.append("")
    return rows, "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jvm", required=True, type=Path)
    parser.add_argument("--js", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument(
        "--summarize",
        type=Path,
        help="render one absolute Node baseline with per-row variability",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--regression-limit",
        type=float,
        default=0.10,
        help="fractional per-row regression flagged in a timing report",
    )
    parser.add_argument(
        "--instability-limit",
        type=float,
        default=0.10,
        help="relative standard deviation visibly flagged in a baseline summary",
    )
    parser.add_argument(
        "--minimum-samples",
        type=int,
        default=MINIMUM_TIMING_SAMPLES,
        help="minimum per-row timing samples accepted as performance evidence",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.regression_limit < 1.0:
        raise SystemExit("--regression-limit must be in [0, 1)")
    if not 0.0 <= args.instability_limit < 1.0:
        raise SystemExit("--instability-limit must be in [0, 1)")
    if args.minimum_samples < 2:
        raise SystemExit("--minimum-samples must be at least 2")
    jvm = load_json(args.jvm)
    js = load_json(args.js)
    validate_semantic_parity(jvm, js)
    print(f"Cross-runtime semantic parity OK: {len(index_results(jvm))} matched rows")

    if args.summarize is not None:
        if args.baseline is not None or args.candidate is not None or args.out is None:
            raise SystemExit(
                "--summarize and --out must be supplied without "
                "--baseline or --candidate"
            )
        receipt = load_json(args.summarize)
        rows, report = summarize_node_receipt(
            receipt,
            instability_limit=args.instability_limit,
            minimum_samples=args.minimum_samples,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        unstable = sum(bool(row["unstable"]) for row in rows)
        print(f"Wrote Node baseline summary to {args.out}; unstable rows={unstable}")
        return

    timing_options = (args.baseline, args.candidate, args.out)
    if all(option is None for option in timing_options):
        return
    if any(option is None for option in timing_options):
        raise SystemExit("--baseline, --candidate, and --out must be supplied together")
    baseline = load_json(args.baseline)
    candidate = load_json(args.candidate)
    rows, report = compare_node_timings(
        baseline,
        candidate,
        regression_limit=args.regression_limit,
        instability_limit=args.instability_limit,
    )
    assert args.out is not None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    regressions = sum(bool(row["regression"]) for row in rows)
    print(f"Wrote Node comparison to {args.out}; visible regressions={regressions}")


if __name__ == "__main__":
    main()
