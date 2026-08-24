#!/usr/bin/env python3
"""Compare Ravel and NumPy elementwise common-ops signatures."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


METADATA_FIELDS = (
    "family",
    "input_dtype",
    "result_dtype",
    "input_layout",
    "comparison",
    "result_kind",
    "result_layout",
    "shape",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def index_results(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for entry in payload["results"]:
        name = str(entry["case"])
        if name in results:
            raise ValueError(f"duplicate common-ops case: {name}")
        results[name] = entry
    return results


def decode_number(value: Any) -> float:
    if value == "nan":
        return math.nan
    if value == "inf":
        return math.inf
    if value == "-inf":
        return -math.inf
    return float(value)


def values_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_values = left["values"]
    right_values = right["values"]
    if len(left_values) != len(right_values):
        return False
    if left["result_kind"] == "error" or right["result_kind"] == "error":
        return left_values == right_values
    if left["comparison"] == "bits":
        return left_values == right_values
    if left["comparison"] == "exact" and left["result_dtype"] != "float32":
        return left_values == right_values
    rel_tol = 2e-6 if left["result_dtype"] == "float32" else 2e-12
    abs_tol = 2e-5 if left["result_dtype"] == "float32" else 2e-12
    for left_value, right_value in zip(left_values, right_values, strict=True):
        if isinstance(left_value, bool) or isinstance(right_value, bool):
            if left_value != right_value:
                return False
            continue
        left_number = decode_number(left_value)
        right_number = decode_number(right_value)
        if math.isnan(left_number) and math.isnan(right_number):
            continue
        if math.isinf(left_number) or math.isinf(right_number):
            if left_number != right_number:
                return False
            continue
        if left["comparison"] == "exact" and left_number == right_number:
            continue
        if not math.isclose(
            left_number, right_number, rel_tol=rel_tol, abs_tol=abs_tol
        ):
            return False
    return True


def validate_signatures(
    ravel: dict[str, dict[str, Any]],
    numpy: dict[str, dict[str, Any]],
) -> None:
    if set(ravel) != set(numpy):
        missing_ravel = sorted(set(numpy) - set(ravel))
        missing_numpy = sorted(set(ravel) - set(numpy))
        raise ValueError(
            "common-ops case mismatch; "
            f"missing Ravel={missing_ravel}, missing NumPy={missing_numpy}"
        )
    for name in sorted(ravel):
        left = ravel[name]
        right = numpy[name]
        for field in METADATA_FIELDS:
            if left[field] != right[field]:
                raise ValueError(
                    f"parity failed for {name}: {field} "
                    f"Ravel={left[field]!r}, NumPy={right[field]!r}"
                )
        if not values_match(left, right):
            raise ValueError(
                f"parity failed for {name}: values "
                f"Ravel={left['values']!r}, NumPy={right['values']!r}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signatures", required=True, type=Path)
    parser.add_argument("--numpy", required=True, type=Path)
    args = parser.parse_args()
    ravel_payload = load_json(args.signatures)
    numpy_payload = load_json(args.numpy)
    ravel = index_results(ravel_payload)
    numpy = index_results(numpy_payload)
    validate_signatures(ravel, numpy)
    print(f"common-ops elementwise parity OK: {len(ravel)} matched cases")


if __name__ == "__main__":
    main()
