#!/usr/bin/env python3
"""Compare sampled-H audio_quality.csv files without modifying either input.

The comparison is intentionally specific to the evaluator's audio-quality
schema. Raw file hashes are retained, while scientific equality is defined by
identical row keys, exact integer/categorical fields, prepared/cached success
normalization, and bit-exact IEEE float32 values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any


EXPECTED_COLUMNS = (
    "path",
    "source_seconds",
    "voiced_seconds",
    "vad_segments",
    "status",
)
KEY_COLUMN = "path"
FLOAT_COLUMNS = ("source_seconds", "voiced_seconds")
INTEGER_COLUMNS = ("vad_segments",)
SUCCESS_STATUSES = frozenset(("prepared", "cached"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def float32_bits(text: str) -> tuple[float, int, str]:
    value64 = float(text)
    if not math.isfinite(value64):
        raise ValueError(f"non-finite floating-point value {text!r}")
    packed = struct.pack(">f", value64)
    value32 = struct.unpack(">f", packed)[0]
    bits = struct.unpack(">I", packed)[0]
    return value32, bits, packed.hex()


def ordered_float32_int(bits: int) -> int:
    if bits & 0x80000000:
        return 0x80000000 - (bits & 0x7FFFFFFF)
    return 0x80000000 + bits


def ulp_distance(left_bits: int, right_bits: int) -> int:
    return abs(ordered_float32_int(left_bits) - ordered_float32_int(right_bits))


def load_rows(path: Path) -> tuple[dict[str, dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        if columns != EXPECTED_COLUMNS:
            raise ValueError(
                f"{path}: columns {columns!r} do not equal expected {EXPECTED_COLUMNS!r}"
            )
        rows: dict[str, dict[str, str]] = {}
        duplicates: list[str] = []
        for row in reader:
            key = row[KEY_COLUMN]
            if key in rows:
                duplicates.append(key)
            else:
                rows[key] = row
    return rows, sorted(set(duplicates))


def canonical_row(row: dict[str, str]) -> dict[str, Any]:
    status = row["status"]
    if status not in SUCCESS_STATUSES:
        raise ValueError(f"{row[KEY_COLUMN]}: unexpected non-success status {status!r}")
    return {
        "path": row["path"],
        "source_seconds_float32_hex": float32_bits(row["source_seconds"])[2],
        "voiced_seconds_float32_hex": float32_bits(row["voiced_seconds"])[2],
        "vad_segments": int(row["vad_segments"]),
        "status": "success",
    }


def canonical_hash(rows: dict[str, dict[str, str]]) -> str:
    digest = hashlib.sha256()
    for key in sorted(rows):
        payload = json.dumps(
            canonical_row(rows[key]),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        digest.update(payload.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def compare(reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    reference, reference_duplicates = load_rows(reference_path)
    candidate, candidate_duplicates = load_rows(candidate_path)
    reference_keys = set(reference)
    candidate_keys = set(candidate)
    missing = sorted(reference_keys - candidate_keys)
    additional = sorted(candidate_keys - reference_keys)

    status_differences: list[dict[str, str]] = []
    unexpected_statuses: list[dict[str, str]] = []
    exact_mismatches: list[dict[str, Any]] = []
    float_mismatches: dict[str, list[dict[str, Any]]] = {
        column: [] for column in FLOAT_COLUMNS
    }

    for key in sorted(reference_keys & candidate_keys):
        left = reference[key]
        right = candidate[key]

        for side, status in (("reference", left["status"]), ("candidate", right["status"])):
            if status not in SUCCESS_STATUSES:
                unexpected_statuses.append(
                    {"row_id": key, "side": side, "status": status}
                )
        if left["status"] != right["status"]:
            status_differences.append(
                {
                    "row_id": key,
                    "reference_status": left["status"],
                    "candidate_status": right["status"],
                    "normalized_status": (
                        "success"
                        if left["status"] in SUCCESS_STATUSES
                        and right["status"] in SUCCESS_STATUSES
                        else "not_equivalent"
                    ),
                }
            )

        for column in INTEGER_COLUMNS:
            try:
                left_value = int(left[column])
                right_value = int(right[column])
            except ValueError as exc:
                raise ValueError(f"{key}: invalid integer in {column}") from exc
            if left_value != right_value:
                exact_mismatches.append(
                    {
                        "row_id": key,
                        "column": column,
                        "reference_value": left[column],
                        "candidate_value": right[column],
                    }
                )

        for column in FLOAT_COLUMNS:
            try:
                left_value, left_bits, _ = float32_bits(left[column])
                right_value, right_bits, _ = float32_bits(right[column])
            except (OverflowError, ValueError) as exc:
                raise ValueError(f"{key}: invalid float in {column}") from exc
            if left_bits != right_bits:
                absolute = abs(right_value - left_value)
                denominator = max(abs(left_value), abs(right_value))
                relative = absolute / denominator if denominator else 0.0
                float_mismatches[column].append(
                    {
                        "row_id": key,
                        "reference_text": left[column],
                        "candidate_text": right[column],
                        "reference_float32": left_value,
                        "candidate_float32": right_value,
                        "absolute_difference": absolute,
                        "relative_difference": relative,
                        "float32_ulp_difference": ulp_distance(left_bits, right_bits),
                    }
                )

    float_summary = {}
    for column, mismatches in float_mismatches.items():
        float_summary[column] = {
            "mismatch_count": len(mismatches),
            "maximum_absolute_difference": max(
                (item["absolute_difference"] for item in mismatches), default=0.0
            ),
            "maximum_relative_difference": max(
                (item["relative_difference"] for item in mismatches), default=0.0
            ),
            "maximum_float32_ulp_difference": max(
                (item["float32_ulp_difference"] for item in mismatches), default=0
            ),
            "rows": mismatches,
        }

    reference_canonical_hash = None
    candidate_canonical_hash = None
    if not reference_duplicates and not candidate_duplicates:
        try:
            reference_canonical_hash = canonical_hash(reference)
            candidate_canonical_hash = canonical_hash(candidate)
        except ValueError:
            pass

    passed = not any(
        (
            reference_duplicates,
            candidate_duplicates,
            missing,
            additional,
            unexpected_statuses,
            exact_mismatches,
            *(summary["rows"] for summary in float_summary.values()),
        )
    ) and reference_canonical_hash == candidate_canonical_hash

    return {
        "semantic_parity_passed": passed,
        "rule": {
            "schema": list(EXPECTED_COLUMNS),
            "row_key": KEY_COLUMN,
            "row_order_ignored": True,
            "success_statuses_normalized_to_success": sorted(SUCCESS_STATUSES),
            "integer_fields_exact": list(INTEGER_COLUMNS),
            "floating_fields": list(FLOAT_COLUMNS),
            "floating_comparison": "parse decimal, cast to IEEE float32, require identical bits",
            "looser_tolerance": None,
        },
        "reference": {
            "path": str(reference_path.resolve()),
            "raw_sha256": sha256_file(reference_path),
            "rows": len(reference),
            "canonical_sha256": reference_canonical_hash,
            "duplicate_keys": reference_duplicates,
        },
        "candidate": {
            "path": str(candidate_path.resolve()),
            "raw_sha256": sha256_file(candidate_path),
            "rows": len(candidate),
            "canonical_sha256": candidate_canonical_hash,
            "duplicate_keys": candidate_duplicates,
        },
        "missing_keys": missing,
        "additional_keys": additional,
        "normalized_status_differences": status_differences,
        "unexpected_statuses": unexpected_statuses,
        "exact_field_mismatches": exact_mismatches,
        "floating_field_comparison": float_summary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = compare(args.reference, args.candidate)
    except (OSError, ValueError) as exc:
        report = {
            "semantic_parity_passed": False,
            "checker_error": f"{type(exc).__name__}: {exc}",
        }
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    sys.stdout.write(serialized)
    return 0 if report.get("semantic_parity_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
