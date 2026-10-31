#!/usr/bin/env python3
"""Validate one sampled-H output against the canonical ARCH_A population."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

EXPECTED_FILES = {
    "fixed_trials.csv", "protocol_config.json", "extraction_summary.json",
    "audio_quality.csv", "audio_preparation_failures.csv",
    "probe_transform_manifest.csv", "embedding_cache_index.json",
    "embedding_failures.csv", "summary_metrics.csv",
    "valid_trials_by_condition.csv", "confidence_intervals.csv",
    "eer_by_duration_and_snr.png", "run_metadata.json",
}
ROW_FIELDS = [
    "trial_id", "label", "enrollment", "probe", "condition",
    "enrollment_duration_s", "probe_duration_s", "duration_s", "snr_db",
]
NUMERIC_METRICS = [
    "eer_percent", "roc_auc", "tar_at_far_1_percent", "min_dcf",
    "scored_trials", "target_trials", "nontarget_trials", "trial_exclusion_percent",
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-sha", required=True)
    parser.add_argument("--enrollment-encoder", required=True)
    parser.add_argument("--probe-encoder", required=True)
    parser.add_argument("--parity-checker", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--require-a-score-reproduction", action="store_true")
    args = parser.parse_args()
    errors: list[str] = []
    details: dict[str, object] = {}
    missing = sorted(name for name in EXPECTED_FILES if not (args.run_dir / name).is_file())
    if missing:
        errors.append(f"missing artifacts: {missing}")
    if errors:
        report = {"model": args.model, "passed": False, "errors": errors}
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        return 1

    metadata = json.loads((args.run_dir / "run_metadata.json").read_text())
    expected_meta = {
        "model": args.model, "checkpoint_sha256": args.checkpoint_sha,
        "protocol_sha256": "97cd1ff475591682c865a0e50b36e3d78d1f86d8c3f14c6e6ce7b4d8076ac83b",
        "official_trial_sha256": "298addeecd574cf1b601f69e2c0971df9173b36cd741e6981846f6b4950793b0",
        "requested_trials": 5000, "bootstrap_replicates": 300,
        "device": "cuda",
    }
    for key, value in expected_meta.items():
        if metadata.get(key) != value:
            errors.append(f"metadata {key}: expected {value!r}, got {metadata.get(key)!r}")
    if len(metadata.get("conditions", [])) != 27 or len(set(metadata.get("conditions", []))) != 27:
        errors.append("run metadata does not contain 27 unique conditions")

    fixed_sha = sha(args.run_dir / "fixed_trials.csv")
    valid_sha = sha(args.run_dir / "valid_trials_by_condition.csv")
    failure_sha = sha(args.run_dir / "audio_preparation_failures.csv")
    transform_sha = sha(args.run_dir / "probe_transform_manifest.csv")
    expected_hashes = {
        "fixed_trials": "97cd1ff475591682c865a0e50b36e3d78d1f86d8c3f14c6e6ce7b4d8076ac83b",
        "valid_trials": "8bdea0482655a471217964dc679a0e408bbba3ba0d21e4af57c758bfa92c6bdb",
        "failures": "e27f89f139a828b4d91f6773f9f42bbffe2dd31dffe68ed8fe97af6b9051e4fe",
        "transforms": "c86e8a1e6e2b59df84cbe16fb11ed30b524e95cefc564c0c0ffc8cd00202f8e1",
    }
    actual_hashes = {"fixed_trials": fixed_sha, "valid_trials": valid_sha,
                     "failures": failure_sha, "transforms": transform_sha}
    for key, value in expected_hashes.items():
        if actual_hashes[key] != value:
            errors.append(f"{key} hash mismatch: {actual_hashes[key]}")

    parity = subprocess.run(
        [sys.executable, str(args.parity_checker), "--reference",
         str(args.canonical_dir / "audio_quality.csv"), "--candidate",
         str(args.run_dir / "audio_quality.csv")],
        text=True, capture_output=True,
    )
    try:
        parity_json = json.loads(parity.stdout)
    except json.JSONDecodeError:
        parity_json = {"parse_error": parity.stdout, "stderr": parity.stderr}
    if parity.returncode or not parity_json.get("semantic_parity_passed"):
        errors.append("audio-quality semantic parity failed")
    if parity_json.get("candidate", {}).get("canonical_sha256") != "d8ffd41fecd799b7bb75e9e0651bc21e38479e9cb4742445f2b05809876b988a":
        errors.append("audio-quality canonical hash mismatch")

    score_names = {
        "LEGACY_C_VECTOR_EMA": "scores_c_vector_ema.csv",
        "LEGACY_D_FULL_KL_EMA": "scores_d_full.csv",
    }
    score_path = args.run_dir / score_names.get(args.model, "scores_aam_only_full.csv")
    if not score_path.is_file():
        errors.append(f"missing native score file: {score_path.name}")
        scores = []
    else:
        scores = table(score_path)
    canonical = table(args.canonical_dir / "scores_aam_only_full.csv")
    score_rows = [tuple(row[field] for field in ROW_FIELDS) for row in scores]
    canonical_rows = [tuple(row[field] for field in ROW_FIELDS) for row in canonical]
    if len(scores) != 101619 or score_rows != canonical_rows:
        errors.append("ordered score-row population differs from canonical ARCH_A")
    if len(score_rows) != len(set(score_rows)):
        errors.append("duplicate scientific score rows")
    if any(row.get("model") != args.model for row in scores):
        errors.append("score file has wrong model name")
    if args.require_a_score_reproduction:
        candidate_scores = [row["score"] for row in scores]
        reference_scores = [row["score"] for row in canonical]
        if candidate_scores != reference_scores:
            max_diff = max(abs(float(a) - float(b)) for a, b in zip(candidate_scores, reference_scores))
            errors.append(f"ARCH_A score reproduction is not decimal-exact; max_abs={max_diff}")

    summaries = table(args.run_dir / "summary_metrics.csv")
    if len(summaries) != 27 or len({row["condition"] for row in summaries}) != 27:
        errors.append("summary does not contain 27 unique conditions")
    for row in summaries:
        if row.get("enrollment_encoder") != args.enrollment_encoder:
            errors.append(f"wrong enrollment encoder in summary: {row.get('enrollment_encoder')}")
        if row.get("probe_encoder") != args.probe_encoder:
            errors.append(f"wrong probe encoder in summary: {row.get('probe_encoder')}")
        for field in NUMERIC_METRICS:
            try:
                if not math.isfinite(float(row[field])):
                    raise ValueError
            except (KeyError, ValueError):
                errors.append(f"non-finite or absent metric {field} in {row.get('condition')}")
    embedding_failures = (args.run_dir / "embedding_failures.csv").read_text().strip()
    if embedding_failures:
        rows = table(args.run_dir / "embedding_failures.csv")
        if rows:
            errors.append(f"embedding failures present: {len(rows)}")

    details.update({"rows": len(scores), "conditions": len(summaries),
                    "hashes": actual_hashes, "audio_quality": parity_json,
                    "checkpoint_sha256": metadata.get("checkpoint_sha256")})
    report = {"model": args.model, "passed": not errors, "errors": errors, "details": details}
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
