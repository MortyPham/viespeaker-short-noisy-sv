#!/usr/bin/env python3
"""Paired EER intervals for historical AAM, vector EMA, and KL EMA by enrollment speaker."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_curve


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/evaluation"
SCORES = {
    "AAM": BASE / "aam_only_full_viespeaker_h_1e1p/final_seed20260818_n5000/scores_aam_only_full.csv",
    "VECTOR_EMA": BASE / "c_vector_ema_viespeaker_h_1e1p/final_seed20260818_n5000/scores_c_vector_ema.csv",
    "GAUSSIAN_KL_EMA": BASE / "d_full_viespeaker_h_1e1p/final_seed20260818_n5000/scores_d_full.csv",
}
KEY = ["condition", "trial_id", "label", "enrollment", "probe"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def eer(labels: np.ndarray, scores: np.ndarray) -> float:
    far, tar, _ = roc_curve(labels, scores, pos_label=1)
    d = 1 - tar - far
    ix = np.flatnonzero(d[:-1] * d[1:] <= 0)
    if len(ix):
        i = int(ix[0])
        den = d[i] - d[i + 1]
        weight = d[i] / den if den else 0.0
        return 100 * float(far[i] + weight * (far[i + 1] - far[i]))
    i = int(np.argmin(np.abs(d)))
    return 50 * float(1 - tar[i] + far[i])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    if args.replicates < 1:
        raise ValueError("replicates must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    inputs = {}
    merged = None
    for name, path in SCORES.items():
        frame = pd.read_csv(path)
        if not set(KEY + ["score"]).issubset(frame.columns):
            raise ValueError(f"Missing required columns in {path}")
        if frame.duplicated(KEY).any() or not np.isfinite(frame.score.to_numpy()).all():
            raise ValueError(f"Duplicate keys or nonfinite scores in {path}")
        inputs[name] = {"path": str(path), "sha256": sha256(path), "rows": len(frame)}
        frame = frame[KEY + ["score"]].rename(columns={"score": name})
        merged = frame if merged is None else merged.merge(frame, on=KEY, how="inner", validate="one_to_one")
    merged = merged.sort_values(["condition", "trial_id"]).reset_index(drop=True)
    if len(merged) != 101610 or merged.condition.nunique() != 27:
        raise ValueError("Unexpected scored-row intersection")
    speaker = merged.enrollment.str.split("/", n=1).str[0]
    if speaker.isna().any() or not speaker.str.fullmatch(r"id\d+").all():
        raise ValueError("Enrollment speaker ID cannot be parsed")
    merged["enrollment_speaker"] = speaker
    rows = []
    rng = np.random.default_rng(args.seed)
    for condition, group in merged.groupby("condition", sort=True):
        labels = group.label.to_numpy(np.int64)
        values = group[list(SCORES)].to_numpy(np.float64)
        speakers, inverse = np.unique(group.enrollment_speaker.to_numpy(), return_inverse=True)
        members = [np.flatnonzero(inverse == i) for i in range(len(speakers))]
        observed = [eer(labels, values[:, i]) for i in range(3)]
        diffs = []
        for _ in range(args.replicates):
            selected = np.concatenate([members[i] for i in rng.integers(0, len(speakers), len(speakers))])
            sampled_labels = labels[selected]
            if sampled_labels.min() == sampled_labels.max():
                continue
            aam, vector, kl = [eer(sampled_labels, values[selected, i]) for i in range(3)]
            diffs.append((vector - aam, kl - aam))
        if len(diffs) < 0.9 * args.replicates:
            raise ValueError(f"Insufficient valid replicates for {condition}")
        diffs = np.asarray(diffs)
        for j, name in enumerate(("VECTOR_EMA", "GAUSSIAN_KL_EMA")):
            lo, hi = np.percentile(diffs[:, j], [2.5, 97.5])
            rows.append({
                "condition": condition, "candidate": name,
                "scored_trials": len(group), "target_trials": int(labels.sum()),
                "nontarget_trials": int((labels == 0).sum()),
                "exclusion_percent": 100 * (1 - len(group) / 5000),
                "enrollment_speaker_clusters": len(speakers),
                "aam_eer_percent": observed[0], "candidate_eer_percent": observed[j + 1],
                "candidate_minus_aam_pp": observed[j + 1] - observed[0],
                "ci_95_low_pp": lo, "ci_95_high_pp": hi,
                "ci_excludes_zero": bool(lo > 0 or hi < 0),
                "valid_replicates": len(diffs), "skipped_replicates": args.replicates - len(diffs),
            })
        print(f"{condition}: {len(speakers)} enrollment speakers, {len(diffs)} valid replicates", flush=True)
    with (args.output_dir / "paired_speaker_bootstrap.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "status": "completed", "research_question": "Do vector EMA or Gaussian-KL EMA change EER relative to AAM on identical sampled-H trials?",
        "command": " ".join(sys.argv), "seed": args.seed, "replicates_requested": args.replicates,
        "method": "paired percentile bootstrap of candidate-minus-AAM EER; resample enrollment speaker IDs parsed from enrollment paths within each condition; unadjusted 95% intervals",
        "cluster_limitation": "One-way enrollment-speaker clustering does not fully model dependence through probe speakers shared across enrollment-speaker clusters; official trial labels are retained even where filename prefixes disagree with labels.",
        "scored_rows": len(merged), "target_score_rows": int(merged.label.sum()),
        "nontarget_score_rows": int((merged.label == 0).sum()),
        "trial_manifest_sha256": "97cd1ff475591682c865a0e50b36e3d78d1f86d8c3f14c6e6ce7b4d8076ac83b",
        "initialization": "none; saved scores only", "checkpoints": "none loaded", "data_paths_and_hashes": inputs,
        "output_dir": str(args.output_dir.resolve()), "software": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        },
        "active_processes": "CPU analysis only; GPU unchanged",
        "limitations": ["No training-seed uncertainty", "No multiple-comparison correction", "One-way enrollment-speaker bootstrap"],
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
