#!/usr/bin/env python3
"""Build paired six-system outputs and durable documentation after supervision."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

PROJECT = Path("/media/ml4u/Samsung_T52/ptnk_team/mortyhuaphuoc")
SUPERVISOR = PROJECT / "experiments/evaluation/supervisors/sampledh_multimodel_resume_20260925_120132"
RESULT_ROOT = PROJECT / "experiments/evaluation/sampledh_multimodel_20260922"
DOCS = PROJECT / "docs/research"
RUNS = {
    "ARCH_A_ORIGINAL": RESULT_ROOT / "ARCH_A_ORIGINAL/final_seed20260818_n5000",
    "ARCH_B_GLOBAL_LOCAL": PROJECT / "experiments/evaluation/ARCH_B_GLOBAL_LOCAL_sampledH_seed20260818_spec20260921_retry1/final_seed20260818_n5000",
    "ARCH_C_RELATIVE_LENGTH_GATE": RESULT_ROOT / "ARCH_C_RELATIVE_LENGTH_GATE/final_seed20260818_n5000",
    "LEGACY_B_AAM_ONLY": RESULT_ROOT / "LEGACY_B_AAM_ONLY/final_seed20260818_n5000",
    "LEGACY_C_VECTOR_EMA": RESULT_ROOT / "LEGACY_C_VECTOR_EMA/final_seed20260818_n5000",
    "LEGACY_D_FULL_KL_EMA": RESULT_ROOT / "LEGACY_D_FULL_KL_EMA/final_seed20260818_n5000",
}
SCORE_FILES = {
    "LEGACY_C_VECTOR_EMA": "scores_c_vector_ema.csv",
    "LEGACY_D_FULL_KL_EMA": "scores_d_full.csv",
}
TRAINING = {
    "ARCH_A_ORIGINAL": "V2 cleaned, 3,359 speakers",
    "ARCH_B_GLOBAL_LOCAL": "V2 cleaned, 3,359 speakers",
    "ARCH_C_RELATIVE_LENGTH_GATE": "V2 cleaned, 3,359 speakers",
    "LEGACY_B_AAM_ONLY": "V1 historical, 3,371 speakers",
    "LEGACY_C_VECTOR_EMA": "V1 historical, 3,371 speakers",
    "LEGACY_D_FULL_KL_EMA": "V1 historical, 3,371 speakers",
}
ENCODERS = {
    "ARCH_A_ORIGINAL": "student/student",
    "ARCH_B_GLOBAL_LOCAL": "student/student",
    "ARCH_C_RELATIVE_LENGTH_GATE": "student/student",
    "LEGACY_B_AAM_ONLY": "student/student",
    "LEGACY_C_VECTOR_EMA": "EMA teacher enrollment/student probe",
    "LEGACY_D_FULL_KL_EMA": "EMA teacher enrollment/student probe",
}


def eer(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    difference = fnr - fpr
    crossings = np.flatnonzero(difference[:-1] * difference[1:] <= 0)
    if len(crossings):
        i = int(crossings[0]); denominator = difference[i] - difference[i + 1]
        weight = difference[i] / denominator if denominator else 0.0
        value = fpr[i] + weight * (fpr[i + 1] - fpr[i])
    else:
        i = int(np.argmin(np.abs(difference))); value = 0.5 * (fnr[i] + fpr[i])
    return 100.0 * float(value)


def validated_models() -> list[str]:
    statuses = pd.read_csv(SUPERVISOR / "model_status.csv")
    return [row.model for row in statuses.itertuples()
            if row.status == "COMPLETED_AND_PARITY_VERIFIED" and row.model in RUNS]


def paired_bootstrap(candidate: pd.DataFrame, baseline: pd.DataFrame, model: str,
                     base_name: str, condition: str, reps: int = 1000) -> dict[str, object]:
    scientific = ["trial_id", "label", "enrollment", "probe", "condition",
                  "enrollment_duration_s", "probe_duration_s", "duration_s", "snr_db"]
    if not candidate[scientific].equals(baseline[scientific]):
        raise RuntimeError(f"paired population differs: {model} vs {base_name} {condition}")
    labels = candidate.label.to_numpy(np.int64)
    c_scores = candidate.score.to_numpy(float); b_scores = baseline.score.to_numpy(float)
    enrollments = candidate.enrollment.to_numpy(str); clusters = np.unique(enrollments)
    indices = {key: np.flatnonzero(enrollments == key) for key in clusters}
    seed_text = f"{model}|{base_name}|{condition}|20260818"
    seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16) % (2**32 - 1)
    rng = np.random.default_rng(seed); values = []
    for _ in range(reps):
        sampled = rng.choice(clusters, len(clusters), replace=True)
        take = np.concatenate([indices[key] for key in sampled])
        if np.unique(labels[take]).size == 2:
            values.append(eer(labels[take], c_scores[take]) - eer(labels[take], b_scores[take]))
    observed = eer(labels, c_scores) - eer(labels, b_scores)
    return {"model": model, "baseline": base_name, "condition": condition,
            "eer_difference_pp": observed, "bootstrap_replicates": len(values),
            "ci_low": float(np.quantile(values, .025)),
            "ci_high": float(np.quantile(values, .975)),
            "bootstrap_unit": "enrollment recording"}


def append_once(path: Path, marker: str, text: str) -> None:
    existing = path.read_text(encoding="utf-8")
    if marker not in existing:
        with path.open("a", encoding="utf-8") as stream:
            stream.write("\n" + text.rstrip() + "\n")


def main() -> int:
    models = validated_models()
    if "ARCH_A_ORIGINAL" not in models or "ARCH_B_GLOBAL_LOCAL" not in models:
        raise RuntimeError("canonical A/B must both pass parity before combined analysis")
    summaries: dict[str, pd.DataFrame] = {}
    scores: dict[str, pd.DataFrame] = {}
    combined = []
    for model in models:
        summary = pd.read_csv(RUNS[model] / "summary_metrics.csv")
        score = pd.read_csv(RUNS[model] / SCORE_FILES.get(model, "scores_aam_only_full.csv"), keep_default_na=False)
        summary["display_model"] = model
        summaries[model] = summary; scores[model] = score
        combined.append(summary)
    all_summary = pd.concat(combined, ignore_index=True)
    lookup_a = summaries["ARCH_A_ORIGINAL"].set_index("condition").eer_percent
    lookup_legacy = (summaries["LEGACY_B_AAM_ONLY"].set_index("condition").eer_percent
                     if "LEGACY_B_AAM_ONLY" in summaries else None)
    all_summary["eer_difference_vs_ARCH_A_pp"] = all_summary.apply(
        lambda row: row.eer_percent - lookup_a[row.condition], axis=1)
    all_summary["eer_difference_vs_LEGACY_B_pp"] = all_summary.apply(
        lambda row: math.nan if lookup_legacy is None else row.eer_percent - lookup_legacy[row.condition], axis=1)
    combined_path = SUPERVISOR / "combined_27_condition_results.csv"
    all_summary.to_csv(combined_path, index=False)

    paired_rows = []
    for baseline in ["ARCH_A_ORIGINAL", "LEGACY_B_AAM_ONLY"]:
        if baseline not in models: continue
        for model in models:
            if model == baseline: continue
            for condition in summaries[model].condition:
                paired_rows.append(paired_bootstrap(
                    scores[model][scores[model].condition == condition].reset_index(drop=True),
                    scores[baseline][scores[baseline].condition == condition].reset_index(drop=True),
                    model, baseline, condition))
    paired = pd.DataFrame(paired_rows)
    paired_path = SUPERVISOR / "paired_eer_differences_1000.csv"
    paired.to_csv(paired_path, index=False)

    aggregates: dict[str, object] = {}
    for model, frame in summaries.items():
        noise = frame.snr_db.fillna(-999).map(lambda x: "clean" if x == -999 else f"{x:g}dB")
        aggregates[model] = {
            "macro": {k: float(frame[k].mean()) for k in ["eer_percent", "tar_at_far_1_percent", "roc_auc", "min_dcf"]},
            "by_noise_eer": frame.assign(noise=noise).groupby("noise").eer_percent.mean().to_dict(),
            "by_enrollment_eer": frame.groupby("enrollment_duration_s").eer_percent.mean().to_dict(),
            "by_probe_eer": frame.groupby("probe_duration_s").eer_percent.mean().to_dict(),
            "training_population": TRAINING[model], "inference_encoder": ENCODERS[model],
        }
    wins = {}
    for model in models:
        if model == "ARCH_A_ORIGINAL": continue
        delta = all_summary[all_summary.display_model == model].eer_difference_vs_ARCH_A_pp
        wins[model] = {"wins": int((delta < -1e-12).sum()), "ties": int((delta.abs() <= 1e-12).sum()),
                       "losses": int((delta > 1e-12).sum())}
    b_pair = paired[(paired.model == "ARCH_B_GLOBAL_LOCAL") & (paired.baseline == "ARCH_A_ORIGINAL")]
    primary = b_pair[b_pair.condition.isin(["enroll3s_probe1.25s_clean", "enroll3s_probe1.25s_10dB", "enroll3s_probe1.25s_0dB"])]
    noisy = primary[~primary.condition.str.endswith("clean")]
    clean = primary[primary.condition.str.endswith("clean")]
    credible = len(noisy) == 2 and bool((noisy.ci_high < 0).all()) and len(clean) == 1 and float(clean.eer_difference_pp.iloc[0]) <= .5
    architecture_classification = "B CREDIBLY IMPROVES OVER A" if credible else "NO RELIABLE ARCHITECTURAL IMPROVEMENT"
    final = {"created_utc": datetime.now(timezone.utc).isoformat(), "models_included": models,
             "models_missing_or_failed": sorted(set(RUNS) - set(models)), "row_count_per_model": 101619,
             "paired_bootstrap_replicates": 1000, "aggregates": aggregates,
             "win_tie_loss_vs_ARCH_A": wins, "architecture_classification": architecture_classification,
             "combined_csv": str(combined_path), "paired_csv": str(paired_path)}
    (SUPERVISOR / "final_summary.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")

    macro_lines = ["| Model | EER % | TAR@FAR1% | ROC-AUC | minDCF | W/T/L vs A |",
                   "|---|---:|---:|---:|---:|---:|"]
    for model in models:
        a = aggregates[model]["macro"]; w = wins.get(model, {"wins": 0, "ties": 27, "losses": 0})
        macro_lines.append(f"| {model} | {a['eer_percent']:.6f} | {a['tar_at_far_1_percent']:.6f} | {a['roc_auc']:.6f} | {a['min_dcf']:.6f} | {w['wins']}/{w['ties']}/{w['losses']} |")
    status_text = (SUPERVISOR / "model_status.csv").read_text()
    results_doc = f"""# Multimodel sampled-H results

Generated automatically by the immutable `sampledh_multimodel_resume_20260925_120132` supervisor after importing the validated completed runs and evaluating only the three pending historical systems. All included systems have 101,619 ordered rows identical to canonical ARCH_A/ARCH_B. ARCH_C_RELATIVE_LENGTH_GATE uses a batch-relative unpadded-length fraction, not absolute duration.

## Final classification

**{architecture_classification}.** Point estimates alone were not used: `paired_eer_differences_1000.csv` uses paired enrollment-recording-cluster bootstrap resampling. Historical systems are evaluation-row matched but are not matched training controls because they use V1/3,371 speakers rather than V2/3,359.

## Macro results

{chr(10).join(macro_lines)}

Full 27-condition values: `{combined_path.relative_to(PROJECT)}`. Paired differences and 95% intervals: `{paired_path.relative_to(PROJECT)}`. Machine summary: `{(SUPERVISOR / 'final_summary.json').relative_to(PROJECT)}`. Exact source artifacts are the run paths recorded there.

## Model statuses

```csv
{status_text.rstrip()}
```
"""
    (DOCS / "MULTIMODEL_SAMPLED_H_RESULTS.md").write_text(results_doc)
    matrix = "# Multimodel dataset and protocol matrix\n\n| Model | Training population | Inference encoder | Evaluation population |\n|---|---|---|---|\n"
    for model in RUNS:
        evaluation = "newer frozen sampled-H; 5,000 nominal / 101,619 ordered condition-trial rows" if model in models else "not included: model did not pass"
        matrix += f"| {model} | {TRAINING[model]} | {ENCODERS[model]} | {evaluation} |\n"
    matrix += "\nAll included outputs require manifest SHA-256 `97cd1ff475591682c865a0e50b36e3d78d1f86d8c3f14c6e6ce7b4d8076ac83b`, valid-row SHA-256 `8bdea0482655a471217964dc679a0e408bbba3ba0d21e4af57c758bfa92c6bdb`, nine-failure SHA-256 `e27f89f139a828b4d91f6773f9f42bbffe2dd31dffe68ed8fe97af6b9051e4fe`, transform SHA-256 `c86e8a1e6e2b59df84cbe16fb11ed30b524e95cefc564c0c0ffc8cd00202f8e1`, and semantic audio canonical hash `d8ffd41fecd799b7bb75e9e0651bc21e38479e9cb4742445f2b05809876b988a`.\n"
    (DOCS / "MULTIMODEL_DATASET_AND_PROTOCOL_MATRIX.md").write_text(matrix)

    marker = "sampledh_multimodel_20260922 final"
    update = f"## {marker}\n\n**VERIFIED.** Immutable sequential evaluation finished with included models: {', '.join(models)}. Classification: **{architecture_classification}**. See `docs/research/MULTIMODEL_SAMPLED_H_RESULTS.md` and `{SUPERVISOR.relative_to(PROJECT)}/final_summary.json`."
    for name in ["PROJECT_STATE.md", "VERIFIED_RESULTS.md", "MODEL_LINEAGE.md", "OPEN_QUESTIONS.md"]:
        append_once(DOCS / name, marker, update)
    ledger = DOCS / "EXPERIMENT_LEDGER.csv"
    ledger_text = ledger.read_text()
    with ledger.open("a", newline="") as stream:
        writer = csv.writer(stream)
        for model in RUNS:
            run_id = f"sampledh_multimodel_20260922_{model.lower()}"
            if run_id in ledger_text: continue
            status = "completed_and_parity_verified" if model in models else "not_completed_or_parity_failed"
            writer.writerow([run_id, status, model, "documented_checkpoint", "immutable_sampledH_evaluator",
                             TRAINING[model].replace(",", ";"), "evaluation_only", 0, 20260818,
                             str(RUNS[model]), "frozen_sampledH_27_conditions", str(RUNS[model]),
                             str(SUPERVISOR / "final_summary.json"), "exact_row_parity_required"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
