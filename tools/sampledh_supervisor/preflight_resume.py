#!/usr/bin/env python3
"""Read-only scientific preflight for the sampled-H resume sequence."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path("/media/ml4u/Samsung_T52/ptnk_team/mortyhuaphuoc")
HERE = Path(__file__).resolve().parent
RESULT_ROOT = PROJECT / "experiments/evaluation/sampledh_multimodel_20260922"
CANONICAL = PROJECT / "experiments/evaluation/architecture_A_viespeaker_h_1e1p/final_seed20260818_n5000"
B_RUN = PROJECT / "experiments/evaluation/ARCH_B_GLOBAL_LOCAL_sampledH_seed20260818_spec20260921_retry1/final_seed20260818_n5000"

FILES = {
    CANONICAL / "fixed_trials.csv": "97cd1ff475591682c865a0e50b36e3d78d1f86d8c3f14c6e6ce7b4d8076ac83b",
    PROJECT / "datasets/VieSpeaker/repository/trials/viespeaker_H.txt": "298addeecd574cf1b601f69e2c0971df9173b36cd741e6981846f6b4950793b0",
    CANONICAL / "audio_preparation_failures.csv": "e27f89f139a828b4d91f6773f9f42bbffe2dd31dffe68ed8fe97af6b9051e4fe",
    CANONICAL / "probe_transform_manifest.csv": "c86e8a1e6e2b59df84cbe16fb11ed30b524e95cefc564c0c0ffc8cd00202f8e1",
    CANONICAL / "valid_trials_by_condition.csv": "8bdea0482655a471217964dc679a0e408bbba3ba0d21e4af57c758bfa92c6bdb",
    PROJECT / "cache/huggingface/hub/models--speechbrain--spkrec-ecapa-voxceleb/snapshots/0f99f2d0ebe89ac095bcc5903c4dd8f72b367286/label_encoder.txt": "e13c3a167bb4112685670ee896d20e2b565af16b3a4ceeaa8689fa4d22adb8b9",
    PROJECT / "cache/huggingface/hub/models--speechbrain--spkrec-ecapa-voxceleb/snapshots/0f99f2d0ebe89ac095bcc5903c4dd8f72b367286/label_encoder.ckpt": "e13c3a167bb4112685670ee896d20e2b565af16b3a4ceeaa8689fa4d22adb8b9",
    PROJECT / "experiments/architecture_screening/A_full_clean_seed20260913/best_validation_eer.pt": "5ae4552e63eec82f23737f4de71601e7286c94a9dc92840fb32eeee82dc7c359",
    PROJECT / "experiments/architecture_screening/B_full_clean_seed20260913/best_validation_eer.pt": "67d188ba3f4c62d1854cb388c8f1eb52f9314343870809d30b2339f9134dd03a",
    PROJECT / "experiments/architecture_screening/C_full_clean_seed20260913/best_validation_eer.pt": "e9c4c0b40df0b0a1b8f5cf58093b4eb57f592c5e41a00a26cbb351eb3204d1e9",
    PROJECT / "experiments/mentor_stage/m0_aam_only_full_dataset/best_validation_eer.pt": "4ea92fd165853d3371f92d252f0993e76ca3c22f23a75d9affbf0d27931bf260",
    PROJECT / "experiments/mentor_stage/m2_legacy_vector_ema/best_validation_eer.pt": "672fba529dd17195a77fc1e0d9731cefb6e0172e12cc185db904eed2c4271d54",
    PROJECT / "experiments/mentor_stage/m3_legacy_ema_gaussian_kl/best_validation_eer.pt": "c01ed0a962426098610b77e5c8b89eb9d1b27a64adbd217765b9ee94637b5f41",
}

MODELS = {
    "ARCH_B_GLOBAL_LOCAL": (B_RUN, "67d188ba3f4c62d1854cb388c8f1eb52f9314343870809d30b2339f9134dd03a", "student", "student"),
    "ARCH_A_ORIGINAL": (RESULT_ROOT / "ARCH_A_ORIGINAL/final_seed20260818_n5000", "5ae4552e63eec82f23737f4de71601e7286c94a9dc92840fb32eeee82dc7c359", "student", "student"),
    "ARCH_C_RELATIVE_LENGTH_GATE": (RESULT_ROOT / "ARCH_C_RELATIVE_LENGTH_GATE/final_seed20260818_n5000", "e9c4c0b40df0b0a1b8f5cf58093b4eb57f592c5e41a00a26cbb351eb3204d1e9", "student", "student"),
    "LEGACY_B_AAM_ONLY": (RESULT_ROOT / "LEGACY_B_AAM_ONLY/final_seed20260818_n5000", "4ea92fd165853d3371f92d252f0993e76ca3c22f23a75d9affbf0d27931bf260", "student", "student"),
    "LEGACY_C_VECTOR_EMA": (RESULT_ROOT / "LEGACY_C_VECTOR_EMA/final_seed20260818_n5000", "672fba529dd17195a77fc1e0d9731cefb6e0172e12cc185db904eed2c4271d54", "ema_teacher", "student"),
    "LEGACY_D_FULL_KL_EMA": (RESULT_ROOT / "LEGACY_D_FULL_KL_EMA/final_seed20260818_n5000", "c01ed0a962426098610b77e5c8b89eb9d1b27a64adbd217765b9ee94637b5f41", "ema_teacher", "student"),
}
ORDER = ["LEGACY_B_AAM_ONLY", "LEGACY_C_VECTOR_EMA", "LEGACY_D_FULL_KL_EMA"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(model: str, report: Path) -> tuple[bool, str]:
    run_dir, checkpoint_sha, enrollment, probe = MODELS[model]
    command = [sys.executable, str(HERE / "validate_model.py"), "--model", model,
               "--run-dir", str(run_dir), "--canonical-dir", str(CANONICAL),
               "--checkpoint-sha", checkpoint_sha, "--enrollment-encoder", enrollment,
               "--probe-encoder", probe, "--parity-checker",
               str(HERE / "check_audio_quality_semantic_parity.py"), "--report", str(report)]
    if model == "ARCH_A_ORIGINAL":
        command.append("--require-a-score-reproduction")
    result = subprocess.run(command, text=True, capture_output=True)
    return result.returncode == 0, result.stdout + result.stderr


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--next-model", choices=ORDER, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    args = parser.parse_args()
    args.validation_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    observed: dict[str, object] = {"next_model": args.next_model, "hashes": {}, "validated": {}}

    for path, expected in FILES.items():
        if not path.is_file():
            errors.append(f"missing frozen file: {path}")
            continue
        actual = sha256(path)
        observed["hashes"][str(path)] = actual
        if actual != expected:
            errors.append(f"hash mismatch: {path}: {actual} != {expected}")

    completed = ["ARCH_B_GLOBAL_LOCAL", "ARCH_A_ORIGINAL", "ARCH_C_RELATIVE_LENGTH_GATE"]
    completed += ORDER[:ORDER.index(args.next_model)]
    for model in completed:
        report = args.validation_dir / f"{model}.json"
        passed, detail = validate(model, report)
        observed["validated"][model] = {"passed": passed, "report": str(report), "process_output": detail}
        if not passed:
            errors.append(f"exact-row validation failed: {model}")

    for model in ORDER[ORDER.index(args.next_model):]:
        output_root = RESULT_ROOT / model
        if output_root.exists():
            errors.append(f"non-overwrite output exists for pending model: {output_root}")

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(HERE))
    checkpoint_specs = [
        ("evaluate_aam_only_full_viespeaker_h_1e1p", "LEGACY_B_AAM_ONLY", 3371,
         PROJECT / "experiments/mentor_stage/m0_aam_only_full_dataset/best_validation_eer.pt"),
        ("evaluate_c_vector_ema_viespeaker_h_1e1p", "LEGACY_C_VECTOR_EMA", 3371,
         PROJECT / "experiments/mentor_stage/m2_legacy_vector_ema/best_validation_eer.pt"),
        ("evaluate_d_full_viespeaker_h_1e1p", "LEGACY_D_FULL_KL_EMA", 3371,
         PROJECT / "experiments/mentor_stage/m3_legacy_ema_gaussian_kl/best_validation_eer.pt"),
    ]
    for module_name, model, expected_speakers, checkpoint in checkpoint_specs:
        try:
            module = importlib.import_module(module_name)
            module.EXPECTED_TRAINING_SPEAKERS = expected_speakers
            metadata = module.inspect_checkpoint(checkpoint)
            observed.setdefault("checkpoint_metadata", {})[model] = metadata
            if metadata.get("training_speakers") != expected_speakers:
                errors.append(f"{model} training-speaker metadata mismatch: {metadata}")
        except Exception as exc:  # recorded rather than hidden by orchestration
            errors.append(f"{model} checkpoint inspection failed: {type(exc).__name__}: {exc}")

    observed["required_audio_quality_canonical_sha256"] = "d8ffd41fecd799b7bb75e9e0651bc21e38479e9cb4742445f2b05809876b988a"
    observed["expected_ordered_rows"] = 101619
    observed["passed"] = not errors
    observed["errors"] = errors
    args.report.write_text(json.dumps(observed, indent=2, sort_keys=True) + "\n")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
