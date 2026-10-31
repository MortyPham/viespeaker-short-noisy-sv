#!/usr/bin/env python3
"""Train the matched VieSpeaker full-dataset AAM-Softmax-only baseline.

Place this script beside the completed ``train_viespeaker_d_full.py`` on the
training server. The D_FULL implementation remains the source of truth for
speaker eligibility, cached audio, MUSAN, balanced sampling, and verification.

The only optimization objective in this script is AAM-Softmax classification
of the 32 short, optionally noisy student examples in each optimizer step.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


SYSTEM_NAME = "AAM_ONLY"
EXPECTED_TRAINING_SPEAKERS = 3359
EXPECTED_VALIDATION_SPEAKERS = 90
PROJECT_ROOT = Path("/media/ml4u/Samsung_T51/ptnk_team/mortyhuaphuoc")
PRETRAINED_ROOT = Path("/home/ml4u/.cache/viespeaker_models/speechbrain_ecapa_voxceleb")


def build_parser() -> argparse.ArgumentParser:
    """Declare the controlled baseline configuration and execution options."""
    parser = argparse.ArgumentParser(
        description="Train the matched 3,371-speaker VieSpeaker AAM-only baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--legacy-script", type=Path, default=Path("train_viespeaker_d_full.py"))
    parser.add_argument("--speaker-splits", type=Path, default=Path("experiments/shared/speaker_splits.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path("experiments/shared/d_full_audio_cache"))
    parser.add_argument(
        "--recording-manifest",
        type=Path,
        default=Path("experiments/shared/d_full_audio_cache/recordings.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/mentor_stage/m0_aam_only_full_dataset"),
    )
    parser.add_argument(
        "--reference-speaker-labels",
        type=Path,
        default=Path("experiments/mentor_stage/m3_legacy_ema_gaussian_kl/training_speaker_labels.json"),
    )
    parser.add_argument("--musan-dir", type=Path, default=Path("datasets/musan"))
    parser.add_argument("--pretrained-dir", type=Path, default=PRETRAINED_ROOT)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--speakers-per-batch", type=int, default=16)
    parser.add_argument("--pairs-per-speaker", type=int, default=2)
    parser.add_argument("--short-min-seconds", type=float, default=1.0)
    parser.add_argument("--short-max-seconds", type=float, default=1.5)
    parser.add_argument("--clean-probability", type=float, default=0.20)
    parser.add_argument("--noise-snr-min-db", type=float, default=0.0)
    parser.add_argument("--noise-snr-max-db", type=float, default=20.0)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--aam-margin", type=float, default=0.2)
    parser.add_argument("--aam-scale", type=float, default=30.0)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--validation-interval", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--periodic-interval", type=int, default=5000)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--validation-enrollment-seconds", type=float, default=3.0)
    parser.add_argument("--validation-probe-seconds", type=float, default=1.25)
    parser.add_argument("--validation-noise-db", type=float, default=10.0)
    parser.add_argument("--target-far", type=float, default=0.05)
    parser.add_argument("--enrollment-count", type=int, default=5)
    parser.add_argument("--calibration-count", type=int, default=10)
    parser.add_argument("--heldout-count", type=int, default=10)
    parser.add_argument("--allow-incomplete-speakers", action="store_true", default=True)
    parser.add_argument("--require-complete-speakers", dest="allow_incomplete_speakers", action="store_false")
    parser.add_argument("--expected-training-speakers", type=int, default=EXPECTED_TRAINING_SPEAKERS)
    parser.add_argument("--expected-validation-speakers", type=int, default=EXPECTED_VALIDATION_SPEAKERS)
    parser.add_argument("--expected-embedding-dim", type=int, default=192)
    parser.add_argument("--system-name", default=SYSTEM_NAME)
    parser.add_argument("--wandb-project", default="viespeaker-mentor-aam-only")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default="aam-only-full-data-short-noisy-3359-speakers")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--pooling-variant", choices=("original", "global_local", "global_local_duration"), default="original")
    parser.add_argument("--local-window-seconds", type=float, default=0.5)
    parser.add_argument("--local-overlap", type=float, default=0.5)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser


def parse_arguments() -> tuple[argparse.Namespace, list[str]]:
    """Parse baseline flags and preserve extra flags for the legacy parser."""
    parser = build_parser()
    args, legacy_arguments = parser.parse_known_args()
    if args.steps < 1:
        parser.error("--steps must be at least 1")
    if args.speakers_per_batch < 1 or args.pairs_per_speaker < 1:
        parser.error("batch dimensions must be positive")
    if args.short_min_seconds <= 0 or args.short_min_seconds > args.short_max_seconds:
        parser.error("short crop durations must satisfy 0 < minimum <= maximum")
    if not 0.0 <= args.clean_probability <= 1.0:
        parser.error("--clean-probability must be between 0 and 1")
    for name in ("validation_interval", "save_interval", "periodic_interval", "log_interval"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    return args, legacy_arguments


def resolve_path(value: Path) -> Path:
    """Resolve relative server paths from the current project directory."""
    return value.expanduser().resolve()


def json_safe(value: Any) -> Any:
    """Convert paths, NumPy values, and nested mappings into JSON values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    return value


def write_json(path: Path, payload: Any) -> None:
    """Write a JSON artifact atomically in the target output directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(payload: Mapping[str, Any], destination: Path) -> None:
    """Write a complete checkpoint before replacing its public filename."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, destination)


def sha256_file(path: Path) -> str:
    """Calculate the digest of a frozen split or recording manifest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_legacy_module(path: Path) -> Any:
    """Import the completed D_FULL implementation without invoking training."""
    if not path.is_file():
        raise FileNotFoundError(
            f"D_FULL implementation not found: {path}. Place this file next to "
            "train_viespeaker_d_full.py or provide --legacy-script."
        )
    spec = importlib.util.spec_from_file_location("viespeaker_d_full_legacy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import the D_FULL implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    required = (
        "load_speaker_splits",
        "load_recording_manifest",
        "select_eligible_records",
        "create_validation_assignments",
        "evaluate_verification",
        "load_pretrained_encoder",
        "AAMSoftmaxHead",
        "MusanAugmenter",
        "SpeakerPairDataset",
        "SpeakerBalancedBatchSampler",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(
            "The D_FULL implementation does not expose the required interfaces: "
            + ", ".join(missing)
        )
    return module


def load_legacy_defaults(legacy: Any, legacy_arguments: list[str]) -> argparse.Namespace:
    """Recover every original D_FULL default before applying baseline overrides."""
    for builder_name in ("build_parser", "create_parser", "get_parser"):
        builder = getattr(legacy, builder_name, None)
        if callable(builder):
            parser = builder()
            if isinstance(parser, argparse.ArgumentParser):
                return parser.parse_args(legacy_arguments)

    for function_name in ("parse_args", "parse_arguments"):
        function = getattr(legacy, function_name, None)
        if not callable(function):
            continue
        parameters = inspect.signature(function).parameters
        if parameters:
            namespace = function(legacy_arguments)
        else:
            original_argv = sys.argv
            try:
                sys.argv = [str(getattr(legacy, "__file__", "train_viespeaker_d_full.py"))]
                sys.argv.extend(legacy_arguments)
                namespace = function()
            finally:
                sys.argv = original_argv
        if isinstance(namespace, tuple):
            namespace = namespace[0]
        if isinstance(namespace, argparse.Namespace):
            return namespace
        raise TypeError(f"legacy.{function_name}() returned {type(namespace).__name__}, not Namespace")

    if legacy_arguments:
        raise RuntimeError("Extra legacy options were supplied, but the D_FULL parser could not be found")
    print("Warning: D_FULL parser was not found; using the baseline namespace directly.", flush=True)
    return argparse.Namespace()


def apply_aliases(namespace: argparse.Namespace, names: Iterable[str], value: Any) -> None:
    """Update recognized legacy aliases and expose the primary canonical name."""
    names = tuple(names)
    for name in names:
        if hasattr(namespace, name):
            setattr(namespace, name, value)
    setattr(namespace, names[0], value)


def merge_legacy_arguments(
    legacy_args: argparse.Namespace,
    baseline_args: argparse.Namespace,
) -> argparse.Namespace:
    """Keep D_FULL defaults while enforcing every matched baseline setting."""
    for key, value in vars(baseline_args).items():
        setattr(legacy_args, key, value)

    aliases: list[tuple[tuple[str, ...], Any]] = [
        (("speaker_splits", "split_csv", "speaker_split_path"), baseline_args.speaker_splits),
        (("cache_dir", "audio_cache", "audio_cache_dir"), baseline_args.cache_dir),
        (("recording_manifest", "manifest_path", "recordings_manifest"), baseline_args.recording_manifest),
        (("output_dir", "result_dir", "experiment_dir"), baseline_args.output_dir),
        (("musan_dir", "musan_root", "noise_dir", "noise_root"), baseline_args.musan_dir),
        (("pretrained_dir", "pretrained_model_dir", "model_dir", "model_cache_dir"), baseline_args.pretrained_dir),
        (("steps", "max_steps", "total_steps", "training_steps"), baseline_args.steps),
        (("speakers_per_batch", "batch_speakers"), baseline_args.speakers_per_batch),
        (("pairs_per_speaker", "examples_per_speaker", "samples_per_speaker"), baseline_args.pairs_per_speaker),
        (("short_min_seconds", "student_min_seconds", "probe_min_seconds", "min_probe_seconds"), baseline_args.short_min_seconds),
        (("short_max_seconds", "student_max_seconds", "probe_max_seconds", "max_probe_seconds"), baseline_args.short_max_seconds),
        (("clean_probability", "clean_prob", "probability_clean"), baseline_args.clean_probability),
        (("noise_probability", "noise_prob", "augmentation_probability"), 1.0 - baseline_args.clean_probability),
        (("noise_snr_min_db", "snr_min_db", "min_snr_db", "train_snr_min"), baseline_args.noise_snr_min_db),
        (("noise_snr_max_db", "snr_max_db", "max_snr_db", "train_snr_max"), baseline_args.noise_snr_max_db),
        (("encoder_lr", "learning_rate_encoder", "lr_encoder"), baseline_args.encoder_lr),
        (("head_lr", "learning_rate_head", "lr_head"), baseline_args.head_lr),
        (("weight_decay",), baseline_args.weight_decay),
        (("aam_margin", "margin"), baseline_args.aam_margin),
        (("aam_scale", "scale"), baseline_args.aam_scale),
        (("warmup_steps", "warmup"), baseline_args.warmup_steps),
        (("gradient_clip", "grad_clip", "max_grad_norm"), baseline_args.gradient_clip),
        (("num_workers", "workers"), baseline_args.num_workers),
        (("seed", "random_seed"), baseline_args.seed),
        (("validation_interval", "eval_every", "validation_every", "eval_interval"), baseline_args.validation_interval),
        (("save_interval", "save_every"), baseline_args.save_interval),
        (("periodic_interval", "periodic_every"), baseline_args.periodic_interval),
        (("log_interval", "log_every"), baseline_args.log_interval),
        (("validation_enrollment_seconds", "val_enroll_seconds", "dev_enroll_seconds", "enrollment_seconds"), baseline_args.validation_enrollment_seconds),
        (("validation_probe_seconds", "val_probe_seconds", "dev_probe_seconds", "probe_seconds"), baseline_args.validation_probe_seconds),
        (("validation_noise_db", "val_snr_db", "dev_snr_db", "validation_snr_db"), baseline_args.validation_noise_db),
        (("target_far", "far_target"), baseline_args.target_far),
        (("enrollment_count", "num_enrollments", "enrollments_per_speaker", "validation_enrollment_count"), baseline_args.enrollment_count),
        (("calibration_count", "num_calibration", "calibration_per_speaker", "validation_calibration_count"), baseline_args.calibration_count),
        (("heldout_count", "test_count", "num_test_probes", "heldout_per_speaker", "validation_test_count"), baseline_args.heldout_count),
        (("allow_incomplete_speakers",), baseline_args.allow_incomplete_speakers),
        (("validation_enrollment_model", "enrollment_model"), "student"),
    ]
    for names, value in aliases:
        apply_aliases(legacy_args, names, value)

    for name in (
        "probe_duration_range",
        "short_duration_range",
        "student_duration_range",
        "train_probe_duration_range",
    ):
        if hasattr(legacy_args, name):
            setattr(legacy_args, name, (baseline_args.short_min_seconds, baseline_args.short_max_seconds))
    for name in ("snr_range", "train_snr_range", "noise_snr_range"):
        if hasattr(legacy_args, name):
            setattr(legacy_args, name, (baseline_args.noise_snr_min_db, baseline_args.noise_snr_max_db))
    return legacy_args


def construct_compatible(factory: Any, candidates: Mapping[str, Any], description: str) -> Any:
    """Instantiate a legacy class by matching its declared constructor names."""
    signature = inspect.signature(factory)
    positional: list[Any] = []
    keyword: dict[str, Any] = {}
    missing: list[str] = []
    for parameter in signature.parameters.values():
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if parameter.name not in candidates:
            if parameter.default is inspect.Parameter.empty:
                missing.append(parameter.name)
            continue
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            positional.append(candidates[parameter.name])
        else:
            keyword[parameter.name] = candidates[parameter.name]
    if missing:
        raise RuntimeError(
            f"Cannot construct {description}; unsupported required parameters: {missing}. "
            f"Actual D_FULL signature: {signature}"
        )
    return factory(*positional, **keyword)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and available PyTorch devices."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def speaker_ids_from_reference(payload: Any) -> list[str]:
    """Read either speaker-to-class mappings or explicit speaker lists."""
    if isinstance(payload, list):
        return sorted(str(item) for item in payload)
    if not isinstance(payload, Mapping):
        raise TypeError("Reference speaker labels must be a JSON list or mapping")
    for key in ("speaker_to_label", "speaker_labels", "training_speaker_labels"):
        if isinstance(payload.get(key), Mapping):
            return sorted(str(item) for item in payload[key])
    for key in ("train_speaker_ids", "training_speakers", "speaker_ids"):
        value = payload.get(key)
        if isinstance(value, list):
            return sorted(str(item) for item in value)
    if all(isinstance(value, (int, np.integer)) for value in payload.values()):
        return sorted(str(item) for item in payload)
    raise ValueError("Could not identify training speaker IDs in the D_FULL label file")


def verify_reference_speakers(path: Path, speakers: Iterable[str]) -> None:
    """Fail if the baseline's training identities differ from completed D_FULL."""
    if not path.is_file():
        raise FileNotFoundError(
            f"D_FULL speaker-label reference not found: {path}. "
            "The controlled comparison requires exact identity-set verification."
        )
    reference = set(speaker_ids_from_reference(json.loads(path.read_text(encoding="utf-8"))))
    selected = {str(speaker) for speaker in speakers}
    if reference != selected:
        only_reference = sorted(reference - selected)[:10]
        only_selected = sorted(selected - reference)[:10]
        raise RuntimeError(
            "The baseline and D_FULL use different training identities: "
            f"reference_only={only_reference}, baseline_only={only_selected}, "
            f"reference_count={len(reference)}, baseline_count={len(selected)}"
        )


def get_first_mapping_value(item: Mapping[str, Any], names: Iterable[str]) -> Any:
    """Extract one supported field from a legacy dataset sample."""
    for name in names:
        if name in item:
            return item[name]
    raise KeyError(f"None of {tuple(names)} exist in legacy dataset sample: {sorted(item)}")


def student_only_collate(batch: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate only short student waveforms and their speaker labels."""
    audio: list[torch.Tensor] = []
    labels: list[int] = []
    for item in batch:
        waveform = get_first_mapping_value(
            item,
            (
                "probe",
                "probe_audio",
                "probe_waveform",
                "student",
                "student_audio",
                "student_waveform",
                "student_wav",
                "short",
                "short_audio",
                "short_waveform",
                "waveform",
            ),
        )
        waveform_tensor = torch.as_tensor(waveform, dtype=torch.float32).squeeze()
        if waveform_tensor.ndim != 1 or waveform_tensor.numel() == 0:
            raise ValueError(f"Expected a nonempty mono student waveform, received {tuple(waveform_tensor.shape)}")
        audio.append(waveform_tensor)
        labels.append(int(get_first_mapping_value(item, ("label", "speaker_label", "class_index", "target"))))

    padded = pad_sequence(audio, batch_first=True)
    lengths = torch.tensor([waveform.numel() for waveform in audio], dtype=torch.float32)
    return {
        "student_audio": padded,
        "student_lengths": lengths / padded.shape[1],
        "labels": torch.tensor(labels, dtype=torch.long),
        "metadata": batch,
    }


def verify_first_batch(batch: Mapping[str, Any], args: argparse.Namespace) -> None:
    """Check the controlled speaker balance before applying the first update."""
    labels = batch["labels"].detach().cpu()
    expected_examples = args.speakers_per_batch * args.pairs_per_speaker
    if labels.numel() != expected_examples:
        raise RuntimeError(f"Expected {expected_examples} student examples; received {labels.numel()}")
    unique, counts = torch.unique(labels, return_counts=True)
    if unique.numel() != args.speakers_per_batch or not torch.all(counts == args.pairs_per_speaker):
        raise RuntimeError(
            f"Expected {args.speakers_per_batch} distinct speakers with "
            f"{args.pairs_per_speaker} examples each; observed counts={counts.tolist()}"
        )
    for item in batch.get("metadata", []):
        for key in ("probe_seconds", "student_seconds", "short_seconds", "crop_seconds"):
            if key in item:
                duration = float(item[key])
                if not args.short_min_seconds - 1e-6 <= duration <= args.short_max_seconds + 1e-6:
                    raise RuntimeError(
                        f"Student crop {duration:.3f}s is outside the matched "
                        f"{args.short_min_seconds:.1f}-{args.short_max_seconds:.1f}s range"
                    )
                break
    print(
        f"Verified batch: {unique.numel()} distinct speakers, "
        f"{args.pairs_per_speaker} examples/speaker, {labels.numel()} student examples.",
        flush=True,
    )


def infer_embedding_dimension(encoder: torch.nn.Module, device: torch.device) -> int:
    """Infer the encoder output size without updating batch-normalization state."""
    previous_mode = encoder.training
    encoder.eval()
    with torch.inference_mode():
        waveform = torch.zeros(2, 16_000, device=device)
        lengths = torch.ones(2, device=device)
        embedding = encoder(waveform, lengths)
    encoder.train(previous_mode)
    if embedding.ndim == 3 and embedding.shape[1] == 1:
        embedding = embedding[:, 0, :]
    if embedding.ndim != 2:
        raise RuntimeError(f"Unexpected ECAPA embedding dimensions: {tuple(embedding.shape)}")
    return int(embedding.shape[-1])


def learning_rate_multiplier(step: int, warmup_steps: int, total_steps: int) -> float:
    """Match the D_FULL linear-warmup and cosine-decay schedule."""
    if step < warmup_steps:
        return max(step, 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def make_resume_signature(args: argparse.Namespace, training_speakers: Iterable[str]) -> dict[str, Any]:
    """Describe every configuration value that must remain fixed on resume."""
    return {
        "system": SYSTEM_NAME,
        "train_speaker_ids": sorted(str(speaker) for speaker in training_speakers),
        "speaker_split_sha256": sha256_file(args.speaker_splits),
        "recording_manifest_sha256": sha256_file(args.recording_manifest),
        "legacy_script_sha256": sha256_file(args.legacy_script),
        "pretrained_dir": str(args.pretrained_dir),
        "steps": args.steps,
        "speakers_per_batch": args.speakers_per_batch,
        "pairs_per_speaker": args.pairs_per_speaker,
        "short_min_seconds": args.short_min_seconds,
        "short_max_seconds": args.short_max_seconds,
        "clean_probability": args.clean_probability,
        "noise_snr_min_db": args.noise_snr_min_db,
        "noise_snr_max_db": args.noise_snr_max_db,
        "encoder_lr": args.encoder_lr,
        "head_lr": args.head_lr,
        "weight_decay": args.weight_decay,
        "aam_margin": args.aam_margin,
        "aam_scale": args.aam_scale,
        "warmup_steps": args.warmup_steps,
        "gradient_clip": args.gradient_clip,
        "seed": args.seed,
        "validation_enrollment_seconds": args.validation_enrollment_seconds,
        "validation_probe_seconds": args.validation_probe_seconds,
        "validation_noise_db": args.validation_noise_db,
        "target_far": args.target_far,
        "enrollment_count": args.enrollment_count,
        "calibration_count": args.calibration_count,
        "heldout_count": args.heldout_count,
    }


def make_checkpoint(
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    step: int,
    best_noise_eer: float,
    best_noise_tar: float,
    history: list[dict[str, Any]],
    wandb_run_id: str | None,
    signature: Mapping[str, Any],
) -> dict[str, Any]:
    """Produce a resumable checkpoint compatible with the shared evaluator."""
    return {
        "system": SYSTEM_NAME,
        "system_name": SYSTEM_NAME,
        "student": encoder.state_dict(),
        "aam_head": head.state_dict(),
        "head": head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "global_step": step,
        "best_noise_eer": best_noise_eer,
        "best_noise_tar": best_noise_tar,
        "history": history,
        "wandb_run_id": wandb_run_id,
        "signature": dict(signature),
        "train_speaker_ids": list(signature["train_speaker_ids"]),
    }


def flatten_metrics(payload: Any, prefix: str = "") -> dict[str, float]:
    """Flatten nested validation metrics without assuming one legacy schema."""
    flattened: dict[str, float] = {}
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            next_prefix = f"{prefix}_{key}" if prefix else str(key)
            flattened.update(flatten_metrics(value, next_prefix))
    elif isinstance(payload, (int, float, np.integer, np.floating)) and not isinstance(payload, bool):
        flattened[prefix.lower()] = float(payload)
    return flattened


def resolve_condition_metric(metrics: Mapping[str, float], condition: str, metric: str) -> float | None:
    """Locate a clean or noisy metric across supported D_FULL metric names."""
    condition_names = ("noise", "noisy", "musan") if condition == "noise" else ("clean",)
    metric_names = {
        "eer": ("eer", "equal_error_rate"),
        "far": ("far", "macro_far", "false_acceptance_rate"),
        "frr": ("frr", "macro_frr", "false_rejection_rate"),
        "tar": ("tar", "macro_tar", "tar_at_target_far", "true_acceptance_rate"),
        "balanced_error": ("balanced_error", "macro_balanced_error", "be"),
    }[metric]
    normalized = {key.replace("/", "_").replace("-", "_"): value for key, value in metrics.items()}
    candidates = [
        f"{condition_name}_{metric_name}"
        for condition_name in condition_names
        for metric_name in metric_names
    ] + [
        f"{metric_name}_{condition_name}"
        for condition_name in condition_names
        for metric_name in metric_names
    ]
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    for key, value in normalized.items():
        for candidate in candidates:
            if key.endswith("_" + candidate):
                return value
    if metric == "tar":
        frr = resolve_condition_metric(metrics, condition, "frr")
        if frr is not None:
            return 1.0 - frr
    if metric == "balanced_error":
        far = resolve_condition_metric(metrics, condition, "far")
        frr = resolve_condition_metric(metrics, condition, "frr")
        if far is not None and frr is not None:
            return (far + frr) / 2.0
    return None


def canonical_validation_metrics(payload: Any) -> tuple[dict[str, float], dict[str, Any]]:
    """Convert the legacy verification output into stable logging metrics."""
    if isinstance(payload, tuple):
        payload = payload[0]
    if not isinstance(payload, Mapping):
        raise TypeError(f"Legacy validation returned {type(payload).__name__}; expected mapping")
    flattened = flatten_metrics(payload)
    canonical: dict[str, float] = {}
    for condition in ("clean", "noise"):
        for metric in ("eer", "far", "frr", "tar", "balanced_error"):
            value = resolve_condition_metric(flattened, condition, metric)
            if value is not None:
                canonical[f"validation/{condition}_{metric}"] = value
    for required in ("validation/noise_eer", "validation/noise_tar"):
        if required not in canonical:
            raise RuntimeError(
                f"Cannot identify required {required} in D_FULL validation output. "
                f"Available numeric keys: {sorted(flattened)}"
            )
    return canonical, dict(payload)


def start_wandb(args: argparse.Namespace, run_id: str | None, signature: Mapping[str, Any]) -> Any:
    """Start or resume the same Weights & Biases experiment."""
    if args.no_wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "Weights & Biases is unavailable in this environment. Install it or pass --no-wandb."
        ) from exc
    safe_signature = {
        key: value
        for key, value in signature.items()
        if key != "train_speaker_ids"
    }
    safe_signature["training_speaker_count"] = len(signature["train_speaker_ids"])
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        id=run_id,
        resume="allow" if run_id else None,
        config=json_safe(safe_signature),
    )


def validate_configuration(args: argparse.Namespace) -> None:
    """Reject missing frozen inputs before constructing models or datasets."""
    required_files = (args.legacy_script, args.speaker_splits, args.recording_manifest)
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"Required frozen input does not exist: {path}")
    for path in (args.cache_dir, args.musan_dir, args.pretrained_dir):
        if not path.is_dir():
            raise NotADirectoryError(f"Required training directory does not exist: {path}")
    protected_output = args.reference_speaker_labels.parent.resolve()
    if args.output_dir.resolve() == protected_output:
        raise RuntimeError(f"Refusing to overwrite the completed D_FULL output directory: {protected_output}")


def main() -> None:
    """Train, validate, checkpoint, and resume the matched AAM-only baseline."""
    baseline_args, legacy_arguments = parse_arguments()
    global SYSTEM_NAME
    SYSTEM_NAME = baseline_args.system_name
    for field in (
        "legacy_script",
        "speaker_splits",
        "cache_dir",
        "recording_manifest",
        "output_dir",
        "reference_speaker_labels",
        "musan_dir",
        "pretrained_dir",
    ):
        setattr(baseline_args, field, resolve_path(getattr(baseline_args, field)))

    validate_configuration(baseline_args)
    legacy = load_legacy_module(baseline_args.legacy_script)
    legacy_defaults = load_legacy_defaults(legacy, legacy_arguments)
    args = merge_legacy_arguments(legacy_defaults, baseline_args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.allow_cpu:
        device = torch.device("cpu")
    else:
        raise RuntimeError("CUDA is unavailable. Inspect nvidia-smi or pass --allow-cpu for debugging.")
    amp_enabled = device.type == "cuda" and not args.disable_amp

    partitions = legacy.load_speaker_splits(args.speaker_splits)
    records = legacy.load_recording_manifest(args.recording_manifest, args.cache_dir)
    records, training_speakers_raw, validation_speakers_raw = legacy.select_eligible_records(
        records,
        partitions,
        args,
    )
    training_speakers = sorted(str(speaker) for speaker in training_speakers_raw)
    validation_speakers = sorted(str(speaker) for speaker in validation_speakers_raw)
    if len(training_speakers) != args.expected_training_speakers:
        raise RuntimeError(
            f"Expected {args.expected_training_speakers} D_FULL training speakers, "
            f"but selected {len(training_speakers)}"
        )
    if len(validation_speakers) != args.expected_validation_speakers:
        raise RuntimeError(
            f"Expected {args.expected_validation_speakers} D_FULL validation speakers, "
            f"but selected {len(validation_speakers)}"
        )
    if set(training_speakers) & set(validation_speakers):
        raise RuntimeError("Training and validation speaker sets overlap")
    verify_reference_speakers(args.reference_speaker_labels, training_speakers)
    speaker_to_label = {speaker: index for index, speaker in enumerate(training_speakers)}
    write_json(args.output_dir / "training_speaker_labels.json", speaker_to_label)
    assignments = legacy.create_validation_assignments(records, validation_speakers, args)

    print(
        json.dumps(
            {
                "system": SYSTEM_NAME,
                "device": str(device),
                "training_speakers": len(training_speakers),
                "validation_speakers": len(validation_speakers),
                "examples_per_step": args.speakers_per_batch * args.pairs_per_speaker,
                "short_seconds": [args.short_min_seconds, args.short_max_seconds],
                "clean_probability": args.clean_probability,
                "noise_snr_db": [args.noise_snr_min_db, args.noise_snr_max_db],
                "steps": args.steps,
                "output_dir": str(args.output_dir),
                "mixed_precision": amp_enabled,
                "loss": "AAM-Softmax only",
            },
            indent=2,
        ),
        flush=True,
    )

    shared_candidates: dict[str, Any] = {
        "args": args,
        "config": args,
        "options": args,
        "records": records,
        "recordings": records,
        "record_index": records,
        "record_map": records,
        "speaker_to_label": speaker_to_label,
        "speaker_to_index": speaker_to_label,
        "speaker_labels": speaker_to_label,
        "label_mapping": speaker_to_label,
        "label_map": speaker_to_label,
        "class_mapping": speaker_to_label,
        "speakers": training_speakers,
        "speaker_ids": training_speakers,
        "training_speakers": training_speakers,
        "train_speakers": training_speakers,
        "root": args.musan_dir,
        "musan_root": args.musan_dir,
        "musan_dir": args.musan_dir,
        "musan_path": args.musan_dir,
        "root_dir": args.musan_dir,
        "noise_root": args.musan_dir,
        "noise_dir": args.musan_dir,
        "seed": args.seed,
        "random_seed": args.seed,
        "sample_rate": getattr(args, "sample_rate", 16_000),
        "clean_probability": args.clean_probability,
        "clean_prob": args.clean_probability,
        "noise_probability": 1.0 - args.clean_probability,
        "snr_min_db": args.noise_snr_min_db,
        "snr_max_db": args.noise_snr_max_db,
        "snr_range": (args.noise_snr_min_db, args.noise_snr_max_db),
        "num_workers": args.num_workers,
        "device": device,
    }
    augmenter = construct_compatible(legacy.MusanAugmenter, shared_candidates, "legacy.MusanAugmenter")
    shared_candidates.update({"augmenter": augmenter, "augmentation": augmenter, "audio_augmenter": augmenter})
    dataset = construct_compatible(legacy.SpeakerPairDataset, shared_candidates, "legacy.SpeakerPairDataset")

    encoder = legacy.load_pretrained_encoder(args).to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(True)
    embedding_dimension = infer_embedding_dimension(encoder, device)
    if embedding_dimension != args.expected_embedding_dim:
        raise RuntimeError(
            f"Expected {args.expected_embedding_dim}-dimensional embeddings, "
            f"received {embedding_dimension}"
        )
    head_candidates = {
        "embedding_dim": embedding_dimension,
        "embedding_dimension": embedding_dimension,
        "in_features": embedding_dimension,
        "classes": len(training_speakers),
        "num_classes": len(training_speakers),
        "n_classes": len(training_speakers),
        "out_features": len(training_speakers),
        "margin": args.aam_margin,
        "aam_margin": args.aam_margin,
        "scale": args.aam_scale,
        "aam_scale": args.aam_scale,
        "args": args,
    }
    head = construct_compatible(legacy.AAMSoftmaxHead, head_candidates, "legacy.AAMSoftmaxHead").to(device)
    if hasattr(head, "weight") and tuple(head.weight.shape) != (len(training_speakers), embedding_dimension):
        raise RuntimeError(
            "AAM head does not match the eligible VieSpeaker identities: "
            f"shape={tuple(head.weight.shape)}"
        )

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder.parameters(), "lr": args.encoder_lr},
            {"params": head.parameters(), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: learning_rate_multiplier(step, args.warmup_steps, args.steps),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    signature = make_resume_signature(args, training_speakers)
    latest_path = args.output_dir / "latest.pt"
    step = 0
    best_noise_eer = float("inf")
    best_noise_tar = float("-inf")
    history: list[dict[str, Any]] = []
    wandb_run_id: str | None = None

    if latest_path.is_file():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        if checkpoint.get("signature") != signature:
            original = checkpoint.get("signature", {})
            changed = sorted(
                key
                for key in set(original) | set(signature)
                if original.get(key) != signature.get(key)
            )
            raise RuntimeError(
                "Existing AAM-only checkpoint belongs to an incompatible configuration. "
                f"Changed signature fields: {changed}. Choose a new --output-dir."
            )
        encoder.load_state_dict(checkpoint["student"])
        head.load_state_dict(checkpoint.get("aam_head", checkpoint["head"]))
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        step = int(checkpoint["global_step"])
        best_noise_eer = float(checkpoint["best_noise_eer"])
        best_noise_tar = float(checkpoint["best_noise_tar"])
        history = list(checkpoint["history"])
        wandb_run_id = checkpoint.get("wandb_run_id")
        print(f"Resuming AAM-only training from step {step:,}: {latest_path}", flush=True)

    if step >= args.steps:
        print(f"AAM-only training already completed at step {step:,}.", flush=True)
        return

    run = start_wandb(args, wandb_run_id, signature)
    if run is not None:
        wandb_run_id = str(run.id)

    remaining_steps = args.steps - step
    sampler_candidates = {
        **shared_candidates,
        "num_batches": remaining_steps,
        "batches": remaining_steps,
        "total_batches": remaining_steps,
        "num_steps": remaining_steps,
        "steps": remaining_steps,
        "start_step": step,
        "initial_step": step,
        "global_step": step,
        "speakers_per_batch": args.speakers_per_batch,
        "batch_speakers": args.speakers_per_batch,
        "pairs_per_speaker": args.pairs_per_speaker,
        "examples_per_speaker": args.pairs_per_speaker,
        "samples_per_speaker": args.pairs_per_speaker,
    }
    sampler = construct_compatible(
        legacy.SpeakerBalancedBatchSampler,
        sampler_candidates,
        "legacy.SpeakerBalancedBatchSampler",
    )
    loader_options: dict[str, Any] = {
        "dataset": dataset,
        "batch_sampler": sampler,
        "collate_fn": student_only_collate,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers:
        loader_options.update({"persistent_workers": True, "prefetch_factor": 2})
    loader = DataLoader(**loader_options)

    stop_requested = False

    def request_stop(signal_number: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"Received signal {signal_number}; saving after the current optimizer step.", flush=True)

    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    previous_sigint = signal.signal(signal.SIGINT, request_stop)

    def save_current(destination: Path) -> None:
        payload = make_checkpoint(
            encoder,
            head,
            optimizer,
            scheduler,
            scaler,
            step,
            best_noise_eer,
            best_noise_tar,
            history,
            wandb_run_id,
            signature,
        )
        atomic_torch_save(payload, destination)

    encoder.train()
    head.train()
    started = time.perf_counter()
    last_log_time = started
    running_loss = 0.0
    running_count = 0
    verified_batch = False

    try:
        for batch in loader:
            if not verified_batch:
                verify_first_batch(batch, args)
                verified_batch = True

            student_audio = batch["student_audio"].to(device, non_blocking=True)
            student_lengths = batch["student_lengths"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            autocast_context = (
                torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_enabled)
                if device.type == "cuda"
                else contextlib.nullcontext()
            )
            with autocast_context:
                embeddings = encoder(student_audio, student_lengths)
                if embeddings.ndim == 3 and embeddings.shape[1] == 1:
                    embeddings = embeddings[:, 0, :]
                normalized_embeddings = F.normalize(embeddings.float(), dim=-1)
                head_output = head(normalized_embeddings, labels)
                loss_aam = head_output[0] if isinstance(head_output, tuple) else head_output

            if not torch.isfinite(loss_aam):
                raise FloatingPointError(f"Nonfinite AAM loss at step {step + 1}: {loss_aam.item()}")
            scaler.scale(loss_aam).backward()
            scaler.unscale_(optimizer)
            encoder_grad_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.gradient_clip)
            head_grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), args.gradient_clip)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if not amp_enabled or scaler.get_scale() >= previous_scale:
                scheduler.step()
            step += 1
            running_loss += float(loss_aam.detach().cpu())
            running_count += 1

            if step % args.log_interval == 0 or step == args.steps:
                elapsed = max(time.perf_counter() - last_log_time, 1e-9)
                steps_per_second = running_count / elapsed
                remaining_hours = (args.steps - step) / max(steps_per_second, 1e-9) / 3600.0
                train_metrics = {
                    "global_step": step,
                    "train/loss_aam": running_loss / running_count,
                    "train/loss_total": running_loss / running_count,
                    "train/encoder_grad_norm": float(encoder_grad_norm.detach().cpu()),
                    "train/head_grad_norm": float(head_grad_norm.detach().cpu()),
                    "train/lr_encoder": float(optimizer.param_groups[0]["lr"]),
                    "train/lr_head": float(optimizer.param_groups[1]["lr"]),
                    "train/steps_per_second": steps_per_second,
                    "train/remaining_hours": remaining_hours,
                    "train/gpu_allocated_gb": (
                        torch.cuda.memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
                    ),
                }
                print(
                    f"step={step:05d}/{args.steps:05d} "
                    f"aam={train_metrics['train/loss_aam']:.4f} "
                    f"lr_encoder={train_metrics['train/lr_encoder']:.2e} "
                    f"lr_head={train_metrics['train/lr_head']:.2e} "
                    f"speed={steps_per_second:.2f} step/s "
                    f"remaining={remaining_hours:.2f}h",
                    flush=True,
                )
                if run is not None:
                    validation_is_due = step % args.validation_interval == 0 or step == args.steps
                    run.log(train_metrics, step=step, commit=not validation_is_due)
                running_loss = 0.0
                running_count = 0
                last_log_time = time.perf_counter()

            if step % args.validation_interval == 0 or step == args.steps:
                args.validation_enrollment_model = "student"
                validation_payload = legacy.evaluate_verification(
                    encoder,
                    encoder,
                    assignments,
                    augmenter,
                    args,
                    device,
                )
                validation_metrics, raw_metrics = canonical_validation_metrics(validation_payload)
                history_entry = {"step": step, "metrics": raw_metrics, **validation_metrics}
                history.append(json_safe(history_entry))
                write_json(args.output_dir / "validation_history.json", history)
                noise_eer = validation_metrics["validation/noise_eer"]
                noise_tar = validation_metrics["validation/noise_tar"]
                print(
                    f"Validation step={step:05d} "
                    f"noisy_EER={100.0 * noise_eer:.3f}% "
                    f"noisy_TAR@FAR5%={100.0 * noise_tar:.3f}%",
                    flush=True,
                )
                print(json.dumps(json_safe(raw_metrics), indent=2), flush=True)
                if run is not None:
                    run.log({"global_step": step, **validation_metrics}, step=step)

                improved_eer = noise_eer < best_noise_eer
                improved_tar = noise_tar > best_noise_tar
                if improved_eer:
                    best_noise_eer = noise_eer
                if improved_tar:
                    best_noise_tar = noise_tar
                if improved_eer:
                    destination = args.output_dir / "best_validation_eer.pt"
                    save_current(destination)
                    print(f"Saved best noisy-EER checkpoint: {destination}", flush=True)
                if improved_tar:
                    destination = args.output_dir / "best_validation_tar_at_target_far.pt"
                    save_current(destination)
                    print(f"Saved best noisy-TAR checkpoint: {destination}", flush=True)
                save_current(latest_path)
                encoder.train()
                head.train()
                last_log_time = time.perf_counter()

            elif step % args.save_interval == 0:
                save_current(latest_path)

            if step % args.periodic_interval == 0:
                save_current(args.output_dir / f"periodic_step_{step:06d}.pt")

            if stop_requested:
                save_current(latest_path)
                print(f"Saved resumable interruption checkpoint at step {step:,}: {latest_path}", flush=True)
                return

        if step != args.steps:
            raise RuntimeError(
                f"Legacy batch sampler stopped at step {step:,}, before the requested {args.steps:,} steps"
            )
        elapsed_hours = (time.perf_counter() - started) / 3600.0
        summary = {
            "system": SYSTEM_NAME,
            "completed_steps": step,
            "training_speakers": len(training_speakers),
            "validation_speakers": len(validation_speakers),
            "best_noisy_validation_eer": best_noise_eer,
            "best_noisy_validation_tar": best_noise_tar,
            "target_far": args.target_far,
            "enrollment_encoder": "student",
            "probe_encoder": "student",
            "elapsed_hours_this_invocation": elapsed_hours,
            "latest_checkpoint": str(latest_path),
            "best_eer_checkpoint": str(args.output_dir / "best_validation_eer.pt"),
            "best_tar_checkpoint": str(args.output_dir / "best_validation_tar_at_target_far.pt"),
            "wandb_run_id": wandb_run_id,
        }
        write_json(args.output_dir / "training_summary.json", summary)
        save_current(latest_path)
        print("AAM-only training complete:", json.dumps(summary, indent=2), flush=True)
        if run is not None:
            run.summary.update(json_safe(summary))
    except BaseException:
        if step > 0:
            try:
                save_current(latest_path)
                print(f"Saved emergency checkpoint at step {step:,}: {latest_path}", flush=True)
            except Exception as checkpoint_error:
                print(f"Emergency checkpoint failed: {checkpoint_error}", file=sys.stderr, flush=True)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGINT, previous_sigint)
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()