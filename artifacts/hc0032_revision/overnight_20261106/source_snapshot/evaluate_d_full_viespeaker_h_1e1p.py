#!/usr/bin/env python3
"""Evaluate D_FULL on the frozen AAM VieSpeaker-H one-enrollment/one-probe trials.

This server program evaluates duration-controlled one-enrollment/one-probe
verification for the full-data EMA teacher-student checkpoint:

    one 2.0, 3.0, or 4.0 second enrollment -> EMA teacher ECAPA
    one 1.0, 1.25, or 1.5 second probe     -> student ECAPA
    score                                   -> cosine similarity

It does not build a multi-recording enrollment centroid and does not use the
AAM classification head at test time. The program is resumable: extracted
audio, VAD results, embedding shards, the frozen protocol, and score files are
kept on the Samsung T5.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
from functools import lru_cache
from typing import Any, Callable, Iterable, Mapping

os.environ.setdefault("MPLCONFIGDIR", "/tmp/viespeaker_matplotlib_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample_poly
from sklearn.metrics import roc_auc_score, roc_curve
from silero_vad import get_speech_timestamps, load_silero_vad
from speechbrain.inference.speaker import EncoderClassifier
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm


PROJECT_ROOT = Path("/media/ml4u/Samsung_T51/ptnk_team/mortyhuaphuoc")
PINNED_REVISION = "e680954cb58667d5800749ecd9ea0747dfa1e7b4"
EXPECTED_PARQUET_SHARDS = 110
EXPECTED_TRAINING_SPEAKERS = 3371
SAMPLE_RATE = 16_000
SEED = 20260818
MODEL_NAME = "D_FULL"
AAM_OUTPUT_ROOT = (
    PROJECT_ROOT / "experiments/evaluation/aam_only_full_viespeaker_h_1e1p"
)


def build_parser() -> argparse.ArgumentParser:
    """Declare paths, protocol controls, and execution modes."""
    parser = argparse.ArgumentParser(
        description="VieSpeaker-H D_FULL teacher-enrollment/student-probe evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=("check", "smoke", "final"),
        default="check",
        help="check validates inputs, smoke runs 200 trials, final runs 5,000 trials",
    )
    parser.add_argument(
        "--repository",
        type=Path,
        default=PROJECT_ROOT / "datasets/VieSpeaker/repository",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "experiments/mentor_stage/m3_legacy_ema_gaussian_kl/best_validation_eer.pt",
    )
    parser.add_argument(
        "--musan-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets/musan",
    )
    parser.add_argument(
        "--pretrained-dir",
        type=Path,
        default=Path("/home/ml4u/.cache/viespeaker_models/speechbrain_ecapa_voxceleb"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "experiments/evaluation/d_full_viespeaker_h_1e1p",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=PROJECT_ROOT / "experiments/shared/viespeaker_h_1e1p_cache",
    )
    parser.add_argument(
        "--reference-protocol",
        type=Path,
        default=None,
        help=(
            "AAM fixed_trials.csv to reuse byte-for-byte; when omitted, the script "
            "resolves the matching AAM smoke/final protocol automatically"
        ),
    )
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--parquet-batch-size", type=int, default=256)
    parser.add_argument(
        "--enrollment-seconds",
        type=float,
        nargs="+",
        default=[2.0, 3.0, 4.0],
        help="requested voiced durations for the single enrollment recording",
    )
    parser.add_argument(
        "--probe-seconds",
        type=float,
        nargs="+",
        default=[1.0, 1.25, 1.5],
        help="requested voiced durations for the single probe recording",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--skip-bootstrap", action="store_true")
    return parser


def resolve_path(path: Path) -> Path:
    """Resolve a user-supplied server path."""
    return path.expanduser().resolve()


def stable_int(text: str, modulo: int = 2**32 - 1) -> int:
    """Create a deterministic integer from text."""
    return int(hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16], 16) % modulo


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Calculate a file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    """Convert paths and NumPy values into JSON-compatible objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def atomic_json(payload: Any, path: Path) -> None:
    """Write a JSON file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a CSV file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def condition_id(duration: float, snr: float | None) -> str:
    """Return a stable directory-safe condition name."""
    return f"{duration:g}s_" + ("clean" if snr is None else f"{snr:g}dB")


def evaluation_condition_id(
    enrollment_duration: float,
    probe_duration: float,
    snr: float | None,
) -> str:
    """Name one enrollment-duration, probe-duration, and noise combination."""
    return f"enroll{enrollment_duration:g}s_probe{condition_id(probe_duration, snr)}"


def conditions_for_mode(
    mode: str,
    enrollment_seconds: list[float],
    probe_seconds: list[float],
) -> tuple[list[float], list[tuple[float, float | None]]]:
    """Return the controlled enrollment and probe/noise grids."""
    enrollment_durations = sorted(set(enrollment_seconds))
    probe_durations = sorted(set(probe_seconds))
    if mode == "smoke":
        enrollment = min(enrollment_durations, key=lambda value: abs(value - 3.0))
        probe = min(probe_durations, key=lambda value: abs(value - 1.25))
        return [enrollment], [(probe, None), (probe, 10.0)]
    return enrollment_durations, [
        (duration, snr)
        for duration in probe_durations
        for snr in (None, 10.0, 0.0)
    ]


def normalize_official_key(value: str) -> str:
    """Normalize a VieSpeaker official recording key."""
    normalized = str(value).replace("\\", "/").lstrip("./")
    if not normalized or normalized.startswith("/") or ".." in Path(normalized).parts:
        raise ValueError(f"Unsafe or empty VieSpeaker key: {value!r}")
    return normalized


def row_key(speaker_id: Any, audio_path: Any) -> str:
    """Reconstruct an official key from one Parquet row."""
    speaker = str(speaker_id)
    path = normalize_official_key(str(audio_path))
    if path.startswith(speaker + "/"):
        return path
    return f"{speaker}/{Path(path).name}"


def find_musan_category(musan_root: Path, category: str) -> list[Path]:
    """Find MUSAN audio for one category under common directory layouts."""
    candidates = [musan_root / category, musan_root / "musan" / category]
    extensions = {".wav", ".flac", ".ogg"}
    for root in candidates:
        if root.is_dir():
            files = sorted(path for path in root.rglob("*") if path.suffix.lower() in extensions)
            if files:
                return files
    return []


def deterministic_evaluation_noise(musan_root: Path) -> dict[str, list[Path]]:
    """Create the historical deterministic 80/20 train/evaluation MUSAN split."""
    evaluation: dict[str, list[Path]] = {}
    for category in ("noise", "music", "speech"):
        files = find_musan_category(musan_root, category)
        held_out = [
            path
            for path in files
            if stable_int(path.relative_to(musan_root).as_posix(), 10_000) >= 8_000
        ]
        if not held_out and files:
            held_out = [files[-1]]
        evaluation[category] = held_out
    return evaluation


def resolve_ema_teacher_state(
    checkpoint: Mapping[str, Any],
) -> tuple[str, Mapping[str, torch.Tensor]]:
    """Find the explicitly saved EMA-teacher state without silent substitution."""
    candidates = []
    for key in ("teacher", "ema_teacher", "teacher_ema"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping) and value:
            candidates.append((key, value))
    if not candidates:
        available = sorted(str(key) for key in checkpoint.keys())
        raise KeyError(
            "D_FULL checkpoint has no EMA teacher state under any supported key "
            f"('teacher', 'ema_teacher', 'teacher_ema'). Available keys: {available}"
        )
    if len(candidates) > 1:
        keys = [key for key, _ in candidates]
        raise RuntimeError(f"Ambiguous EMA teacher states in checkpoint: {keys}")
    return candidates[0]


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    """Validate the full-data D_FULL checkpoint without modifying it."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    system_name = str(checkpoint.get("system_name", checkpoint.get("system", "")))
    if system_name != "D_FULL":
        raise RuntimeError(f"Unexpected checkpoint system {system_name!r}")
    if not isinstance(checkpoint.get("student"), dict):
        raise KeyError("Checkpoint does not contain student encoder weights")
    teacher_key, teacher_state = resolve_ema_teacher_state(checkpoint)
    signature = checkpoint.get("signature", {})
    speaker_ids = checkpoint.get("train_speaker_ids", signature.get("train_speaker_ids", []))
    if speaker_ids and len(speaker_ids) != EXPECTED_TRAINING_SPEAKERS:
        raise RuntimeError(
            f"Expected {EXPECTED_TRAINING_SPEAKERS:,} training speakers, "
            f"checkpoint records {len(speaker_ids):,}"
        )
    result = {
        "system_name": system_name,
        "global_step": int(checkpoint.get("global_step", -1)),
        "best_noise_eer": float(checkpoint.get("best_noise_eer", np.nan)),
        "training_speakers": len(speaker_ids) if speaker_ids else None,
        "has_student": True,
        "has_ema_teacher": bool(teacher_state),
        "ema_teacher_key": teacher_key,
    }
    del checkpoint
    return result


def check_inputs(args: argparse.Namespace) -> dict[str, Any]:
    """Inspect the local repository, checkpoint, model cache, and MUSAN data."""
    data_dir = args.repository / "data"
    trial_file = args.repository / "trials" / "viespeaker_H.txt"
    shards = sorted(data_dir.glob("part-*-of-00110.parquet"))
    noise = deterministic_evaluation_noise(args.musan_dir) if args.musan_dir.is_dir() else {}

    report: dict[str, Any] = {
        "repository": args.repository,
        "repository_exists": args.repository.is_dir(),
        "parquet_shards": len(shards),
        "expected_parquet_shards": EXPECTED_PARQUET_SHARDS,
        "viespeaker_h_trials": trial_file,
        "viespeaker_h_trials_exists": trial_file.is_file(),
        "checkpoint": args.checkpoint,
        "checkpoint_exists": args.checkpoint.is_file(),
        "musan_dir": args.musan_dir,
        "musan_dir_exists": args.musan_dir.is_dir(),
        "heldout_musan_counts": {key: len(value) for key, value in noise.items()},
        "pretrained_dir": args.pretrained_dir,
        "pretrained_dir_exists": args.pretrained_dir.is_dir(),
        "reference_protocol": args.reference_protocol,
        "reference_protocol_exists": (
            args.reference_protocol.is_file()
            if args.reference_protocol is not None
            else False
        ),
        "pinned_revision_from_handoff": PINNED_REVISION,
    }
    if trial_file.is_file():
        report["viespeaker_h_trial_file_size_mb"] = trial_file.stat().st_size / 2**20
        report["viespeaker_h_trial_sha256"] = sha256_file(trial_file)
    if args.checkpoint.is_file():
        report["checkpoint_metadata"] = inspect_checkpoint(args.checkpoint)
        report["checkpoint_sha256"] = sha256_file(args.checkpoint)

    problems = []
    if len(shards) != EXPECTED_PARQUET_SHARDS:
        problems.append(
            f"found {len(shards)} of {EXPECTED_PARQUET_SHARDS} expected Parquet shards"
        )
    if not trial_file.is_file():
        problems.append(f"missing official H trial file: {trial_file}")
    if not args.checkpoint.is_file():
        problems.append(f"missing checkpoint: {args.checkpoint}")
    if not args.musan_dir.is_dir():
        problems.append(f"missing MUSAN directory: {args.musan_dir}")
    elif any(not noise.get(category) for category in ("noise", "music", "speech")):
        problems.append("MUSAN must contain nonempty noise, music, and speech categories")
    if not args.pretrained_dir.is_dir():
        problems.append(f"missing local pretrained ECAPA directory: {args.pretrained_dir}")
    if args.reference_protocol is None or not args.reference_protocol.is_file():
        problems.append(
            "missing matched AAM fixed_trials.csv; D_FULL is not allowed to create "
            "a new evaluation protocol"
        )
    report["problems"] = problems
    report["ready"] = not problems
    return report


def load_official_trials(trial_file: Path) -> pd.DataFrame:
    """Load and validate the official VieSpeaker-H target/nontarget pairs."""
    trials = pd.read_csv(
        trial_file,
        sep=r"\s+",
        names=("enrollment", "probe", "label_text"),
        engine="python",
    )
    trials["enrollment"] = trials.enrollment.map(normalize_official_key)
    trials["probe"] = trials.probe.map(normalize_official_key)
    labels = trials.label_text.astype(str).str.lower()
    trials["label"] = labels.map({"target": 1, "nontarget": 0, "1": 1, "0": 0})
    if trials.label.isna().any():
        bad = sorted(labels[trials.label.isna()].unique())
        raise ValueError(f"Unexpected official labels: {bad}")
    trials["label"] = trials.label.astype(np.int64)
    trials.insert(0, "trial_id", np.arange(len(trials), dtype=np.int64))
    return trials


def freeze_protocol(
    all_trials: pd.DataFrame,
    output_path: Path,
    requested_trials: int,
    seed: int,
    reference_protocol: Path | None,
) -> pd.DataFrame:
    """Reuse the immutable AAM trial subset and reject any protocol mismatch."""
    if reference_protocol is None or not reference_protocol.is_file():
        raise FileNotFoundError(
            "A matched AAM fixed_trials.csv is required for D_FULL evaluation"
        )
    if output_path.is_file():
        if sha256_file(output_path) != sha256_file(reference_protocol):
            raise RuntimeError(
                "Existing D_FULL fixed_trials.csv does not match the AAM reference: "
                f"{output_path} vs {reference_protocol}"
            )
        frozen = pd.read_csv(output_path)
        source = "existing checksum-matched copy of AAM protocol"
    else:
        frozen = pd.read_csv(reference_protocol)
        atomic_csv(frozen, output_path)
        source = f"reference protocol {reference_protocol}"

    for column in ("trial_id", "enrollment", "probe", "label"):
        if column not in frozen:
            raise KeyError(f"Frozen protocol lacks required column {column!r}")
    if len(frozen) != requested_trials:
        raise RuntimeError(
            f"Frozen protocol contains {len(frozen):,} trials, expected {requested_trials:,}"
        )
    counts = frozen.label.value_counts().to_dict()
    expected = {0: requested_trials // 2, 1: requested_trials // 2}
    if counts != expected:
        raise RuntimeError(f"Unbalanced frozen protocol: {counts}, expected {expected}")
    if not frozen.trial_id.is_unique:
        raise RuntimeError("Frozen protocol has duplicate trial IDs")
    if not (frozen.enrollment != frozen.probe).all():
        raise RuntimeError("At least one trial reuses the same recording on both sides")
    print(f"Protocol source: {source}")
    print(f"Frozen protocol: {output_path} ({len(frozen):,} trials)")
    return frozen


def raw_audio_path(raw_root: Path, key: str) -> Path:
    """Map an official recording key to a safe cache path."""
    normalized = normalize_official_key(key)
    return raw_root.joinpath(*Path(normalized).parts)


def audio_payload(row: Mapping[str, Any]) -> tuple[str, bytes] | None:
    """Read the official key and encoded audio bytes from a Parquet row."""
    audio = row.get("audio")
    if not isinstance(audio, Mapping):
        return None
    payload = audio.get("bytes")
    path = audio.get("path")
    speaker_id = row.get("speaker_id")
    if payload is None or path is None or speaker_id is None:
        return None
    if isinstance(payload, memoryview):
        payload = payload.tobytes()
    if not isinstance(payload, bytes):
        payload = bytes(payload)
    return row_key(speaker_id, path), payload


def extract_required_audio(
    shards: list[Path],
    required_keys: set[str],
    raw_root: Path,
    parquet_batch_size: int,
) -> dict[str, Any]:
    """Extract only frozen-protocol recordings from the local Parquet repository."""
    raw_root.mkdir(parents=True, exist_ok=True)
    normalized_required = {normalize_official_key(key) for key in required_keys}
    existing = {
        key for key in normalized_required if raw_audio_path(raw_root, key).is_file()
    }
    missing = normalized_required - existing
    print(
        f"Required unique recordings: {len(normalized_required):,}; "
        f"already cached: {len(existing):,}; remaining: {len(missing):,}"
    )
    started = time.perf_counter()
    rows_scanned = 0
    files_written = 0

    if missing:
        for shard in tqdm(shards, desc="Scanning local VieSpeaker Parquet shards"):
            parquet = pq.ParquetFile(shard)
            for batch in parquet.iter_batches(
                batch_size=parquet_batch_size,
                columns=["audio", "speaker_id"],
            ):
                rows = batch.to_pylist()
                rows_scanned += len(rows)
                for row in rows:
                    item = audio_payload(row)
                    if item is None:
                        continue
                    key, payload = item
                    if key not in missing:
                        continue
                    destination = raw_audio_path(raw_root, key)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_name(destination.name + ".tmp")
                    temporary.write_bytes(payload)
                    os.replace(temporary, destination)
                    missing.remove(key)
                    files_written += 1
                if not missing:
                    break
            if not missing:
                break

    if missing:
        examples = sorted(missing)[:10]
        raise RuntimeError(
            f"Could not find {len(missing):,} required recordings in the 110 local shards. "
            f"Examples: {examples}"
        )
    return {
        "required_recordings": len(normalized_required),
        "previously_cached": len(existing),
        "written_this_run": files_written,
        "parquet_rows_scanned_this_run": rows_scanned,
        "elapsed_minutes": (time.perf_counter() - started) / 60.0,
    }


def to_mono_16k(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """Convert decoded audio to finite mono 16 kHz float32."""
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if waveform.ndim != 1:
        raise ValueError(f"Unexpected waveform shape {waveform.shape}")
    waveform = np.nan_to_num(waveform, copy=False)
    if int(sample_rate) != SAMPLE_RATE:
        divisor = math.gcd(int(sample_rate), SAMPLE_RATE)
        waveform = resample_poly(
            waveform,
            SAMPLE_RATE // divisor,
            int(sample_rate) // divisor,
        ).astype(np.float32)
    return np.ascontiguousarray(waveform, dtype=np.float32)


def prepared_audio_path(prepared_root: Path, key: str) -> Path:
    """Map an official key to its compressed VAD cache."""
    relative = Path(normalize_official_key(key)).with_suffix(".npz")
    return prepared_root.joinpath(*relative.parts)


def prepare_vad_cache(
    keys: Iterable[str],
    raw_root: Path,
    prepared_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Silero VAD once per required recording and cache the result."""
    prepared_root.mkdir(parents=True, exist_ok=True)
    vad_model = load_silero_vad(onnx=False)
    quality_rows = []
    failure_rows = []
    for key in tqdm(sorted(set(keys)), desc="Preparing VAD cache"):
        output = prepared_audio_path(prepared_root, key)
        try:
            if output.is_file():
                stored = np.load(output, allow_pickle=False)
                quality_rows.append(
                    {
                        "path": key,
                        "source_seconds": float(stored["source_seconds"]),
                        "voiced_seconds": float(stored["voiced_seconds"]),
                        "vad_segments": len(stored["intervals"]),
                        "status": "cached",
                    }
                )
                continue

            waveform, sample_rate = sf.read(
                raw_audio_path(raw_root, key),
                dtype="float32",
                always_2d=False,
            )
            waveform = to_mono_16k(waveform, sample_rate)
            stamps = get_speech_timestamps(
                torch.from_numpy(waveform),
                vad_model,
                sampling_rate=SAMPLE_RATE,
                threshold=0.50,
                min_speech_duration_ms=100,
                min_silence_duration_ms=100,
                speech_pad_ms=30,
            )
            intervals = np.asarray(
                [[int(item["start"]), int(item["end"])] for item in stamps],
                dtype=np.int64,
            ).reshape(-1, 2)
            voiced_samples = int(sum(end - start for start, end in intervals))
            if voiced_samples <= 0:
                raise RuntimeError("no speech detected")
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(output.name + ".tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    waveform=waveform,
                    intervals=intervals,
                    source_seconds=np.float32(len(waveform) / SAMPLE_RATE),
                    voiced_seconds=np.float32(voiced_samples / SAMPLE_RATE),
                )
            os.replace(temporary, output)
            quality_rows.append(
                {
                    "path": key,
                    "source_seconds": len(waveform) / SAMPLE_RATE,
                    "voiced_seconds": voiced_samples / SAMPLE_RATE,
                    "vad_segments": len(intervals),
                    "status": "prepared",
                }
            )
        except Exception as exc:
            failure_rows.append({"path": key, "error": repr(exc)})
    return pd.DataFrame(quality_rows), pd.DataFrame(failure_rows)


def load_prepared(prepared_root: Path, key: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one VAD-prepared waveform and its intervals."""
    stored = np.load(prepared_audio_path(prepared_root, key), allow_pickle=False)
    return stored["waveform"].astype(np.float32), stored["intervals"].astype(np.int64)


def peak_guard(waveform: np.ndarray, peak: float = 0.99) -> np.ndarray:
    """Prevent clipping after noise mixing."""
    waveform = np.asarray(waveform, dtype=np.float32)
    maximum = float(np.max(np.abs(waveform)) + 1e-8)
    if maximum > peak:
        waveform = waveform * (peak / maximum)
    return waveform.astype(np.float32)


def crop_by_voiced_duration(
    waveform: np.ndarray,
    intervals: np.ndarray,
    seconds: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select a contiguous crop containing the requested cumulative voiced duration."""
    needed = int(round(seconds * SAMPLE_RATE))
    lengths = np.asarray([end - start for start, end in intervals], dtype=np.int64)
    total = int(lengths.sum())
    if total < needed or len(intervals) == 0:
        raise ValueError(f"need {needed} voiced samples, found {total}")
    cumulative = np.concatenate([[0], np.cumsum(lengths)])
    voiced_start = int(rng.integers(0, total - needed + 1))
    voiced_end = voiced_start + needed
    start_interval = int(
        np.searchsorted(cumulative[1:], voiced_start, side="right")
    )
    end_interval = int(
        np.searchsorted(cumulative[1:], voiced_end - 1, side="right")
    )
    raw_start = int(
        intervals[start_interval][0] + voiced_start - cumulative[start_interval]
    )
    raw_end = int(
        intervals[end_interval][0] + voiced_end - cumulative[end_interval]
    )
    return np.asarray(
        waveform[max(0, raw_start) : min(len(waveform), raw_end)],
        dtype=np.float32,
    )


def fit_noise(
    waveform: np.ndarray,
    length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Repeat and crop a noise waveform to a target length."""
    if len(waveform) < length:
        waveform = np.tile(waveform, math.ceil(length / len(waveform)))
    start = int(rng.integers(0, len(waveform) - length + 1))
    return np.asarray(waveform[start : start + length], dtype=np.float32)


def transform_recipe(
    key: str,
    duration: float,
    snr: float | None,
    seed: int,
    evaluation_noise: Mapping[str, list[Path]],
) -> dict[str, Any]:
    """Create the deterministic historical crop and corruption recipe."""
    crop_seed = stable_int(f"crop|{seed}|{key}|{duration:g}")
    if snr is None:
        return {
            "crop_seed": crop_seed,
            "category": "clean",
            "noise_seed": None,
            "noise_files": [],
        }
    noise_rng = np.random.default_rng(
        stable_int(f"noise|{seed}|{key}|{duration:g}|{snr:g}")
    )
    category = ("noise", "music", "speech")[int(noise_rng.integers(0, 3))]
    count = 3 if category == "speech" else 1
    pool = evaluation_noise[category]
    files = [pool[int(noise_rng.integers(0, len(pool)))] for _ in range(count)]
    return {
        "crop_seed": crop_seed,
        "category": category,
        "noise_seed": stable_int(f"noise-fit|{seed}|{key}|{duration:g}|{snr:g}"),
        "noise_files": files,
    }


def make_enrollment(
    prepared_root: Path,
    key: str,
    duration: float,
    seed: int,
) -> np.ndarray:
    """Create one deterministic unaugmented enrollment crop."""
    waveform, intervals = load_prepared(prepared_root, key)
    crop_seed = stable_int(f"enrollment-crop|{seed}|{key}|{duration:g}")
    return peak_guard(
        crop_by_voiced_duration(
            waveform,
            intervals,
            duration,
            np.random.default_rng(crop_seed),
        )
    )


@lru_cache(maxsize=256)
def load_noise_waveform(path: str) -> np.ndarray:
    """Reuse decoded MUSAN waveforms across thousands of probe crops."""
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    return to_mono_16k(waveform, sample_rate)


def make_probe(
    prepared_root: Path,
    key: str,
    duration: float,
    snr: float | None,
    seed: int,
    evaluation_noise: Mapping[str, list[Path]],
) -> np.ndarray:
    """Create one deterministic clean or MUSAN-corrupted short probe."""
    recipe = transform_recipe(key, duration, snr, seed, evaluation_noise)
    waveform, intervals = load_prepared(prepared_root, key)
    clean = crop_by_voiced_duration(
        waveform,
        intervals,
        duration,
        np.random.default_rng(recipe["crop_seed"]),
    )
    if snr is None:
        return peak_guard(clean)

    rng = np.random.default_rng(recipe["noise_seed"])
    noises = []
    for path in recipe["noise_files"]:
        noise = load_noise_waveform(str(path))
        noises.append(fit_noise(noise, len(clean), rng))
    mixed_noise = np.mean(noises, axis=0).astype(np.float32)
    clean_power = float(np.mean(clean**2) + 1e-8)
    noise_power = float(np.mean(mixed_noise**2) + 1e-8)
    scale = math.sqrt(clean_power / (noise_power * 10 ** (float(snr) / 10.0)))
    return peak_guard(clean + scale * mixed_noise)


class ECAPAEncoder(nn.Module):
    """Training-compatible SpeechBrain ECAPA wrapper."""

    def __init__(
        self,
        compute_features: nn.Module,
        mean_var_norm: nn.Module,
        embedding_model: nn.Module,
    ) -> None:
        super().__init__()
        self.compute_features = compute_features
        self.mean_var_norm = mean_var_norm
        self.embedding_model = embedding_model

    def forward(self, waveforms: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Encode padded waveforms into 192-dimensional embeddings."""
        features = self.compute_features(waveforms.float())
        features = self.mean_var_norm(features, lengths)
        embeddings = self.embedding_model(features, lengths)
        if embeddings.ndim == 3 and embeddings.shape[1] == 1:
            embeddings = embeddings[:, 0, :]
        return embeddings


def load_pretrained_encoder(pretrained_dir: Path) -> ECAPAEncoder:
    """Load the same pretrained ECAPA modules used during fine-tuning."""
    pretrained = EncoderClassifier.from_hparams(
        source=str(pretrained_dir),
        savedir=str(pretrained_dir),
        run_opts={"device": "cpu"},
        freeze_params=False,
    )
    model = ECAPAEncoder(
        copy.deepcopy(pretrained.mods.compute_features),
        copy.deepcopy(pretrained.mods.mean_var_norm),
        copy.deepcopy(pretrained.mods.embedding_model),
    )
    del pretrained
    gc.collect()
    return model


def embedding_cache_folder(
    cache_root: Path,
    checkpoint_hash: str,
    protocol_hash: str,
    role: str,
    condition: str,
    batch_size: int,
) -> Path:
    """Build a checkpoint-specific resumable embedding-cache path."""
    return (
        cache_root
        / "embeddings"
        / f"{MODEL_NAME}_{checkpoint_hash[:12]}"
        / f"protocol_{protocol_hash[:12]}"
        / role
        / condition
        / f"bs{batch_size}"
    )


def save_embedding_shard(
    path: Path,
    keys: list[str],
    embeddings: np.ndarray,
) -> None:
    """Save one embedding batch atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            keys=np.asarray(keys),
            embeddings=np.asarray(embeddings, dtype=np.float32),
        )
    os.replace(temporary, path)


@torch.inference_mode()
def encode_cached(
    model: nn.Module,
    keys: Iterable[str],
    waveform_function: Callable[[str], np.ndarray],
    folder: Path,
    description: str,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, str]]:
    """Extract normalized embeddings with one resumable file per batch."""
    model = model.to(device).eval()
    ordered_keys = sorted(set(keys))
    failures = []
    for start in tqdm(range(0, len(ordered_keys), batch_size), desc=description):
        batch_keys = ordered_keys[start : start + batch_size]
        shard = folder / f"{start:08d}.npz"
        if shard.is_file():
            stored = np.load(shard, allow_pickle=False)
            if (
                list(stored["keys"].astype(str)) == batch_keys
                and stored["embeddings"].ndim == 2
                and stored["embeddings"].shape[1] == 192
            ):
                continue
            raise RuntimeError(f"Stale or incompatible embedding shard: {shard}")

        waveforms = []
        valid_keys = []
        for key in batch_keys:
            try:
                waveforms.append(torch.from_numpy(waveform_function(key)))
                valid_keys.append(key)
            except Exception as exc:
                failures.append({"path": key, "error": repr(exc)})
        if not waveforms:
            save_embedding_shard(shard, [], np.empty((0, 192), dtype=np.float32))
            continue

        lengths = torch.tensor([len(waveform) for waveform in waveforms], dtype=torch.float32)
        padded = pad_sequence(waveforms, batch_first=True)
        relative_lengths = (lengths / padded.shape[1]).to(device)
        embeddings = F.normalize(
            model(padded.to(device), relative_lengths),
            dim=-1,
        ).cpu().numpy()
        if embeddings.shape[1] != 192 or not np.isfinite(embeddings).all():
            raise RuntimeError(f"Invalid embedding matrix: {embeddings.shape}")
        save_embedding_shard(shard, valid_keys, embeddings)
    return failures


def load_embedding_map(folder: Path) -> dict[str, np.ndarray]:
    """Load every resumable embedding shard in one condition."""
    result = {}
    for shard in sorted(folder.glob("*.npz")):
        stored = np.load(shard, allow_pickle=False)
        for key, embedding in zip(
            stored["keys"].astype(str), stored["embeddings"], strict=True
        ):
            result[key] = embedding
    return result


def verification_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Compute global EER, minDCF, ROC AUC, and TAR at FAR 1%."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(scores)
    labels, scores = labels[finite], scores[finite]
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Target and nontarget trials are both required")

    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        labels, scores, pos_label=1
    )
    false_negative_rate = 1.0 - true_positive_rate
    difference = false_negative_rate - false_positive_rate
    crossings = np.flatnonzero(difference[:-1] * difference[1:] <= 0)
    if len(crossings):
        index = int(crossings[0])
        denominator = difference[index] - difference[index + 1]
        weight = difference[index] / denominator if denominator else 0.0
        eer = false_positive_rate[index] + weight * (
            false_positive_rate[index + 1] - false_positive_rate[index]
        )
        eer_threshold = thresholds[index] + weight * (
            thresholds[index + 1] - thresholds[index]
        )
    else:
        index = int(np.argmin(np.abs(difference)))
        eer = 0.5 * (false_negative_rate[index] + false_positive_rate[index])
        eer_threshold = thresholds[index]

    p_target = 0.01
    costs = (
        p_target * false_negative_rate
        + (1.0 - p_target) * false_positive_rate
    )
    minimum_index = int(np.argmin(costs))
    allowed = np.flatnonzero(false_positive_rate <= 0.01)
    return {
        "eer_percent": 100.0 * float(eer),
        "eer_threshold": float(eer_threshold),
        "min_dcf": float(costs[minimum_index] / p_target),
        "min_dcf_threshold": float(thresholds[minimum_index]),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "tar_at_far_1_percent": (
            float(np.max(true_positive_rate[allowed])) if len(allowed) else 0.0
        ),
        "empirical_far_resolution": 1.0 / int((labels == 0).sum()),
    }


def bootstrap_eer_intervals(
    scores: pd.DataFrame,
    summary: pd.DataFrame,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Estimate EER uncertainty by resampling enrollment recordings."""
    rows = []
    rng = np.random.default_rng(seed + 909)
    for condition, group in scores.groupby("condition"):
        enrollment_array = group.enrollment.to_numpy()
        clusters = np.unique(enrollment_array)
        indices_by_cluster = {
            cluster: np.flatnonzero(enrollment_array == cluster) for cluster in clusters
        }
        values = []
        for _ in tqdm(range(replicates), desc=f"bootstrap {condition}", leave=False):
            sampled = rng.choice(clusters, len(clusters), replace=True)
            indices = np.concatenate(
                [indices_by_cluster[cluster] for cluster in sampled]
            )
            labels = group.label.to_numpy()[indices]
            if np.unique(labels).size < 2:
                continue
            values.append(
                verification_metrics(labels, group.score.to_numpy()[indices])[
                    "eer_percent"
                ]
            )
        series = pd.Series(values, dtype=float)
        observed = float(
            summary.loc[summary.condition == condition, "eer_percent"].iloc[0]
        )
        rows.append(
            {
                "condition": condition,
                "model": MODEL_NAME,
                "observed_eer_percent": observed,
                "replicates": len(series),
                "eer_ci_low": series.quantile(0.025),
                "eer_ci_high": series.quantile(0.975),
                "bootstrap_unit": "enrollment recording",
            }
        )
    return pd.DataFrame(rows)


def save_plot(summary: pd.DataFrame, output_path: Path) -> None:
    """Save the paper-ready EER duration/noise chart."""
    plot_data = summary.copy()
    plot_data["noise"] = plot_data.snr_db.apply(
        lambda value: "clean" if pd.isna(value) else f"{value:g} dB"
    )
    chart = sns.relplot(
        data=plot_data,
        x="probe_duration_s",
        y="eer_percent",
        hue="noise",
        col="enrollment_duration_s",
        kind="line",
        marker="o",
        height=3.8,
        aspect=1.05,
    )
    chart.set_axis_labels("Probe duration (s)", "EER (%)")
    chart.set_titles("Enrollment: {col_name} s")
    chart.fig.subplots_adjust(top=0.88)
    chart.fig.suptitle(
        "D_FULL: EMA-teacher enrollment / student-probe evaluation"
    )
    chart.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(chart.fig)


def seed_everything(seed: int) -> None:
    """Set deterministic random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_evaluation(args: argparse.Namespace, report: Mapping[str, Any]) -> None:
    """Prepare data, embed audio, score trials, and save final metrics."""
    if not report["ready"]:
        raise RuntimeError("Input check failed:\n- " + "\n- ".join(report["problems"]))
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable; pass --allow-cpu only for debugging")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enrollment_durations, probe_conditions = conditions_for_mode(
        args.mode,
        args.enrollment_seconds,
        args.probe_seconds,
    )
    requested_trials = 200 if args.mode == "smoke" else 5_000
    bootstrap_replicates = (
        args.bootstrap_replicates
        if args.bootstrap_replicates is not None
        else (20 if args.mode == "smoke" else 300)
    )
    run_dir = args.output_root / f"{args.mode}_seed{args.seed}_n{requested_trials}"
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = run_dir / "fixed_trials.csv"
    trial_file = args.repository / "trials" / "viespeaker_H.txt"
    shards = sorted((args.repository / "data").glob("part-*-of-00110.parquet"))

    all_trials = load_official_trials(trial_file)
    frozen = freeze_protocol(
        all_trials,
        protocol_path,
        requested_trials,
        args.seed,
        args.reference_protocol,
    )
    protocol_hash = sha256_file(protocol_path)
    required_keys = set(frozen.enrollment) | set(frozen.probe)
    protocol_metadata = {
        "seed": args.seed,
        "requested_trials": requested_trials,
        "official_trial_sha256": sha256_file(trial_file),
        "frozen_protocol_sha256": protocol_hash,
        "target": int((frozen.label == 1).sum()),
        "nontarget": int((frozen.label == 0).sum()),
        "unique_recordings": len(required_keys),
        "enrollment_recordings_per_trial": 1,
        "probe_recordings_per_trial": 1,
        "enrollment_durations_seconds": enrollment_durations,
        "probe_durations_seconds": sorted(
            {duration for duration, _ in probe_conditions}
        ),
        "uses_enrollment_centroid": False,
    }
    atomic_json(protocol_metadata, run_dir / "protocol_config.json")

    raw_root = args.cache_root / "raw_audio"
    extraction = extract_required_audio(
        shards,
        required_keys,
        raw_root,
        args.parquet_batch_size,
    )
    atomic_json(extraction, run_dir / "extraction_summary.json")
    print(json.dumps(extraction, indent=2))

    prepared_root = args.cache_root / "prepared_audio"
    quality, preparation_failures = prepare_vad_cache(
        required_keys,
        raw_root,
        prepared_root,
    )
    atomic_csv(quality, run_dir / "audio_quality.csv")
    atomic_csv(preparation_failures, run_dir / "audio_preparation_failures.csv")
    quality_map = quality.set_index("path").to_dict("index")
    print(
        f"VAD prepared: {len(quality):,}/{len(required_keys):,}; "
        f"failures: {len(preparation_failures):,}"
    )

    evaluation_noise = deterministic_evaluation_noise(args.musan_dir)
    transform_rows = []
    for duration, snr in probe_conditions:
        for key in sorted(set(frozen.probe)):
            if key not in quality_map:
                continue
            if quality_map[key]["voiced_seconds"] + 1e-6 < duration:
                continue
            recipe = transform_recipe(
                key, duration, snr, args.seed, evaluation_noise
            )
            transform_rows.append(
                {
                    "probe": key,
                    "duration_s": duration,
                    "snr_db": snr,
                    "condition": condition_id(duration, snr),
                    "crop_seed": recipe["crop_seed"],
                    "noise_category": recipe["category"],
                    "noise_seed": recipe["noise_seed"],
                    "noise_files": json.dumps(
                        [str(path) for path in recipe["noise_files"]]
                    ),
                }
            )
    atomic_csv(
        pd.DataFrame(transform_rows),
        run_dir / "probe_transform_manifest.csv",
    )

    checkpoint_hash = sha256_file(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    teacher_key, teacher_state = resolve_ema_teacher_state(checkpoint)
    student_state = checkpoint.get("student")
    if not isinstance(student_state, Mapping) or not student_state:
        raise KeyError("D_FULL checkpoint does not contain student encoder weights")

    enrollment_encoder = load_pretrained_encoder(args.pretrained_dir)
    enrollment_encoder.load_state_dict(teacher_state, strict=True)
    del teacher_state

    probe_encoder = load_pretrained_encoder(args.pretrained_dir)
    probe_encoder.load_state_dict(student_state, strict=True)
    del student_state
    del checkpoint

    enrollment_keys = sorted(set(frozen.enrollment) & set(quality_map))
    probe_keys = sorted(set(frozen.probe) & set(quality_map))
    cache_index: dict[str, str] = {}
    embedding_failures = []

    for enrollment_duration in enrollment_durations:
        enrollment_condition = f"enrollment_{enrollment_duration:g}s"
        eligible_enrollments = [
            key
            for key in enrollment_keys
            if quality_map[key]["voiced_seconds"] + 1e-6 >= enrollment_duration
        ]
        enrollment_folder = embedding_cache_folder(
            args.cache_root,
            checkpoint_hash,
            protocol_hash,
            "ema_teacher",
            enrollment_condition,
            args.embedding_batch_size,
        )
        embedding_failures.extend(
            encode_cached(
                enrollment_encoder,
                eligible_enrollments,
                lambda key, duration=enrollment_duration: make_enrollment(
                    prepared_root,
                    key,
                    duration,
                    args.seed,
                ),
                enrollment_folder,
                f"{MODEL_NAME} enrollment {enrollment_duration:g}s",
                args.embedding_batch_size,
                device,
            )
        )
        cache_index[enrollment_condition] = str(enrollment_folder)

    del enrollment_encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for duration, snr in probe_conditions:
        condition = condition_id(duration, snr)
        eligible = [
            key
            for key in probe_keys
            if quality_map[key]["voiced_seconds"] + 1e-6 >= duration
        ]
        folder = embedding_cache_folder(
            args.cache_root,
            checkpoint_hash,
            protocol_hash,
            "student",
            condition,
            args.embedding_batch_size,
        )
        embedding_failures.extend(
            encode_cached(
                probe_encoder,
                eligible,
                lambda key, duration=duration, snr=snr: make_probe(
                    prepared_root,
                    key,
                    duration,
                    snr,
                    args.seed,
                    evaluation_noise,
                ),
                folder,
                f"{MODEL_NAME} {condition}",
                args.embedding_batch_size,
                device,
            )
        )
        cache_index[condition] = str(folder)

    del probe_encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    atomic_json(cache_index, run_dir / "embedding_cache_index.json")
    atomic_csv(pd.DataFrame(embedding_failures), run_dir / "embedding_failures.csv")

    summary_rows = []
    score_frames = []
    valid_frames = []
    for enrollment_duration in enrollment_durations:
        enrollment_key = f"enrollment_{enrollment_duration:g}s"
        enrollment_embeddings = load_embedding_map(Path(cache_index[enrollment_key]))
        for duration, snr in probe_conditions:
            probe_condition = condition_id(duration, snr)
            condition = evaluation_condition_id(enrollment_duration, duration, snr)
            probe_embeddings = load_embedding_map(Path(cache_index[probe_condition]))
            valid = frozen[
                frozen.enrollment.isin(enrollment_embeddings)
                & frozen.probe.isin(probe_embeddings)
            ].copy().reset_index(drop=True)
            if set(valid.label.unique()) != {0, 1}:
                raise RuntimeError(f"{condition}: valid trials lack both classes")
            valid["condition"] = condition
            valid["enrollment_duration_s"] = enrollment_duration
            valid["probe_duration_s"] = duration
            valid["duration_s"] = duration
            valid["snr_db"] = snr
            labels = valid.label.to_numpy(np.int64)
            scores = np.asarray(
                [
                    float(
                        np.dot(
                            enrollment_embeddings[enrollment],
                            probe_embeddings[probe],
                        )
                    )
                    for enrollment, probe in valid[
                        ["enrollment", "probe"]
                    ].itertuples(index=False, name=None)
                ],
                dtype=np.float64,
            )
            metrics = verification_metrics(labels, scores)
            summary_rows.append(
                {
                    "model": MODEL_NAME,
                    "enrollment_encoder": "ema_teacher",
                    "probe_encoder": "student",
                    "enrollment_recordings_per_trial": 1,
                    "probe_recordings_per_trial": 1,
                    "uses_enrollment_centroid": False,
                    "condition": condition,
                    "enrollment_duration_s": enrollment_duration,
                    "probe_duration_s": duration,
                    "duration_s": duration,
                    "snr_db": snr,
                    "requested_trials": len(frozen),
                    "scored_trials": len(valid),
                    "target_trials": int(labels.sum()),
                    "nontarget_trials": int((labels == 0).sum()),
                    "trial_exclusion_percent": 100.0
                    * (1.0 - len(valid) / len(frozen)),
                    **metrics,
                }
            )
            score_frame = valid[
                [
                    "trial_id",
                    "label",
                    "enrollment",
                    "probe",
                    "condition",
                    "enrollment_duration_s",
                    "probe_duration_s",
                    "duration_s",
                    "snr_db",
                ]
            ].copy()
            score_frame.insert(0, "model", MODEL_NAME)
            score_frame["score"] = scores
            score_frames.append(score_frame)
            valid_frames.append(valid)

    summary = pd.DataFrame(summary_rows)
    scores = pd.concat(score_frames, ignore_index=True)
    valid_trials = pd.concat(valid_frames, ignore_index=True)
    atomic_csv(summary, run_dir / "summary_metrics.csv")
    atomic_csv(scores, run_dir / "scores_d_full.csv")
    atomic_csv(valid_trials, run_dir / "valid_trials_by_condition.csv")
    print(
        summary[
            [
                "condition",
                "enrollment_duration_s",
                "probe_duration_s",
                "eer_percent",
                "min_dcf",
                "roc_auc",
                "tar_at_far_1_percent",
                "scored_trials",
            ]
        ].round(5).to_string(index=False)
    )

    if not args.skip_bootstrap:
        confidence = bootstrap_eer_intervals(
            scores,
            summary,
            bootstrap_replicates,
            args.seed,
        )
        atomic_csv(confidence, run_dir / "confidence_intervals.csv")
        print(confidence.round(4).to_string(index=False))

    save_plot(summary, run_dir / "eer_by_duration_and_snr.png")
    run_metadata = {
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": MODEL_NAME,
        "enrollment_encoder": "ema_teacher",
        "probe_encoder": "student",
        "ema_teacher_checkpoint_key": teacher_key,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": checkpoint_hash,
        "protocol_sha256": protocol_hash,
        "official_trial_sha256": sha256_file(trial_file),
        "conditions": [
            evaluation_condition_id(enrollment_duration, duration, snr)
            for enrollment_duration in enrollment_durations
            for duration, snr in probe_conditions
        ],
        "enrollment_durations_seconds": enrollment_durations,
        "probe_durations_seconds": sorted(
            {duration for duration, _ in probe_conditions}
        ),
        "requested_trials": requested_trials,
        "bootstrap_replicates": 0 if args.skip_bootstrap else bootstrap_replicates,
        "enrollment_recordings_per_trial": 1,
        "probe_recordings_per_trial": 1,
        "uses_enrollment_centroid": False,
        "threshold_method": "global score sweep for benchmark EER",
        "device": str(device),
    }
    atomic_json(run_metadata, run_dir / "run_metadata.json")
    archive = shutil.make_archive(str(run_dir), "zip", root_dir=run_dir)
    print(f"Results: {run_dir}")
    print(f"Archive: {archive}")


def main() -> None:
    """Validate local resources and run the requested evaluation mode."""
    parser = build_parser()
    args = parser.parse_args()
    if args.reference_protocol is None:
        protocol_trials = 200 if args.mode == "smoke" else 5_000
        protocol_mode = "smoke" if args.mode == "smoke" else "final"
        args.reference_protocol = (
            AAM_OUTPUT_ROOT
            / f"{protocol_mode}_seed{args.seed}_n{protocol_trials}"
            / "fixed_trials.csv"
        )
    for name in (
        "repository",
        "checkpoint",
        "musan_dir",
        "pretrained_dir",
        "output_root",
        "cache_root",
    ):
        setattr(args, name, resolve_path(getattr(args, name)))
    args.reference_protocol = resolve_path(args.reference_protocol)
    if args.embedding_batch_size < 1 or args.parquet_batch_size < 1:
        parser.error("batch sizes must be positive")
    if any(duration <= 0 for duration in args.enrollment_seconds):
        parser.error("--enrollment-seconds values must be positive")
    if any(duration <= 0 for duration in args.probe_seconds):
        parser.error("--probe-seconds values must be positive")
    if args.bootstrap_replicates is not None and args.bootstrap_replicates < 1:
        parser.error("--bootstrap-replicates must be positive")

    seed_everything(args.seed)
    report = check_inputs(args)
    print(json.dumps(json_safe(report), indent=2, sort_keys=True))
    if args.mode == "check":
        if not report["ready"]:
            sys.exit(2)
        print("Input check passed. Run --mode smoke next.")
        return
    run_evaluation(args, report)


if __name__ == "__main__":
    main()