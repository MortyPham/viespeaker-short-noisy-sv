"""Train the full-data VieSpeaker EMA teacher-student Gaussian-KL system.

The first invocation prepares a resumable local waveform/VAD cache from the
existing VieSpeaker Parquet shards. Subsequent invocations reuse that cache.
Only speakers assigned to the training split receive training labels. Held-out
validation speakers are reserved for five-enrollment, ten-calibration, and
ten-test speaker-verification evaluation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import io
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample_poly
from sklearn.metrics import roc_auc_score, roc_curve
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler

from temporal_pooling_variants import GlobalLocalPoolingECAPA, frame_shift_from_fbank


DEFAULT_PROJECT_ROOT = Path(
    "/media/ml4u/Samsung_T51/ptnk_team/mortyhuaphuoc"
)
DEFAULT_MODEL_CACHE = Path(
    "/home/ml4u/.cache/viespeaker_models/speechbrain_ecapa_voxceleb"
)
SYSTEM_NAME = "D_FULL"
CACHE_COLUMNS = (
    "speaker_id",
    "split",
    "source_id",
    "relative_path",
    "duration_seconds",
    "voiced_seconds",
    "vad_intervals",
)


@dataclass(frozen=True)
class Recording:
    """One cached source recording with its precomputed speech intervals."""

    speaker_id: str
    split: str
    source_id: str
    path: str
    duration_seconds: float
    voiced_seconds: float
    vad_intervals: tuple[tuple[int, int], ...]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse all server, dataset, optimization, and evaluation settings."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--parquet-dir", type=Path, default=None)
    parser.add_argument("--speaker-splits", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--recording-manifest", type=Path, default=None)
    parser.add_argument("--musan-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-cache", type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument("--model-source", default="speechbrain/spkrec-ecapa-voxceleb")
    parser.add_argument("--pooling-variant", choices=("original", "global_local", "global_local_duration"), default="original")
    parser.add_argument("--local-window-seconds", type=float, default=0.5)
    parser.add_argument("--local-overlap", type=float, default=0.5)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--speakers-per-batch", type=int, default=16)
    parser.add_argument("--pairs-per-speaker", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-recordings-per-speaker", type=int, default=12)
    parser.add_argument("--validation-recordings-per-speaker", type=int, default=25)
    parser.add_argument("--allow-incomplete-speakers", action="store_true")
    parser.add_argument("--long-min-seconds", type=float, default=2.0)
    parser.add_argument("--long-max-seconds", type=float, default=4.0)
    parser.add_argument("--short-min-seconds", type=float, default=1.0)
    parser.add_argument("--short-max-seconds", type=float, default=1.5)
    parser.add_argument("--validation-enroll-seconds", type=float, default=3.0)
    parser.add_argument("--validation-probe-seconds", type=float, default=1.25)
    parser.add_argument("--validation-enrollments", type=int, default=5)
    parser.add_argument("--validation-calibration", type=int, default=10)
    parser.add_argument("--validation-tests", type=int, default=10)
    parser.add_argument("--target-far", type=float, default=0.05)
    parser.add_argument("--validation-noise-db", type=float, default=10.0)
    parser.add_argument(
        "--validation-enrollment-model", choices=("teacher", "student"), default="teacher"
    )
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--ema-alpha", type=float, default=0.999)
    parser.add_argument("--lambda-vector", type=float, default=1.0)
    parser.add_argument("--lambda-kl", type=float, default=0.01)
    parser.add_argument("--kl-epsilon", type=float, default=1e-5)
    parser.add_argument("--aam-margin", type=float, default=0.2)
    parser.add_argument("--aam-scale", type=float, default=30.0)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--noise-min-db", type=float, default=0.0)
    parser.add_argument("--noise-max-db", type=float, default=20.0)
    parser.add_argument("--clean-probability", type=float, default=0.20)
    parser.add_argument("--eval-every", type=int, default=1_000)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--periodic-save-every", type=int, default=5_000)
    parser.add_argument("--wandb-project", default="viespeaker-mentor-d-full")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--run-name", default="d-full-ema-gaussian-kl-full-data")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        return args

    root = args.project_root.expanduser().resolve()
    args.project_root = root
    args.parquet_dir = resolve_path(
        args.parquet_dir,
        root / "datasets/VieSpeaker/repository/data",
    )
    args.speaker_splits = resolve_path(
        args.speaker_splits,
        root / "experiments/shared/speaker_splits.csv",
    )
    args.cache_dir = resolve_path(
        args.cache_dir,
        root / "experiments/shared/d_full_audio_cache",
    )
    args.musan_dir = resolve_path(args.musan_dir, root / "datasets/musan")
    args.output_dir = resolve_path(
        args.output_dir,
        root / "experiments/mentor_stage/m3_legacy_ema_gaussian_kl",
    )
    args.model_cache = args.model_cache.expanduser().resolve()
    if args.recording_manifest is not None:
        args.recording_manifest = args.recording_manifest.expanduser().resolve()

    if args.steps <= 0 or args.speakers_per_batch <= 1 or args.pairs_per_speaker <= 0:
        parser.error("Steps, speaker count, and pair count must define a valid batch.")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative.")
    if not 0.0 < args.target_far < 1.0:
        parser.error("--target-far must be between zero and one.")
    if not 0.0 <= args.clean_probability <= 1.0:
        parser.error("--clean-probability must be between zero and one.")
    if args.short_min_seconds <= 0 or args.short_min_seconds > args.short_max_seconds:
        parser.error("The short-duration interval is invalid.")
    if args.long_min_seconds < args.short_max_seconds:
        parser.error("The minimum long duration must not be shorter than the maximum probe.")
    if args.long_min_seconds > args.long_max_seconds:
        parser.error("The long-duration interval is invalid.")
    minimum_validation = (
        args.validation_enrollments + args.validation_calibration + args.validation_tests
    )
    if args.validation_recordings_per_speaker < minimum_validation:
        parser.error(
            f"Validation speakers require at least {minimum_validation} separate recordings."
        )
    if args.train_recordings_per_speaker < 2:
        parser.error("Training speakers require at least two separate recordings.")
    return args


def resolve_path(value: Path | None, default: Path) -> Path:
    """Expand an optional path while retaining the documented server defaults."""

    return (default if value is None else value).expanduser().resolve()


def stable_int(value: object, modulo: int = 2**32 - 1) -> int:
    """Create a reproducible integer without Python's process-randomized hash."""

    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulo


def file_sha256(path: Path) -> str:
    """Hash a split or recording manifest for safe checkpoint resumption."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch randomness."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_split_name(value: object) -> str:
    """Map common speaker-manifest labels to train, val, and test."""

    token = str(value).strip().lower()
    if token in {"train", "training", "trn"}:
        return "train"
    if token in {"val", "valid", "validation", "dev", "development"}:
        return "val"
    if token in {"test", "testing", "heldout", "held_out", "final_test"}:
        return "test"
    raise ValueError(f"Unrecognized speaker partition: {value!r}")


def load_speaker_splits(path: Path) -> dict[str, list[str]]:
    """Read the existing speaker-disjoint manifest without creating a new split."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Speaker split manifest not found: {path}\n"
            "Provide its actual location with --speaker-splits."
        )

    partitions: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        names = reader.fieldnames or []
        speaker_column = next(
            (name for name in names if name.lower() in {"speaker_id", "speaker", "spk", "spkid"}),
            None,
        )
        split_column = next(
            (name for name in names if name.lower() in {"split", "partition", "subset", "set"}),
            None,
        )
        if speaker_column is None or split_column is None:
            raise ValueError(
                f"Expected speaker and partition columns in {path}; found {names}."
            )
        memberships: dict[str, str] = {}
        for row in reader:
            speaker = str(row[speaker_column]).strip()
            partition = normalize_split_name(row[split_column])
            if not speaker:
                raise ValueError("The split manifest contains an empty speaker identity.")
            previous = memberships.get(speaker)
            if previous is not None and previous != partition:
                raise ValueError(
                    f"Speaker leakage: {speaker!r} appears in both {previous} and {partition}."
                )
            if previous is None:
                memberships[speaker] = partition
                partitions[partition].append(speaker)

    for speakers in partitions.values():
        speakers.sort()
    if not partitions["train"] or not partitions["val"]:
        raise ValueError("The split manifest must contain nonempty train and validation partitions.")
    return partitions


def decode_audio(audio_bytes: bytes, expected_sr: int) -> np.ndarray:
    """Decode stored audio bytes, downmix them, and resample to 16 kHz."""

    waveform, source_sr = sf.read(
        io.BytesIO(audio_bytes), dtype="float32", always_2d=False
    )
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if source_sr != expected_sr:
        divisor = math.gcd(int(source_sr), expected_sr)
        waveform = resample_poly(
            waveform, expected_sr // divisor, int(source_sr) // divisor
        ).astype(np.float32)
    if not len(waveform) or not np.isfinite(waveform).all():
        raise ValueError("Audio is empty or contains non-finite values.")
    return waveform


def load_audio(path: str | Path, expected_sr: int) -> np.ndarray:
    """Read one cached waveform or MUSAN file and normalize its sample rate."""

    waveform, source_sr = sf.read(str(path), dtype="float32", always_2d=False)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if int(source_sr) != expected_sr:
        divisor = math.gcd(int(source_sr), expected_sr)
        waveform = resample_poly(
            waveform, expected_sr // divisor, int(source_sr) // divisor
        ).astype(np.float32)
    if not len(waveform) or not np.isfinite(waveform).all():
        raise ValueError(f"Invalid waveform: {path}")
    return waveform


def peak_guard(waveform: np.ndarray, peak: float = 0.99) -> np.ndarray:
    """Prevent clipping while preserving the requested waveform shape."""

    waveform = np.asarray(waveform, dtype=np.float32)
    amplitude = float(np.max(np.abs(waveform)) + 1e-8)
    if amplitude > peak:
        waveform = waveform * (peak / amplitude)
    return waveform.astype(np.float32)


def crop_by_voiced_duration(
    waveform: np.ndarray,
    intervals: tuple[tuple[int, int], ...] | list[list[int]],
    voiced_seconds: float,
    rng: np.random.Generator,
    sample_rate: int,
) -> np.ndarray:
    """Return a contiguous crop containing the requested amount of voiced speech."""

    required = int(round(voiced_seconds * sample_rate))
    lengths = np.asarray([end - start for start, end in intervals], dtype=np.int64)
    available = int(lengths.sum())
    if not intervals or available < required:
        raise ValueError(f"Need {required} voiced samples but found {available}.")

    cumulative = np.concatenate(([0], np.cumsum(lengths)))
    voiced_start = int(rng.integers(0, available - required + 1))
    voiced_end = voiced_start + required
    first = int(np.searchsorted(cumulative[1:], voiced_start, side="right"))
    last = int(np.searchsorted(cumulative[1:], voiced_end - 1, side="right"))
    raw_start = int(intervals[first][0] + voiced_start - cumulative[first])
    raw_end = int(intervals[last][0] + voiced_end - cumulative[last])
    result = np.asarray(waveform[max(0, raw_start) : min(len(waveform), raw_end)], dtype=np.float32)
    if not len(result):
        raise ValueError("Speech-aware cropping produced an empty waveform.")
    return result


def parse_intervals(value: object) -> tuple[tuple[int, int], ...]:
    """Read stored JSON VAD intervals from the reusable recording manifest."""

    raw = json.loads(value) if isinstance(value, str) else value
    return tuple((int(start), int(end)) for start, end in raw)


def load_recording_manifest(path: Path, cache_dir: Path) -> dict[str, list[Recording]]:
    """Load a prepared cache manifest and resolve relative waveform paths."""

    records: dict[str, list[Recording]] = defaultdict(list)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_id = row.get("source_id") or row.get("key")
            speaker_id = row.get("speaker_id") or row.get("speaker")
            waveform_path = row.get("absolute_path") or row.get("path") or row.get("relative_path")
            split = row.get("split") or row.get("partition") or ""
            if not source_id or not speaker_id or not waveform_path:
                raise ValueError(f"Recording manifest row lacks a source, speaker, or audio path: {row}")
            audio_path = Path(waveform_path)
            if not audio_path.is_absolute():
                candidates = (cache_dir / audio_path, path.parent / audio_path)
                audio_path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
            if not audio_path.is_file():
                raise FileNotFoundError(f"Cached recording is missing: {audio_path}")
            if not row.get("vad_intervals"):
                raise ValueError("A reusable recording manifest must contain precomputed vad_intervals.")
            record = Recording(
                speaker_id=str(speaker_id),
                split=normalize_split_name(split) if split else "",
                source_id=str(source_id),
                path=str(audio_path),
                duration_seconds=float(row["duration_seconds"]),
                voiced_seconds=float(row["voiced_seconds"]),
                vad_intervals=parse_intervals(row["vad_intervals"]),
            )
            records[record.speaker_id].append(record)
    return dict(records)


def parquet_columns(parquet_file: pq.ParquetFile) -> tuple[str, str]:
    """Find the speaker and audio columns in a VieSpeaker Parquet shard."""

    names = list(parquet_file.schema_arrow.names)
    speaker_column = next(
        (name for name in names if name.lower() in {"speaker_id", "speaker", "spk", "spkid"}),
        None,
    )
    audio_column = next((name for name in names if name.lower() in {"audio", "waveform"}), None)
    if speaker_column is None or audio_column is None:
        raise ValueError(f"Cannot identify speaker/audio columns in Parquet schema: {names}")
    return speaker_column, audio_column


def unpack_audio_value(value: object, speaker: str, shard: Path, row_number: int) -> tuple[str, bytes]:
    """Extract a stable source identity and audio bytes from a Parquet value."""

    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"{speaker}/{shard.stem}_{row_number:08d}", bytes(value)
    if not isinstance(value, dict):
        raise ValueError(f"Unsupported audio value type: {type(value).__name__}")
    audio_bytes = value.get("bytes")
    if audio_bytes is None:
        external_path = value.get("path")
        if external_path and Path(str(external_path)).is_file():
            audio_bytes = Path(str(external_path)).read_bytes()
        else:
            raise ValueError("The Parquet audio entry contains no embedded bytes.")
    original_path = str(value.get("path") or f"{shard.stem}_{row_number:08d}").replace("\\", "/")
    source_id = original_path if "/" in original_path else f"{speaker}/{original_path}"
    return source_id, bytes(audio_bytes)


def cache_requirements(args: argparse.Namespace, partition: str) -> tuple[int, int, float]:
    """Return recording quota, required long recordings, and long cutoff."""

    if partition == "train":
        return args.train_recordings_per_speaker, 1, args.long_max_seconds
    return (
        args.validation_recordings_per_speaker,
        args.validation_enrollments,
        args.validation_enroll_seconds,
    )


def cache_is_complete(
    records: dict[str, list[Recording]],
    partitions: dict[str, list[str]],
    args: argparse.Namespace,
) -> bool:
    """Determine whether every requested training and validation identity is cached."""

    for partition in ("train", "val"):
        quota, required_long, cutoff = cache_requirements(args, partition)
        for speaker in partitions[partition]:
            rows = records.get(speaker, [])
            long_count = sum(row.voiced_seconds >= cutoff for row in rows)
            if len(rows) < quota or long_count < required_long:
                return False
    return True


def prepare_recording_cache(
    args: argparse.Namespace,
    partitions: dict[str, list[str]],
) -> tuple[dict[str, list[Recording]], Path]:
    """Build or resume the local waveform cache without using held-out test identities."""

    if args.recording_manifest is not None:
        manifest = args.recording_manifest
        records = load_recording_manifest(manifest, args.cache_dir)
        if not records:
            raise ValueError(f"The supplied recording manifest is empty: {manifest}")
        return records, manifest

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.cache_dir / "recordings.csv"
    records = load_recording_manifest(manifest, args.cache_dir)
    split_by_speaker = {
        speaker: partition
        for partition in ("train", "val")
        for speaker in partitions[partition]
    }
    for speaker, rows in records.items():
        expected = split_by_speaker.get(speaker)
        if expected is None:
            raise ValueError(f"Cache contains unexpected or held-out test speaker: {speaker}")
        if any(row.split and row.split != expected for row in rows):
            raise ValueError(f"Cached speaker {speaker} has a conflicting partition.")

    if cache_is_complete(records, partitions, args):
        print(f"Reusing complete waveform/VAD cache: {manifest}", flush=True)
        return records, manifest

    shards = sorted(args.parquet_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No Parquet shards found in {args.parquet_dir}")

    try:
        from silero_vad import get_speech_timestamps, load_silero_vad
    except ImportError as exc:
        raise RuntimeError(
            "silero-vad is required to build the speech-aware audio cache. "
            "Install it in the same Python environment used for training."
        ) from exc

    vad_model = load_silero_vad()
    seen_sources = {row.source_id for rows in records.values() for row in rows}
    failures: Counter[str] = Counter()
    write_header = not manifest.exists() or manifest.stat().st_size == 0
    buffered = 0
    print(
        "Preparing local audio cache. This one-time stage can take longer than training "
        "because it scans Parquet shards and runs voice activity detection.",
        flush=True,
    )

    with manifest.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CACHE_COLUMNS)
        if write_header:
            writer.writeheader()

        for shard_index, shard in enumerate(shards, start=1):
            parquet_file = pq.ParquetFile(shard)
            speaker_column, audio_column = parquet_columns(parquet_file)
            row_number = 0
            for batch in parquet_file.iter_batches(
                batch_size=32, columns=[speaker_column, audio_column], use_threads=True
            ):
                speaker_values = batch.column(0)
                audio_values = batch.column(1)
                for item in range(batch.num_rows):
                    row_number += 1
                    speaker = str(speaker_values[item].as_py())
                    partition = split_by_speaker.get(speaker)
                    if partition is None:
                        continue

                    quota, required_long, long_cutoff = cache_requirements(args, partition)
                    current = records.get(speaker, [])
                    current_long = sum(row.voiced_seconds >= long_cutoff for row in current)
                    if len(current) >= quota and current_long >= required_long:
                        continue

                    try:
                        source_id, audio_bytes = unpack_audio_value(
                            audio_values[item].as_py(), speaker, shard, row_number
                        )
                        if source_id in seen_sources:
                            continue
                        waveform = decode_audio(audio_bytes, args.sample_rate)
                        timestamps = get_speech_timestamps(
                            torch.from_numpy(waveform),
                            vad_model,
                            sampling_rate=args.sample_rate,
                            return_seconds=False,
                        )
                        intervals = tuple(
                            (int(stamp["start"]), int(stamp["end"]))
                            for stamp in timestamps
                            if int(stamp["end"]) > int(stamp["start"])
                        )
                        voiced_seconds = sum(end - start for start, end in intervals) / args.sample_rate
                        if voiced_seconds < args.short_max_seconds:
                            failures["insufficient_speech"] += 1
                            continue

                        remaining_slots = quota - len(current)
                        missing_long = max(0, required_long - current_long)
                        qualifies_as_long = voiced_seconds >= long_cutoff
                        if remaining_slots <= missing_long and not qualifies_as_long:
                            failures["reserved_long_slot"] += 1
                            continue

                        speaker_folder = hashlib.sha256(speaker.encode("utf-8")).hexdigest()[:16]
                        source_digest = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:24]
                        relative_path = Path("audio") / speaker_folder / f"{source_digest}.wav"
                        destination = args.cache_dir / relative_path
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        sf.write(str(destination), waveform, args.sample_rate, subtype="PCM_16")

                        row = {
                            "speaker_id": speaker,
                            "split": partition,
                            "source_id": source_id,
                            "relative_path": relative_path.as_posix(),
                            "duration_seconds": f"{len(waveform) / args.sample_rate:.8f}",
                            "voiced_seconds": f"{voiced_seconds:.8f}",
                            "vad_intervals": json.dumps(intervals, separators=(",", ":")),
                        }
                        writer.writerow(row)
                        record = Recording(
                            speaker_id=speaker,
                            split=partition,
                            source_id=source_id,
                            path=str(destination),
                            duration_seconds=len(waveform) / args.sample_rate,
                            voiced_seconds=voiced_seconds,
                            vad_intervals=intervals,
                        )
                        records.setdefault(speaker, []).append(record)
                        seen_sources.add(source_id)
                        buffered += 1
                        if buffered >= 100:
                            handle.flush()
                            buffered = 0
                    except Exception as exc:
                        failures[type(exc).__name__] += 1

            handle.flush()
            complete_train = sum(
                len(records.get(speaker, [])) >= args.train_recordings_per_speaker
                for speaker in partitions["train"]
            )
            complete_val = sum(
                len(records.get(speaker, [])) >= args.validation_recordings_per_speaker
                for speaker in partitions["val"]
            )
            print(
                f"cache shard={shard_index}/{len(shards)} "
                f"train_ready={complete_train}/{len(partitions['train'])} "
                f"val_ready={complete_val}/{len(partitions['val'])} "
                f"recordings={sum(len(rows) for rows in records.values()):,} "
                f"skipped={dict(failures)}",
                flush=True,
            )
            if cache_is_complete(records, partitions, args):
                break

    del vad_model
    gc.collect()
    return records, manifest


def select_eligible_records(
    records: dict[str, list[Recording]],
    partitions: dict[str, list[str]],
    args: argparse.Namespace,
) -> tuple[dict[str, list[Recording]], list[str], list[str]]:
    """Audit split integrity, long examples, and recording-disjoint validation capacity."""

    selected: dict[str, list[Recording]] = {}
    eligible_train: list[str] = []
    eligible_val: list[str] = []
    missing_train: list[str] = []
    missing_val: list[str] = []
    reserved_test = set(partitions["test"])

    for partition, destination, missing in (
        ("train", eligible_train, missing_train),
        ("val", eligible_val, missing_val),
    ):
        for speaker in partitions[partition]:
            if speaker in reserved_test:
                raise ValueError(f"Speaker leakage into the final-test split: {speaker}")
            source_map = {
                row.source_id: row
                for row in records.get(speaker, [])
                if row.voiced_seconds >= args.short_max_seconds
            }
            rows = sorted(source_map.values(), key=lambda row: row.source_id)
            if any(row.split and row.split != partition for row in rows):
                raise ValueError(f"Speaker {speaker} appears under an incompatible recording partition.")
            if partition == "train":
                valid = len(rows) >= 2 and any(
                    row.voiced_seconds >= args.long_max_seconds for row in rows
                )
            else:
                long_count = sum(
                    row.voiced_seconds >= args.validation_enroll_seconds for row in rows
                )
                required = (
                    args.validation_enrollments
                    + args.validation_calibration
                    + args.validation_tests
                )
                valid = len(rows) >= required and long_count >= args.validation_enrollments
            if valid:
                selected[speaker] = rows
                destination.append(speaker)
            else:
                missing.append(speaker)

    if missing_train or missing_val:
        message = (
            f"Unusable split identities: train={len(missing_train)} "
            f"val={len(missing_val)}; examples train={missing_train[:5]} "
            f"val={missing_val[:5]}"
        )
        if not args.allow_incomplete_speakers:
            raise RuntimeError(
                message
                + "\nRefusing to silently train on less than the full split. "
                "Fix the cache or explicitly pass --allow-incomplete-speakers."
            )
        print(f"WARNING: {message}", flush=True)

    if len(eligible_train) < args.speakers_per_batch:
        raise RuntimeError("Too few eligible training speakers for the requested batch.")
    if len(eligible_val) < 2:
        raise RuntimeError("At least two eligible validation speakers are required.")
    return selected, eligible_train, eligible_val


class MusanAugmenter:
    """Apply speaker-safe MUSAN noise with disjoint training and validation files."""

    def __init__(self, root: Path, args: argparse.Namespace):
        self.sample_rate = args.sample_rate
        self.noise_min_db = args.noise_min_db
        self.noise_max_db = args.noise_max_db
        self.clean_probability = args.clean_probability
        self.training: dict[str, list[str]] = defaultdict(list)
        self.validation: dict[str, list[str]] = defaultdict(list)

        if not root.exists():
            raise FileNotFoundError(f"MUSAN directory not found: {root}")
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".wav", ".flac", ".ogg"}:
                continue
            components = {part.lower() for part in path.relative_to(root).parts}
            category = next(
                (name for name in ("noise", "music", "speech") if name in components),
                "noise",
            )
            bucket = stable_int(path.relative_to(root).as_posix(), 100)
            if bucket < 80:
                self.training[category].append(str(path))
            elif bucket < 90:
                self.validation[category].append(str(path))

        self.train_categories = sorted(
            category for category, files in self.training.items() if files
        )
        self.val_categories = sorted(
            category for category, files in self.validation.items() if files
        )
        if not self.train_categories or not self.val_categories:
            raise RuntimeError(
                "MUSAN must contain separate nonempty training and validation noise partitions."
            )

    @staticmethod
    def fit_length(waveform: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
        """Tile and randomly crop noise to match the speech waveform length."""

        if len(waveform) < length:
            waveform = np.tile(waveform, math.ceil(length / len(waveform)))
        start = int(rng.integers(0, len(waveform) - length + 1))
        return np.asarray(waveform[start : start + length], dtype=np.float32)

    def apply(
        self,
        clean: np.ndarray,
        rng: np.random.Generator,
        partition: str = "train",
        snr_db: float | None = None,
        force_noise: bool = False,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Create a clean or noisy recording at the requested signal-to-noise ratio."""

        if not force_noise and float(rng.random()) < self.clean_probability:
            return clean, {"augmentation": "clean", "snr_db": None, "files": []}
        pools = self.training if partition == "train" else self.validation
        categories = self.train_categories if partition == "train" else self.val_categories
        category = categories[int(rng.integers(0, len(categories)))]
        paths = pools[category]
        count = 3 if category == "speech" and len(paths) >= 3 else 1
        noises: list[np.ndarray] = []
        chosen: list[str] = []
        for _ in range(count):
            path = paths[int(rng.integers(0, len(paths)))]
            waveform = load_audio(path, self.sample_rate)
            noises.append(self.fit_length(waveform, len(clean), rng))
            chosen.append(path)
        noise = np.mean(noises, axis=0).astype(np.float32)
        target_snr = float(
            rng.uniform(self.noise_min_db, self.noise_max_db)
            if snr_db is None
            else snr_db
        )
        speech_power = float(np.mean(clean**2) + 1e-8)
        noise_power = float(np.mean(noise**2) + 1e-8)
        scaling = math.sqrt(speech_power / (noise_power * 10 ** (target_snr / 10.0)))
        mixed = peak_guard(clean + scaling * noise)
        return mixed, {
            "augmentation": f"musan_{category}",
            "snr_db": target_snr,
            "files": chosen,
        }


class SpeakerPairDataset(Dataset):
    """Sample different-source clean long teacher and noisy short student examples."""

    def __init__(
        self,
        records: dict[str, list[Recording]],
        speaker_to_label: dict[str, int],
        augmenter: MusanAugmenter,
        args: argparse.Namespace,
    ):
        self.records = records
        self.speaker_to_label = speaker_to_label
        self.augmenter = augmenter
        self.sample_rate = args.sample_rate
        self.long_min = args.long_min_seconds
        self.long_max = args.long_max_seconds
        self.short_min = args.short_min_seconds
        self.short_max = args.short_max_seconds
        self.long_candidates = {
            speaker: tuple(row for row in records[speaker] if row.voiced_seconds >= self.long_max)
            for speaker in speaker_to_label
        }

    def __len__(self) -> int:
        return 10**9

    def __getitem__(self, specification: tuple[str, int]) -> dict[str, Any]:
        speaker, item_seed = specification
        rng = np.random.default_rng(int(item_seed))
        long_options = self.long_candidates[speaker]
        teacher_record = long_options[int(rng.integers(0, len(long_options)))]
        student_options = tuple(
            row for row in self.records[speaker] if row.source_id != teacher_record.source_id
        )
        student_record = student_options[int(rng.integers(0, len(student_options)))]
        teacher_seconds = float(rng.uniform(self.long_min, self.long_max))
        student_seconds = float(rng.uniform(self.short_min, self.short_max))
        teacher_wave = crop_by_voiced_duration(
            load_audio(teacher_record.path, self.sample_rate),
            teacher_record.vad_intervals,
            teacher_seconds,
            rng,
            self.sample_rate,
        )
        student_wave = crop_by_voiced_duration(
            load_audio(student_record.path, self.sample_rate),
            student_record.vad_intervals,
            student_seconds,
            rng,
            self.sample_rate,
        )
        student_wave, augmentation = self.augmenter.apply(student_wave, rng, partition="train")
        return {
            "teacher_audio": torch.from_numpy(peak_guard(teacher_wave)),
            "student_audio": torch.from_numpy(peak_guard(student_wave)),
            "label": self.speaker_to_label[speaker],
            "speaker_id": speaker,
            "teacher_source_id": teacher_record.source_id,
            "student_source_id": student_record.source_id,
            "teacher_seconds": teacher_seconds,
            "student_seconds": student_seconds,
            "augmentation": augmentation["augmentation"],
        }


class SpeakerBalancedBatchSampler(Sampler[list[tuple[str, int]]]):
    """Reproduce the notebook's deterministic 16-speaker, two-pair batch sampler."""

    def __init__(
        self,
        speakers: list[str],
        num_batches: int,
        speakers_per_batch: int,
        pairs_per_speaker: int,
        seed: int,
        start_step: int = 0,
    ):
        self.speakers = list(speakers)
        self.num_batches = int(num_batches)
        self.speakers_per_batch = int(speakers_per_batch)
        self.pairs_per_speaker = int(pairs_per_speaker)
        self.seed = int(seed)
        self.start_step = int(start_step)
        if len(self.speakers) < self.speakers_per_batch:
            raise ValueError("Not enough unique speakers for one balanced batch.")

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        for offset in range(self.num_batches):
            global_step = self.start_step + offset
            rng = np.random.default_rng(self.seed + global_step)
            speakers = rng.choice(self.speakers, self.speakers_per_batch, replace=False)
            yield [
                (
                    str(speaker),
                    stable_int(f"{self.seed}:{global_step}:{speaker}:{pair_index}"),
                )
                for speaker in speakers
                for pair_index in range(self.pairs_per_speaker)
            ]


def collate_pairs(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad teacher/student waveforms independently and create relative lengths."""

    teacher_lengths = torch.tensor([len(item["teacher_audio"]) for item in items], dtype=torch.float32)
    student_lengths = torch.tensor([len(item["student_audio"]) for item in items], dtype=torch.float32)
    teacher_audio = pad_sequence([item["teacher_audio"] for item in items], batch_first=True)
    student_audio = pad_sequence([item["student_audio"] for item in items], batch_first=True)
    return {
        "teacher_audio": teacher_audio,
        "student_audio": student_audio,
        "teacher_lengths": teacher_lengths / teacher_audio.shape[1],
        "student_lengths": student_lengths / student_audio.shape[1],
        "labels": torch.tensor([item["label"] for item in items], dtype=torch.long),
        "metadata": items,
    }


class ECAPAEncoder(nn.Module):
    """Expose only SpeechBrain feature extraction, normalization, and ECAPA embeddings."""

    def __init__(self, compute_features: nn.Module, mean_var_norm: nn.Module, embedding_model: nn.Module, *, pooling_variant: str = "original", local_window_seconds: float = 0.5, local_overlap: float = 0.5):
        super().__init__()
        self.compute_features = compute_features
        self.mean_var_norm = mean_var_norm
        self.embedding_model = embedding_model
        self.temporal_pooling = None
        if pooling_variant != "original":
            self.temporal_pooling = GlobalLocalPoolingECAPA(
                embedding_model,
                variant=pooling_variant,
                frame_shift_seconds=frame_shift_from_fbank(compute_features),
                local_window_seconds=local_window_seconds,
                local_overlap=local_overlap,
            )

    def forward(self, waveform: torch.Tensor, relative_lengths: torch.Tensor) -> torch.Tensor:
        features = self.compute_features(waveform.float())
        normalized = self.mean_var_norm(features, relative_lengths)
        embeddings = (
            self.embedding_model(normalized, relative_lengths)
            if self.temporal_pooling is None
            else self.temporal_pooling(normalized, relative_lengths)
        )
        if embeddings.ndim == 3 and embeddings.shape[1] == 1:
            embeddings = embeddings[:, 0, :]
        return embeddings


def load_pretrained_encoder(args: argparse.Namespace) -> ECAPAEncoder:
    """Load the existing internal-disk SpeechBrain model without its 7,205-class head."""

    from speechbrain.inference.speaker import EncoderClassifier

    source = str(args.model_cache) if (args.model_cache / "hyperparams.yaml").is_file() else args.model_source
    pretrained = EncoderClassifier.from_hparams(
        source=source,
        savedir=str(args.model_cache),
        run_opts={"device": "cpu"},
        freeze_params=False,
    )
    encoder = ECAPAEncoder(
        copy.deepcopy(pretrained.mods.compute_features),
        copy.deepcopy(pretrained.mods.mean_var_norm),
        copy.deepcopy(pretrained.mods.embedding_model),
        pooling_variant=getattr(args, "pooling_variant", "original"),
        local_window_seconds=getattr(args, "local_window_seconds", 0.5),
        local_overlap=getattr(args, "local_overlap", 0.5),
    )
    del pretrained
    gc.collect()
    return encoder


class AAMSoftmaxHead(nn.Module):
    """Classify only the actual VieSpeaker training identities with AAM-Softmax."""

    def __init__(self, embedding_dim: int, classes: int, margin: float, scale: float):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(classes, embedding_dim))
        nn.init.xavier_normal_(self.weight)
        self.scale = float(scale)
        self.cos_margin = math.cos(margin)
        self.sin_margin = math.sin(margin)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cosine = F.linear(
            F.normalize(embeddings.float(), dim=-1),
            F.normalize(self.weight.float(), dim=-1),
        ).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        sine = torch.sqrt((1.0 - cosine.square()).clamp_min(1e-7))
        adjusted = cosine * self.cos_margin - sine * self.sin_margin
        one_hot = F.one_hot(labels, num_classes=cosine.shape[1]).to(cosine.dtype)
        logits = (one_hot * adjusted + (1.0 - one_hot) * cosine) * self.scale
        return F.cross_entropy(logits, labels), logits


def vector_alignment_loss(teacher: torch.Tensor, student: torch.Tensor) -> torch.Tensor:
    """Match paired teacher/student embeddings using the notebook's cosine loss."""

    return 1.0 - (teacher.detach().float() * student.float()).sum(dim=-1).mean()


def diagonal_gaussian_kl(source: torch.Tensor, target: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Compute KL between diagonal Gaussian distributions estimated across a batch."""

    source = source.float()
    target = target.float()
    source_mean = source.mean(dim=0)
    target_mean = target.mean(dim=0)
    source_var = source.var(dim=0, unbiased=False) + epsilon
    target_var = target.var(dim=0, unbiased=False) + epsilon
    return 0.5 * (
        torch.log(target_var / source_var)
        + (source_var + (source_mean - target_mean).square()) / target_var
        - 1.0
    ).mean()


def symmetric_diagonal_gaussian_kl(
    teacher: torch.Tensor, student: torch.Tensor, epsilon: float
) -> torch.Tensor:
    """Reproduce the original symmetric batch-Gaussian KL distribution loss."""

    detached_teacher = teacher.detach()
    return 0.5 * (
        diagonal_gaussian_kl(detached_teacher, student, epsilon)
        + diagonal_gaussian_kl(student, detached_teacher, epsilon)
    )


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, alpha: float) -> None:
    """Update both EMA parameters and BatchNorm buffers after a real optimizer step."""

    for teacher_parameter, student_parameter in zip(teacher.parameters(), student.parameters(), strict=True):
        teacher_parameter.mul_(alpha).add_(student_parameter, alpha=1.0 - alpha)
    for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers(), strict=True):
        if teacher_buffer.dtype.is_floating_point:
            teacher_buffer.mul_(alpha).add_(student_buffer, alpha=1.0 - alpha)
        else:
            teacher_buffer.copy_(student_buffer)


def create_validation_assignments(
    records: dict[str, list[Recording]],
    speakers: list[str],
    args: argparse.Namespace,
) -> dict[str, dict[str, list[Recording]]]:
    """Reserve distinct enrollment, calibration, and test source recordings."""

    assignments: dict[str, dict[str, list[Recording]]] = {}
    for speaker in speakers:
        rows = records[speaker]
        enrollable = [
            row for row in rows if row.voiced_seconds >= args.validation_enroll_seconds
        ]
        enrollment = enrollable[: args.validation_enrollments]
        excluded = {row.source_id for row in enrollment}
        remaining = [row for row in rows if row.source_id not in excluded]
        calibration = remaining[: args.validation_calibration]
        held_out = remaining[
            args.validation_calibration : args.validation_calibration + args.validation_tests
        ]
        all_sources = enrollment + calibration + held_out
        if len({row.source_id for row in all_sources}) != len(all_sources):
            raise AssertionError(f"Recording leakage inside validation speaker {speaker}.")
        assignments[speaker] = {
            "enrollment": enrollment,
            "calibration": calibration,
            "test": held_out,
        }
    return assignments


@torch.no_grad()
def encode_validation_records(
    model: nn.Module,
    rows: list[Recording],
    role: str,
    condition: str,
    augmenter: MusanAugmenter,
    args: argparse.Namespace,
    device: torch.device,
    batch_size: int = 32,
) -> dict[str, np.ndarray]:
    """Encode deterministic validation crops and optional held-out MUSAN noise."""

    model.eval()
    result: dict[str, np.ndarray] = {}
    for start in range(0, len(rows), batch_size):
        current = rows[start : start + batch_size]
        waveforms: list[torch.Tensor] = []
        for row in current:
            crop_rng = np.random.default_rng(stable_int(f"val-crop:{role}:{row.source_id}"))
            seconds = (
                args.validation_enroll_seconds
                if role == "enrollment"
                else args.validation_probe_seconds
            )
            waveform = crop_by_voiced_duration(
                load_audio(row.path, args.sample_rate),
                row.vad_intervals,
                seconds,
                crop_rng,
                args.sample_rate,
            )
            if condition == "noise":
                noise_rng = np.random.default_rng(stable_int(f"val-noise:{row.source_id}"))
                waveform, _ = augmenter.apply(
                    waveform,
                    noise_rng,
                    partition="val",
                    snr_db=args.validation_noise_db,
                    force_noise=True,
                )
            waveforms.append(torch.from_numpy(peak_guard(waveform)))
        lengths = torch.tensor([len(waveform) for waveform in waveforms], dtype=torch.float32)
        padded = pad_sequence(waveforms, batch_first=True)
        relative = (lengths / padded.shape[1]).to(device)
        with autocast_context(device, args):
            embeddings = F.normalize(model(padded.to(device), relative).float(), dim=-1)
        for row, embedding in zip(current, embeddings.cpu().numpy(), strict=True):
            result[row.source_id] = embedding
    return result


def calculate_eer(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Return global equal-error rate and its nearest score threshold."""

    false_positive, true_positive, thresholds = roc_curve(labels, scores)
    false_negative = 1.0 - true_positive
    index = int(np.argmin(np.abs(false_positive - false_negative)))
    eer = float((false_positive[index] + false_negative[index]) / 2.0)
    return eer, float(thresholds[index])


def evaluate_verification(
    student: nn.Module,
    teacher: nn.Module,
    assignments: dict[str, dict[str, list[Recording]]],
    augmenter: MusanAugmenter,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float | int | str]:
    """Evaluate clean-calibrated centroids on clean and noisy held-out recordings."""

    speakers = sorted(assignments)
    enrollment_rows = [row for speaker in speakers for row in assignments[speaker]["enrollment"]]
    calibration_rows = [row for speaker in speakers for row in assignments[speaker]["calibration"]]
    test_rows = [row for speaker in speakers for row in assignments[speaker]["test"]]
    enrollment_model = teacher if args.validation_enrollment_model == "teacher" else student

    enrollment_embeddings = encode_validation_records(
        enrollment_model, enrollment_rows, "enrollment", "clean", augmenter, args, device
    )
    calibration_embeddings = encode_validation_records(
        student, calibration_rows, "calibration", "clean", augmenter, args, device
    )
    centroids = []
    for speaker in speakers:
        values = np.stack(
            [enrollment_embeddings[row.source_id] for row in assignments[speaker]["enrollment"]]
        )
        centroid = values.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-12
        centroids.append(centroid)
    centroid_matrix = np.stack(centroids)

    calibration_matrix = np.stack(
        [calibration_embeddings[row.source_id] for row in calibration_rows]
    )
    calibration_owner = np.asarray(
        [index for index, speaker in enumerate(speakers) for _ in assignments[speaker]["calibration"]]
    )
    calibration_scores = centroid_matrix @ calibration_matrix.T
    thresholds = np.asarray(
        [
            np.quantile(
                calibration_scores[index, calibration_owner != index],
                1.0 - args.target_far,
                method="higher",
            )
            for index in range(len(speakers))
        ],
        dtype=np.float64,
    )

    metrics: dict[str, float | int | str] = {
        "validation_speakers": len(speakers),
        "target_far": args.target_far,
        "enrollment_model": args.validation_enrollment_model,
    }
    owner = np.asarray(
        [index for index, speaker in enumerate(speakers) for _ in assignments[speaker]["test"]]
    )
    for condition in ("clean", "noise"):
        test_embeddings = encode_validation_records(
            student, test_rows, "test", condition, augmenter, args, device
        )
        matrix = np.stack([test_embeddings[row.source_id] for row in test_rows])
        score_matrix = centroid_matrix @ matrix.T
        labels_matrix = np.arange(len(speakers))[:, None] == owner[None, :]
        accepted = score_matrix >= thresholds[:, None]
        fars = []
        frrs = []
        for index in range(len(speakers)):
            genuine = labels_matrix[index]
            fars.append(float(np.mean(accepted[index, ~genuine])))
            frrs.append(float(np.mean(~accepted[index, genuine])))
        flattened_labels = labels_matrix.ravel().astype(np.int64)
        flattened_scores = score_matrix.ravel()
        eer, eer_threshold = calculate_eer(flattened_labels, flattened_scores)
        far = float(np.mean(fars))
        frr = float(np.mean(frrs))
        prefix = f"{condition}_"
        metrics.update(
            {
                prefix + "eer": eer,
                prefix + "eer_percent": 100.0 * eer,
                prefix + "eer_threshold": eer_threshold,
                prefix + "far": far,
                prefix + "frr": frr,
                prefix + "tar": 1.0 - frr,
                prefix + "balanced_error": 0.5 * (far + frr),
                prefix + "roc_auc": float(roc_auc_score(flattened_labels, flattened_scores)),
                prefix + "genuine_trials": int(flattened_labels.sum()),
                prefix + "impostor_trials": int((flattened_labels == 0).sum()),
            }
        )
    return metrics


def autocast_context(device: torch.device, args: argparse.Namespace):
    """Use CUDA mixed precision only when it is available and requested."""

    if device.type == "cuda" and not args.no_amp:
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def checkpoint_signature(
    args: argparse.Namespace,
    train_speakers: list[str],
    split_hash: str,
    recording_hash: str,
) -> dict[str, Any]:
    """Record settings that must remain identical when a checkpoint resumes."""

    return {
        "system": SYSTEM_NAME,
        "train_speaker_ids": train_speakers,
        "speaker_split_sha256": split_hash,
        "recording_manifest_sha256": recording_hash,
        "speakers_per_batch": args.speakers_per_batch,
        "pairs_per_speaker": args.pairs_per_speaker,
        "steps": args.steps,
        "encoder_lr": args.encoder_lr,
        "head_lr": args.head_lr,
        "warmup_steps": args.warmup_steps,
        "ema_alpha": args.ema_alpha,
        "lambda_vector": args.lambda_vector,
        "lambda_kl": args.lambda_kl,
        "aam_margin": args.aam_margin,
        "aam_scale": args.aam_scale,
        "long_range": [args.long_min_seconds, args.long_max_seconds],
        "short_range": [args.short_min_seconds, args.short_max_seconds],
        "validation_enrollment_model": args.validation_enrollment_model,
        "seed": args.seed,
    }


def atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    """Write a checkpoint atomically to protect against interrupted training."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def save_checkpoint(
    destination: Path,
    student: nn.Module,
    teacher: nn.Module,
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    step: int,
    best_eer: float,
    best_tar: float,
    history: list[dict[str, Any]],
    signature: dict[str, Any],
    wandb_run_id: str | None,
) -> None:
    """Persist all state required for an exact D_FULL and W&B resume."""

    payload = {
        "system": SYSTEM_NAME,
        "student": student.state_dict(),
        "teacher": teacher.state_dict(),
        "aam_head": head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "global_step": step,
        "best_noise_eer": best_eer,
        "best_noise_tar": best_tar,
        "history": history,
        "signature": signature,
        "wandb_run_id": wandb_run_id,
    }
    atomic_torch_save(payload, destination)


def initialize_wandb(
    args: argparse.Namespace,
    signature: dict[str, Any],
    validation_speakers: int,
    checkpoint: dict[str, Any] | None,
) -> Any:
    """Start or resume W&B without interrupting training if logging is unavailable."""

    if args.wandb_mode == "disabled":
        print("W&B logging disabled.", flush=True)
        return None
    try:
        import wandb

        if args.wandb_mode == "online":
            api_key = os.environ.get("WANDB_API_KEY")
            if api_key:
                wandb.login(key=api_key)
            else:
                wandb.login()

        previous_id = checkpoint.get("wandb_run_id") if checkpoint else None
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            group="mentor-stage-m3",
            job_type="train",
            tags=["D_FULL", "EMA", "Gaussian-KL", "full-training-split"],
            id=previous_id,
            resume="allow" if previous_id else None,
            mode=args.wandb_mode,
            dir=str(args.output_dir),
            config={
                **signature,
                "training_speakers": len(signature["train_speaker_ids"]),
                "validation_speakers": validation_speakers,
                "recordings_per_batch": 2 * args.speakers_per_batch * args.pairs_per_speaker,
                "model_source": args.model_source,
            },
        )
        run.define_metric("global_step")
        run.define_metric("train/*", step_metric="global_step")
        run.define_metric("validation/*", step_metric="global_step")
        run.define_metric("validation/noise_eer", step_metric="global_step", summary="min")
        run.define_metric("validation/noise_tar", step_metric="global_step", summary="max")
        print(f"W&B run: {run.url or run.id}", flush=True)
        return run
    except Exception as exc:
        print(f"W&B unavailable; continuing without online logging: {exc!r}", flush=True)
        return None


def wandb_log(run: Any, step: int, values: dict[str, Any]) -> None:
    """Prevent an experiment-tracking outage from aborting the training job."""

    if run is None:
        return
    try:
        run.log({"global_step": int(step), **values})
    except Exception as exc:
        print(f"W&B logging warning: {exc!r}", flush=True)


def make_loader(
    dataset: SpeakerPairDataset,
    speakers: list[str],
    args: argparse.Namespace,
    start_step: int,
) -> DataLoader:
    """Create a reproducible speaker-balanced loader with CPU prefetch workers."""

    sampler = SpeakerBalancedBatchSampler(
        speakers,
        num_batches=args.steps - start_step,
        speakers_per_batch=args.speakers_per_batch,
        pairs_per_speaker=args.pairs_per_speaker,
        seed=args.seed,
        start_step=start_step,
    )
    options: dict[str, Any] = {
        "dataset": dataset,
        "batch_sampler": sampler,
        "collate_fn": collate_pairs,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers:
        options["persistent_workers"] = True
        options["prefetch_factor"] = 2
    return DataLoader(**options)


def train(
    args: argparse.Namespace,
    records: dict[str, list[Recording]],
    train_speakers: list[str],
    validation_speakers: list[str],
    manifest: Path,
) -> None:
    """Run the complete AAM + cosine alignment + Gaussian-KL EMA experiment."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    else:
        print("WARNING: CUDA is unavailable; full ECAPA training on CPU will be very slow.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    speaker_to_label = {speaker: index for index, speaker in enumerate(train_speakers)}
    mapping_path = args.output_dir / "training_speaker_labels.json"
    mapping_path.write_text(json.dumps(speaker_to_label, indent=2, sort_keys=True), encoding="utf-8")
    split_hash = file_sha256(args.speaker_splits)
    recording_hash = file_sha256(manifest)
    signature = checkpoint_signature(args, train_speakers, split_hash, recording_hash)
    augmenter = MusanAugmenter(args.musan_dir, args)
    dataset = SpeakerPairDataset(records, speaker_to_label, augmenter, args)
    validation_assignments = create_validation_assignments(records, validation_speakers, args)

    student = load_pretrained_encoder(args).to(device)
    teacher = copy.deepcopy(student).to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    with torch.no_grad():
        dummy = torch.zeros(2, args.sample_rate, device=device)
        embedding_dim = int(student(dummy, torch.ones(2, device=device)).shape[-1])
    head = AAMSoftmaxHead(
        embedding_dim, len(train_speakers), args.aam_margin, args.aam_scale
    ).to(device)
    if tuple(head.weight.shape) != (len(train_speakers), embedding_dim):
        raise AssertionError("The AAM classifier does not match the training-speaker count.")

    optimizer = torch.optim.AdamW(
        [
            {"params": student.parameters(), "lr": args.encoder_lr},
            {"params": head.parameters(), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )

    def learning_rate_multiplier(step: int) -> float:
        if step < args.warmup_steps:
            return max(step, 1) / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_multiplier)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and not args.no_amp
    )
    latest = args.output_dir / "latest.pt"
    best_eer_path = args.output_dir / "best_validation_eer.pt"
    best_tar_path = args.output_dir / "best_validation_tar_at_target_far.pt"
    history: list[dict[str, Any]] = []
    step = 0
    best_eer = float("inf")
    best_tar = -float("inf")
    checkpoint = None

    if latest.exists():
        print(f"Resuming checkpoint: {latest}", flush=True)
        checkpoint = torch.load(latest, map_location=device, weights_only=False)
        if checkpoint.get("signature") != signature:
            mismatches = {
                key: {"saved": checkpoint.get("signature", {}).get(key), "current": value}
                for key, value in signature.items()
                if checkpoint.get("signature", {}).get(key) != value
            }
            raise RuntimeError(
                "Checkpoint configuration does not match this run. "
                "Use a different --output-dir for a new experiment:\n"
                + json.dumps(mismatches, indent=2)
            )
        student.load_state_dict(checkpoint["student"])
        teacher.load_state_dict(checkpoint["teacher"])
        head.load_state_dict(checkpoint["aam_head"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        step = int(checkpoint["global_step"])
        best_eer = float(checkpoint["best_noise_eer"])
        best_tar = float(checkpoint["best_noise_tar"])
        history = list(checkpoint["history"])

    print(
        json.dumps(
            {
                "system": SYSTEM_NAME,
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "training_speakers": len(train_speakers),
                "validation_speakers": len(validation_speakers),
                "embedding_dimension": embedding_dim,
                "aam_head_shape": list(head.weight.shape),
                "distinct_speakers_per_batch": args.speakers_per_batch,
                "pairs_per_speaker": args.pairs_per_speaker,
                "teacher_recordings_per_batch": args.speakers_per_batch * args.pairs_per_speaker,
                "student_recordings_per_batch": args.speakers_per_batch * args.pairs_per_speaker,
                "short_seconds": [args.short_min_seconds, args.short_max_seconds],
                "long_seconds": [args.long_min_seconds, args.long_max_seconds],
                "max_steps": args.steps,
                "resume_step": step,
                "musan_train_files": sum(map(len, augmenter.training.values())),
                "musan_validation_files": sum(map(len, augmenter.validation.values())),
                "split_sha256": split_hash,
                "recording_manifest": str(manifest),
            },
            indent=2,
        ),
        flush=True,
    )
    if step >= args.steps:
        print("Training already complete.", flush=True)
        return

    run = initialize_wandb(args, signature, len(validation_speakers), checkpoint)
    wandb_run_id = run.id if run is not None else checkpoint.get("wandb_run_id") if checkpoint else None

    def persist(destination: Path) -> None:
        save_checkpoint(
            destination,
            student,
            teacher,
            head,
            optimizer,
            scheduler,
            scaler,
            step,
            best_eer,
            best_tar,
            history,
            signature,
            wandb_run_id,
        )

    if wandb_run_id and (checkpoint is None or checkpoint.get("wandb_run_id") != wandb_run_id):
        persist(latest)

    loader = make_loader(dataset, train_speakers, args, step)
    student.train()
    teacher.eval()
    head.train()
    averages: dict[str, float] = defaultdict(float)
    recent = 0
    window_start = time.perf_counter()
    started_at = time.perf_counter()
    exit_code = 0

    try:
        for batch in loader:
            step += 1
            metadata = batch["metadata"]
            if len({item["speaker_id"] for item in metadata}) != args.speakers_per_batch:
                raise AssertionError("The batch does not contain the requested number of distinct speakers.")
            if any(item["teacher_source_id"] == item["student_source_id"] for item in metadata):
                raise AssertionError("A teacher/student pair reused the same source recording.")

            teacher_audio = batch["teacher_audio"].to(device, non_blocking=True)
            student_audio = batch["student_audio"].to(device, non_blocking=True)
            teacher_lengths = batch["teacher_lengths"].to(device, non_blocking=True)
            student_lengths = batch["student_lengths"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            if int(labels.min()) < 0 or int(labels.max()) >= len(train_speakers):
                raise AssertionError("An AAM target lies outside the training-speaker class range.")

            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                with autocast_context(device, args):
                    teacher_embedding = F.normalize(
                        teacher(teacher_audio, teacher_lengths).float(), dim=-1
                    )
            with autocast_context(device, args):
                student_embedding = F.normalize(
                    student(student_audio, student_lengths).float(), dim=-1
                )
                loss_aam, _ = head(student_embedding, labels)
                loss_vector = vector_alignment_loss(teacher_embedding, student_embedding)
                loss_kl = symmetric_diagonal_gaussian_kl(
                    teacher_embedding, student_embedding, args.kl_epsilon
                )
                loss = (
                    loss_aam
                    + args.lambda_vector * loss_vector
                    + args.lambda_kl * loss_kl
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at step {step}: aam={loss_aam}, "
                    f"vector={loss_vector}, gaussian_kl={loss_kl}"
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            student_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), args.gradient_clip)
            head_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), args.gradient_clip)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_updated = scaler.get_scale() >= old_scale
            if optimizer_updated:
                scheduler.step()
                ema_update(teacher, student, args.ema_alpha)

            averages["loss_total"] += float(loss.detach())
            averages["loss_aam"] += float(loss_aam.detach())
            averages["loss_vector"] += float(loss_vector.detach())
            averages["loss_gaussian_kl"] += float(loss_kl.detach())
            averages["student_grad_norm"] += float(student_norm)
            averages["head_grad_norm"] += float(head_norm)
            averages["optimizer_steps_skipped"] += float(not optimizer_updated)
            recent += 1

            if step % args.log_every == 0:
                elapsed = max(time.perf_counter() - window_start, 1e-9)
                speed = recent / elapsed
                values = {name: value / recent for name, value in averages.items()}
                remaining_hours = (args.steps - step) / speed / 3600.0
                print(
                    f"step={step:05d}/{args.steps} "
                    f"loss={values['loss_total']:.4f} "
                    f"aam={values['loss_aam']:.4f} "
                    f"vec={values['loss_vector']:.4f} "
                    f"kl={values['loss_gaussian_kl']:.4f} "
                    f"speed={speed:.2f} step/s "
                    f"eta={remaining_hours:.2f}h",
                    flush=True,
                )
                payload = {
                    **{f"train/{name}": value for name, value in values.items()},
                    "train/steps_per_second": speed,
                    "train/remaining_hours": remaining_hours,
                    "train/lr_encoder": optimizer.param_groups[0]["lr"],
                    "train/lr_head": optimizer.param_groups[1]["lr"],
                }
                if device.type == "cuda":
                    payload["train/gpu_allocated_gb"] = torch.cuda.memory_allocated() / 2**30
                    payload["train/gpu_reserved_gb"] = torch.cuda.memory_reserved() / 2**30
                wandb_log(run, step, payload)
                averages.clear()
                recent = 0
                window_start = time.perf_counter()

            if step % args.save_every == 0:
                persist(latest)
            if args.periodic_save_every and step % args.periodic_save_every == 0:
                persist(args.output_dir / f"periodic_step_{step:06d}.pt")

            if step % args.eval_every == 0 or step == args.steps:
                metrics = evaluate_verification(
                    student,
                    teacher,
                    validation_assignments,
                    augmenter,
                    args,
                    device,
                )
                metrics["step"] = step
                history.append(metrics)
                print("Validation: " + json.dumps(metrics, indent=2), flush=True)
                wandb_log(
                    run,
                    step,
                    {
                        f"validation/{name}": value
                        for name, value in metrics.items()
                        if name != "step" and isinstance(value, (int, float))
                    },
                )
                if float(metrics["noise_eer"]) < best_eer:
                    best_eer = float(metrics["noise_eer"])
                    persist(best_eer_path)
                    print(f"New best noisy validation EER: {best_eer:.4f}", flush=True)
                if float(metrics["noise_tar"]) > best_tar:
                    best_tar = float(metrics["noise_tar"])
                    persist(best_tar_path)
                    print(f"New best noisy validation TAR: {best_tar:.4f}", flush=True)
                persist(latest)
                (args.output_dir / "validation_history.json").write_text(
                    json.dumps(history, indent=2), encoding="utf-8"
                )
                student.train()
                teacher.eval()
                head.train()
                window_start = time.perf_counter()

    except BaseException:
        exit_code = 1
        if step > 0:
            try:
                persist(latest)
                print(f"Emergency checkpoint saved at step {step}: {latest}", flush=True)
            except Exception as save_error:
                print(f"Emergency checkpoint failed: {save_error!r}", file=sys.stderr, flush=True)
        raise
    finally:
        if run is not None:
            try:
                run.finish(exit_code=exit_code)
            except Exception as exc:
                print(f"W&B finalization warning: {exc!r}", flush=True)

    elapsed_hours = (time.perf_counter() - started_at) / 3600.0
    summary = {
        "system": SYSTEM_NAME,
        "completed_steps": step,
        "training_speakers": len(train_speakers),
        "validation_speakers": len(validation_speakers),
        "best_noisy_validation_eer": best_eer,
        "best_noisy_validation_tar": best_tar,
        "elapsed_hours_this_invocation": elapsed_hours,
        "latest_checkpoint": str(latest),
        "best_eer_checkpoint": str(best_eer_path),
        "best_tar_checkpoint": str(best_tar_path),
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("D_FULL training complete: " + json.dumps(summary, indent=2), flush=True)


def self_test() -> None:
    """Check Gaussian KL, cosine matching, EMA, crop, sampler, and split logic."""

    torch.manual_seed(2026)
    embeddings = F.normalize(torch.randn(32, 192), dim=-1)
    if abs(float(vector_alignment_loss(embeddings, embeddings))) >= 1e-5:
        raise AssertionError("Identical normalized embeddings must have zero cosine loss.")
    if abs(float(symmetric_diagonal_gaussian_kl(embeddings, embeddings, 1e-5))) >= 1e-5:
        raise AssertionError("Identical embedding batches must have zero Gaussian KL.")
    student_embeddings = F.normalize(torch.randn(32, 192), dim=-1).requires_grad_()
    kl = symmetric_diagonal_gaussian_kl(embeddings, student_embeddings, 1e-5)
    kl.backward()
    if student_embeddings.grad is None or not torch.isfinite(student_embeddings.grad).all():
        raise AssertionError("Gaussian-KL gradients did not reach the student embeddings.")

    teacher = nn.Linear(4, 2, bias=False)
    student = copy.deepcopy(teacher)
    with torch.no_grad():
        student.weight.add_(1.0)
    previous = teacher.weight.detach().clone()
    ema_update(teacher, student, 0.9)
    expected = 0.9 * previous + 0.1 * student.weight.detach()
    if not torch.allclose(teacher.weight, expected):
        raise AssertionError("EMA teacher update is incorrect.")

    waveform = np.zeros(20_000, dtype=np.float32)
    crop = crop_by_voiced_duration(
        waveform,
        ((1_000, 6_000), (8_000, 15_000)),
        0.5,
        np.random.default_rng(1),
        10_000,
    )
    if len(crop) < 5_000:
        raise AssertionError("Speech-aware cropping returned insufficient voiced audio.")

    sampler = SpeakerBalancedBatchSampler(
        [f"speaker_{index}" for index in range(20)],
        num_batches=2,
        speakers_per_batch=16,
        pairs_per_speaker=2,
        seed=2026,
    )
    batch = next(iter(sampler))
    if len(batch) != 32 or len({speaker for speaker, _ in batch}) != 16:
        raise AssertionError("Speaker-balanced batches do not contain 16 speakers and 32 pairs.")
    print("Self-test passed: Gaussian KL, gradients, EMA, voiced crop, and batch sampler.")


def main(argv: list[str] | None = None) -> None:
    """Prepare the full training split and execute or resume the D_FULL run."""

    args = parse_args(argv)
    if args.self_test:
        self_test()
        return
    seed_everything(args.seed)
    partitions = load_speaker_splits(args.speaker_splits)
    print(
        json.dumps(
            {
                "speaker_splits": str(args.speaker_splits),
                "train_speakers": len(partitions["train"]),
                "validation_speakers": len(partitions["val"]),
                "reserved_final_test_speakers": len(partitions["test"]),
                "cache_dir": str(args.cache_dir),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        ),
        flush=True,
    )
    records, manifest = prepare_recording_cache(args, partitions)
    records, train_speakers, validation_speakers = select_eligible_records(
        records, partitions, args
    )
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "cache_manifest": str(manifest),
                    "eligible_training_speakers": len(train_speakers),
                    "eligible_validation_speakers": len(validation_speakers),
                    "cached_recordings": sum(len(rows) for rows in records.values()),
                },
                indent=2,
            ),
            flush=True,
        )
        return
    train(args, records, train_speakers, validation_speakers, manifest)


if __name__ == "__main__":
    main()