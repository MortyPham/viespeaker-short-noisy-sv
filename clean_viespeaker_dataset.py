#!/usr/bin/env python3
"""Create non-destructive cleaned VieSpeaker training manifests and audit reports.

The downloaded Parquet files and official trials are never modified. Speakers
involved in byte-identical cross-label audio are quarantined because the audio
alone cannot determine which identity label is correct.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "experiments/shared/d_full_audio_cache"
RECORDINGS = CACHE_DIR / "recordings.csv"
CLEAN_RECORDINGS = CACHE_DIR / "recordings.cleaned.csv"
SPEAKER_SPLITS = ROOT / "experiments/shared/speaker_splits.csv"
CLEAN_SPEAKER_SPLITS = ROOT / "experiments/shared/speaker_splits.cleaned.csv"
CLEAN_SPEAKER_LABELS = ROOT / "experiments/shared/training_speaker_labels.cleaned.json"
H_TRIALS = ROOT / "datasets/VieSpeaker/repository/trials/viespeaker_H.txt"
AUDIT_DIR = ROOT / "experiments/dataset_audit"
DUPLICATE_REPORT = AUDIT_DIR / "duplicate_audio.csv"
QUARANTINE_REPORT = AUDIT_DIR / "quarantined_speakers.csv"
H_ID_REPORT = AUDIT_DIR / "h_id_label_inconsistencies.csv"
CLEANING_REPORT = AUDIT_DIR / "cleaning_report.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_recordings() -> tuple[list[str], list[dict[str, str]]]:
    with RECORDINGS.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"Missing header: {RECORDINGS}")
        rows = list(reader)
        return list(reader.fieldnames), rows


def find_duplicate_groups(rows: list[dict[str, str]]) -> list[tuple[str, list[int]]]:
    paths = [CACHE_DIR / row["relative_path"] for row in rows]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} cached WAV files; first: {missing[0]}")
    workers = min(8, max(1, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        hashes = list(executor.map(file_sha256, paths))
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, digest in enumerate(hashes):
        grouped[digest].append(index)
    return sorted(
        ((digest, indices) for digest, indices in grouped.items() if len(indices) > 1),
        key=lambda item: item[0],
    )


def h_identity_inconsistencies() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with H_TRIALS.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            enrollment, probe, label = line.split()
            enrollment_id = enrollment.split("/", 1)[0]
            probe_id = probe.split("/", 1)[0]
            same_id = enrollment_id == probe_id
            if (label == "target") == same_id:
                continue
            rows.append(
                {
                    "line_number": line_number,
                    "kind": "cross_id_target" if label == "target" else "same_id_nontarget",
                    "label": label,
                    "enrollment": enrollment,
                    "probe": probe,
                    "enrollment_speaker_id": enrollment_id,
                    "probe_speaker_id": probe_id,
                }
            )
    return rows


def main() -> None:
    fields, recordings = load_recordings()
    duplicate_groups = find_duplicate_groups(recordings)

    duplicate_rows: list[dict[str, object]] = []
    quarantined: set[str] = set()
    same_id_drop_indices: set[int] = set()
    cross_groups = 0
    same_groups = 0

    for group_number, (digest, indices) in enumerate(duplicate_groups, start=1):
        speakers = {recordings[index]["speaker_id"] for index in indices}
        cross_label = len(speakers) > 1
        if cross_label:
            cross_groups += 1
            quarantined.update(speakers)
        else:
            same_groups += 1
            ordered = sorted(indices, key=lambda index: recordings[index]["source_id"])
            same_id_drop_indices.update(ordered[1:])
        for index in indices:
            row = recordings[index]
            duplicate_rows.append(
                {
                    "duplicate_group": group_number,
                    "sha256": digest,
                    "cross_speaker_labels": cross_label,
                    "action": "quarantine_speaker" if cross_label else (
                        "keep" if index not in same_id_drop_indices else "remove_duplicate"
                    ),
                    "speaker_id": row["speaker_id"],
                    "split": row["split"],
                    "source_id": row["source_id"],
                    "relative_path": row["relative_path"],
                }
            )

    preliminary_recordings = [
        row for index, row in enumerate(recordings)
        if row["speaker_id"] not in quarantined and index not in same_id_drop_indices
    ]

    with SPEAKER_SPLITS.open(newline="", encoding="utf-8-sig") as handle:
        split_reader = csv.DictReader(handle)
        if split_reader.fieldnames is None:
            raise RuntimeError(f"Missing header: {SPEAKER_SPLITS}")
        split_fields = list(split_reader.fieldnames)
        split_rows = list(split_reader)

    by_speaker: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in preliminary_recordings:
        if row["split"] == "train" and float(row["voiced_seconds"]) >= 1.5:
            by_speaker[row["speaker_id"]].append(row)
    clean_train_ids = sorted(
        speaker for speaker, rows in by_speaker.items()
        if len(rows) >= 2 and any(float(row["voiced_seconds"]) >= 4.0 for row in rows)
    )
    original_train_ids = {
        row["speaker_id"] for row in split_rows if row["split"].strip() == "train"
    }
    ineligible_train_ids = sorted(original_train_ids - quarantined - set(clean_train_ids))
    excluded_train_ids = quarantined | set(ineligible_train_ids)
    cleaned_recordings = [
        row for row in preliminary_recordings
        if not (row["split"] == "train" and row["speaker_id"] in excluded_train_ids)
    ]
    clean_split_rows = [
        row for row in split_rows
        if not (row["split"].strip() == "train" and row["speaker_id"] in excluded_train_ids)
    ]
    write_csv(CLEAN_RECORDINGS, fields, cleaned_recordings)
    write_csv(CLEAN_SPEAKER_SPLITS, split_fields, clean_split_rows)
    write_csv(
        DUPLICATE_REPORT,
        [
            "duplicate_group", "sha256", "cross_speaker_labels", "action",
            "speaker_id", "split", "source_id", "relative_path",
        ],
        duplicate_rows,
    )
    CLEAN_SPEAKER_LABELS.write_text(
        json.dumps({speaker: index for index, speaker in enumerate(clean_train_ids)}, indent=2) + "\n",
        encoding="utf-8",
    )

    quarantine_rows = []
    for speaker in sorted(quarantined):
        partners = sorted(
            {
                recordings[index]["speaker_id"]
                for _, indices in duplicate_groups
                if any(recordings[item]["speaker_id"] == speaker for item in indices)
                for index in indices
                if recordings[index]["speaker_id"] != speaker
            }
        )
        quarantine_rows.append(
            {
                "speaker_id": speaker,
                "reason": "byte_identical_audio_under_different_speaker_ids",
                "conflicting_speaker_ids": ";".join(partners),
            }
        )
    write_csv(
        QUARANTINE_REPORT,
        ["speaker_id", "reason", "conflicting_speaker_ids"],
        quarantine_rows,
    )

    h_rows = h_identity_inconsistencies()
    write_csv(
        H_ID_REPORT,
        [
            "line_number", "kind", "label", "enrollment", "probe",
            "enrollment_speaker_id", "probe_speaker_id",
        ],
        h_rows,
    )

    report = {
        "policy": {
            "original_files_modified": False,
            "cross_speaker_duplicate_audio": "quarantine every involved speaker",
            "same_speaker_duplicate_audio": "keep lexicographically first source_id",
            "official_h_trials": "audit only; do not rewrite benchmark labels",
        },
        "input_recordings": len(recordings),
        "clean_recordings": len(cleaned_recordings),
        "removed_recording_rows": len(recordings) - len(cleaned_recordings),
        "duplicate_hash_groups": len(duplicate_groups),
        "cross_speaker_duplicate_groups": cross_groups,
        "same_speaker_duplicate_groups": same_groups,
        "quarantined_speakers": sorted(quarantined),
        "quarantined_speaker_count": len(quarantined),
        "ineligible_training_speakers": ineligible_train_ids,
        "ineligible_training_speaker_count": len(ineligible_train_ids),
        "eligible_clean_training_speakers": len(clean_train_ids),
        "h_id_label_inconsistencies": len(h_rows),
        "outputs": {
            "clean_recordings": str(CLEAN_RECORDINGS),
            "clean_speaker_splits": str(CLEAN_SPEAKER_SPLITS),
            "clean_training_labels": str(CLEAN_SPEAKER_LABELS),
            "duplicate_report": str(DUPLICATE_REPORT),
            "quarantine_report": str(QUARANTINE_REPORT),
            "h_identity_report": str(H_ID_REPORT),
        },
    }
    CLEANING_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
