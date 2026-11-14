# Short and noisy Vietnamese speaker verification

Code for *Do Teacher–Student Alignment and Gaussian KL Improve Short and Noisy Vietnamese Speaker Verification?* (ICIIT 2027).

All systems adapt the SpeechBrain ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`) on VieSpeaker-T and are scored on a frozen 5,000-trial sample of VieSpeaker-H under 27 conditions (enrollment 2/3/4 s × probe 1.0/1.25/1.5 s × clean/10 dB/0 dB MUSAN).

| Paper name | Training | Evaluation |
|---|---|---|
| A: pretrained | none | `eval_ecapa_pretrained_baseline.py` |
| B: AAM-only | `train_viespeaker_aam_only_full.py` | `evaluate_aam_only_full_viespeaker_h_1e1p.py` |
| C: vector-EMA | `train_c_vector_ema_full.py` | `evaluate_c_vector_ema_viespeaker_h_1e1p.py` |
| D: full (vector-EMA + Gaussian KL) | `train_viespeaker_d_full.py` | `evaluate_d_full_viespeaker_h_1e1p.py` |
| Arch-A/B/C pooling variants | `train_viespeaker_aam_only_full.py` + `temporal_pooling_variants.py` | `evaluate_aam_only_full_viespeaker_h_1e1p.py` via `tools/sampledh_supervisor/` |

`train_viespeaker_d_full.py` also holds the shared data loading, Silero VAD cache, MUSAN/RIR augmentation and speaker sampling used by the other trainers.

## Layout

- `scripts/download_viespeaker.py`: download the VieSpeaker snapshot.
- `clean_viespeaker_dataset.py`: build the cleaned (V2, 3,359-speaker) manifests used by the pooling runs.
- `tools/research/compare_aux_losses_speaker_bootstrap.py`: paired enrollment-speaker bootstrap (1,000 replicates, seed 20260927) behind the 54 auxiliary-versus-AAM intervals.
- `tools/research/check_audio_quality_semantic_parity.py`: preprocessing parity check between evaluation runs.
- `tools/sampledh_supervisor/`: the archived supervisor that ran the sampled-H evaluations for Table 2 and checked that every model was scored on the same trial rows.
- `artifacts/hc0032_revision/`: conference revision source code, training configurations, and source-hash metadata, retained in their original dated directories. This includes historical replay and calibration, paired and two-way bootstrap analyses, C/D enrollment-role ablations, matched-seed B/C/D training, memory qualification, and FP16 overflow recovery.

## Conference revision code

**VERIFIED:** The dated revision directories preserve separate implementations and their configurations; they do not identify every earlier attempt as a completed experiment. The most recent matched-cohort training entry point is `artifacts/hc0032_revision/amp_recovery_20261108_083452/scripts/train_recovery.py`, with six configurations in the adjacent `configs/` directory (B/C/D, seeds 2027/2028). It depends on the earlier `fixed_capacity_20261108_060219/scripts/` and `cohort_finish_20261107_211439/matched/` implementations. Their source snapshots are retained at the paths used by the code.

**VERIFIED:** This is an archive of the implemented code and settings. Hardcoded lab paths, interpreter paths, initialization/input hashes, and numerical settings are preserved. Several scripts perform file access, write outputs, or execute work at import time; inspect the source before importing or running it. The launchers and supervisors can start training or evaluation.

To use these scripts elsewhere, first provision the referenced private inputs and review all paths and output locations. The archive does not include datasets, audio, splits or trial manifests, pretrained bundles, initialization files, checkpoints, cached embeddings, score files, generated input registries, qualification reports, or run logs. It also omits manuscript/reviewer documents and their dedicated narrative generators. Consequently, end-to-end supervisors that require those generators or frozen run evidence are not standalone workflows in this repository. The preserved source-hash metadata can reference such omitted files; it has not been rewritten to describe a new run.

**UNVERIFIED:** Runtime reproducibility and experiment completion are not established by this code-only update. Validation of this update covers Python syntax, shell syntax, and JSON parsing; it does not execute training, inference, or analysis.

## Notes

- `SHA256SUMS` describes the original release files and is preserved unchanged. The revision additions preserve the server source bytes; they are not covered by that original checksum list. Default paths inside the original release scripts point to the authors' machine, so pass your own paths on the command line (`python <script> --help`). The revision scripts require the additional path and side-effect review described above.
- Datasets, checkpoints and score files are not included.
- In the original release, training seed is 2026 for B/C/D and Arch-B/C, 20260913 for Arch-A; each original system was trained once. The separately archived revision configurations must not be treated as additional completed results solely because they are present here.
