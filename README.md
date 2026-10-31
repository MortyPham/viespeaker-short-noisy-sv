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

## Notes

- Files are released byte-identical to the versions that produced the paper results; `SHA256SUMS` lists their hashes. Default paths inside the scripts point to the authors' machine, so pass your own paths on the command line (`python <script> --help`).
- Datasets, checkpoints and score files are not included.
- Training seed is 2026 for B/C/D and Arch-B/C, 20260913 for Arch-A. Each system was trained once.
