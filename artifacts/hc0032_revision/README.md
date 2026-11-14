# Conference revision archive

This directory contains the dated source implementations, experiment configurations, source snapshots, and source-hash metadata developed during the paper's conference revision. Directory names retain the original workflow layout. Earlier attempts and later recovery implementations are kept separately; the presence of code or configurations does not establish that an experiment completed.

## Contents

| Directory | Contents |
|---|---|
| `prompt1_20261106_2153/` | Checkpoint, validation, input, and cached-score inspection code. |
| `overnight_20261106/` | Replay queue, calibration, paired bootstrap, and two-way clustering sensitivity. |
| `cpu_replay_20261107_0834/` | CPU replay smoke script. |
| `matched_seeds_20261107_0858/` | Initial matched-seed preparation, training, evaluation, and configurations. |
| `completion_20261107_154453/` | Follow-up analysis, replay, calibration, and matched-seed implementations. |
| `concurrent_20261107_160527/` | Concurrent-inference and matched-training implementation variant. |
| `bounded_20261107_162926/` | Bounded inference and training-memory probes. |
| `backend_diagnostic_20261107_163125/` | Inference backend diagnostics. |
| `roles_cd_20261107_165713/` | C/D enrollment-role ablations, paired statistics, output checks, and B diagnostics. |
| `cohort_finish_20261107_211439/` | Full-batch engines, memory qualification, and matched training/evaluation. |
| `fixed_capacity_20261108_060219/` | Fixed-capacity state staging, qualification, and configurations. |
| `amp_recovery_20261108_083452/` | FP16 overflow recovery, diagnostics, and matched configurations. |
| `reviewer_audit_20261108_151108/` | Saved-evidence inspection code; narrative documents are excluded. |

The archive contains 168 Python files, five shell launchers, 48 training configurations, and 33 source-hash metadata files. The Python files include 70 snapshots of the original release sources, retained at the locations used by the archived imports and scripts.

## Evaluation protocol

The main README describes the frozen, balanced 5,000-trial sample of VieSpeaker-H used for the 27 duration/noise conditions. This sampled protocol is distinct from evaluation on the full official H trial list. Eligibility exclusions can vary by condition: trials require sufficient voiced duration and usable embeddings on both sides. Compare systems on identical scored trial rows, retaining the trial labels and recording pairs, and report target/nontarget counts and exclusions for each condition.

## Dependencies and execution

The latest matched-cohort training entry point is [train_recovery.py](amp_recovery_20261108_083452/scripts/train_recovery.py), with six [configurations](amp_recovery_20261108_083452/configs/) for B/C/D and seeds 2027/2028. It imports [engine_recovery_v2.py](amp_recovery_20261108_083452/scripts/engine_recovery_v2.py), which depends on [engine_fixed.py](fixed_capacity_20261108_060219/scripts/engine_fixed.py) and the earlier [matched engine implementations](cohort_finish_20261107_211439/matched/). Other dated variants retain their own dependencies and settings.

Hardcoded lab paths, interpreter paths, initialization/input hashes, and numerical settings are preserved. Review the relevant scripts, configurations, dependencies, and output locations before use elsewhere. Some scripts read files, create output directories, parse arguments, or execute work at import time. Importing them is not a safe substitute for inspecting their source. Launchers and supervisors can start training, inference, or analysis; they are not installation or demonstration commands.

Datasets, checkpoints, and generated outputs are excluded from this repository. Some scripts require lab-specific paths and inputs. Runtime reproduction of these archived workflows has not been validated. They require inputs and run evidence that are not distributed here:

- Datasets and audio, speaker splits, recording and trial manifests, and pretrained bundles.
- Initialization tensors, checkpoints, cached embeddings, and score files.
- Generated model/input registries, frozen-input inventories, qualification reports, and run logs.

Manuscript/reviewer documents and dedicated narrative generators are also omitted. Some supervisors invoke these generators or require frozen evidence from an earlier stage, so they cannot run end to end from this checkout alone. Do not bypass those checks or interpret archived configurations as completion records.

## Exploratory pooling variants

The original release also retains [temporal_pooling_variants.py](../../temporal_pooling_variants.py), selected through `--pooling-variant` in [train_viespeaker_aam_only_full.py](../../train_viespeaker_aam_only_full.py), and the [sampled-H supervisor](../../tools/sampledh_supervisor/). These exploratory Arch-A/B/C variants are separate from the paper's A/B/C/D system names. They cover original pooling, global/local pooling, and a gate that additionally receives the batch-relative valid-length fraction. That last input is not an absolute duration in seconds. The original release notes record seed 20260913 for Arch-A and 2026 for Arch-B/C; these files and settings are preserved.

## Checksums and validation scope

The root [SHA256SUMS](../../SHA256SUMS) lists original release files, not this revision archive. It is unchanged. The archived additions preserve the server source bytes. Source-hash metadata records the original paths and hashes and may reference omitted files, including narrative generators or private run evidence; it has not been rewritten as a self-contained release manifest. The whitespace-only line in `cohort_finish_20261107_211439/matched/qualify_capacity.py:128` is retained because archived inventories hash that source file.

Preparation of the code archive checked Python syntax without imports or execution, shell syntax, JSON parsing, structural/numerical consistency of the training configurations, source/staged byte equality, and the original release checksums. These checks did not execute training, inference, or statistical analysis, and do not establish runtime compatibility, numerical reproduction, or experiment completion. The original release notes identify seed 2026 for B/C/D; the separate revision configurations do not by themselves establish additional completed runs.
