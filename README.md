# Short and noisy Vietnamese speaker verification

Code accompanying *Do Teacher–Student Alignment and Gaussian KL Improve Short and Noisy Vietnamese Speaker Verification?* (ICIIT 2027).

## Task and systems

The task is speaker verification from short Vietnamese speech under clean and noisy conditions. System A uses the pretrained SpeechBrain ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`) without adaptation. Systems B/C/D adapt that encoder on VieSpeaker-T with the objectives below; EMA denotes an exponential moving average teacher.

Evaluation uses a frozen, balanced 5,000-trial sample of VieSpeaker-H, with one enrollment recording and one probe per trial. The 27 conditions combine enrollment durations of 2/3/4 seconds, probe durations of 1.0/1.25/1.5 seconds, and clean/10 dB/0 dB MUSAN probe noise. Enrollment is clean.

| System | Training | Evaluation |
|---|---|---|
| A: pretrained | None | `eval_ecapa_pretrained_baseline.py` |
| B: AAM-only | `train_viespeaker_aam_only_full.py` | `evaluate_aam_only_full_viespeaker_h_1e1p.py` |
| C: AAM + vector alignment + EMA | `train_c_vector_ema_full.py` | `evaluate_c_vector_ema_viespeaker_h_1e1p.py` |
| D: C + Gaussian KL | `train_viespeaker_d_full.py` | `evaluate_d_full_viespeaker_h_1e1p.py` |

## Repository layout

- `requirements.txt`: dependency versions recorded for the original runs.
- `scripts/download_viespeaker.py`: Dataset download script.
- `clean_viespeaker_dataset.py`: cleaned-manifest preparation.
- `tools/research/`: paired bootstrap and preprocessing parity checks.
- `tools/sampledh_supervisor/`: archived sampled-H evaluation orchestration.
- [Conference revision archive](artifacts/hc0032_revision/README.md): Conference revision implementations and configurations.

## Usage

Install the listed dependencies in a Python/PyTorch environment and inspect training and evaluation options from the repository root:

```bash
python -m pip install -r requirements.txt
python train_viespeaker_aam_only_full.py --help
python eval_ecapa_pretrained_baseline.py --help
```

Required local inputs include VieSpeaker data, speaker and recording manifests, pretrained model files, model checkpoints for adapted-system evaluation, and MUSAN noise audio. Command-line arguments configure data, cache, model, and output paths.
