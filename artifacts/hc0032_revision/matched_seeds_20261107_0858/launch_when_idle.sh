#!/usr/bin/env bash
set -euo pipefail
cd '/media/ml4u/Samsung_T52/ptnk_team/mortyhuaphuoc'
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
exec /home/ml4u/.conda/envs/asr/bin/python '/media/ml4u/Samsung_T52/ptnk_team/mortyhuaphuoc/artifacts/hc0032_revision/matched_seeds_20261107_0858/execute_plan.py' >> '/media/ml4u/Samsung_T52/ptnk_team/mortyhuaphuoc/artifacts/hc0032_revision/matched_seeds_20261107_0858/logs/plan_execution.log' 2>&1
