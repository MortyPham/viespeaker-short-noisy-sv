#!/bin/bash
set -euo pipefail
set -o noclobber
exec env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/ml4u/ENTER/bin/python /media/ml4u/Samsung_T52/ptnk_team/mortyhuaphuoc/artifacts/hc0032_revision/cohort_finish_20261107_211439/queue/supervisor_v3.py > /media/ml4u/Samsung_T52/ptnk_team/mortyhuaphuoc/artifacts/hc0032_revision/cohort_finish_20261107_211439/logs/queue_supervisor.log 2>&1
