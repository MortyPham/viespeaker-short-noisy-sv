#!/usr/bin/env bash
set -Eeuo pipefail

readonly PROJECT="/media/ml4u/Samsung_T52/ptnk_team/mortyhuaphuoc"
readonly SUPERVISOR="${PROJECT}/experiments/evaluation/supervisors/sampledh_multimodel_resume_20260925_120132"
readonly SNAPSHOT="${SUPERVISOR}/immutable_snapshot"
readonly RESULT_ROOT="${PROJECT}/experiments/evaluation/sampledh_multimodel_20260922"
readonly LOCK="${PROJECT}/experiments/evaluation/supervisors/.locks/sampledh_multimodel_resume_20260925_120132.lock"
readonly PYTHON="/home/ml4u/miniconda3/bin/python"
readonly REPOSITORY="${PROJECT}/datasets/VieSpeaker/repository"
readonly MUSAN="${PROJECT}/datasets/musan"
readonly PRETRAINED="${PROJECT}/cache/huggingface/hub/models--speechbrain--spkrec-ecapa-voxceleb/snapshots/0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
readonly CACHE="${PROJECT}/experiments/shared/viespeaker_h_1e1p_cache"
readonly PROTOCOL="${PROJECT}/experiments/evaluation/architecture_A_viespeaker_h_1e1p/final_seed20260818_n5000/fixed_trials.csv"
readonly LOG="${SUPERVISOR}/supervisor.log"
readonly STATUS_JSON="${SUPERVISOR}/supervisor_status.json"
readonly MODEL_STATUS="${SUPERVISOR}/model_status.csv"
readonly GPU_CSV="${SUPERVISOR}/gpu_checks.csv"
readonly FINAL_STATUS="${SUPERVISOR}/final_status.txt"
readonly DEADLINE_EPOCH="$(( $(date +%s) + 72 * 60 * 60 ))"
FINALIZED=0
CURRENT_MODEL="none"

utc_now() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
local_now() { date '+%Y-%m-%dT%H:%M:%S%z'; }
log() { printf '[utc=%s local=%s] %s\n' "$(utc_now)" "$(local_now)" "$*" | tee -a "${LOG}"; }
write_status() {
    local state="$1" detail="$2" tmp="${STATUS_JSON}.tmp.$$"
    printf '{"utc":"%s","local":"%s","state":"%s","current_model":"%s","detail":"%s","deadline_epoch":%s}\n' \
        "$(utc_now)" "$(local_now)" "${state}" "${CURRENT_MODEL}" "${detail//\"/\\\"}" "${DEADLINE_EPOCH}" > "${tmp}"
    mv "${tmp}" "${STATUS_JSON}"
}
finish() {
    local classification="$1" code="$2" detail="$3"
    if (( FINALIZED == 0 )); then
        FINALIZED=1
        write_status "${classification}" "${detail}"
        printf 'classification=%s\nutc=%s\nlocal=%s\ndetail=%s\nexit_code=%s\n' \
            "${classification}" "$(utc_now)" "$(local_now)" "${detail}" "${code}" > "${FINAL_STATUS}"
        log "FINAL ${classification}: ${detail} (exit ${code})"
    fi
    trap - ERR EXIT INT TERM
    exit "${code}"
}
on_err() { local code=$? line=${BASH_LINENO[0]:-unknown}; finish "EXECUTION_FAILED" "${code}" "resume supervisor error at line ${line}"; }
on_int() { finish "EXECUTION_FAILED" 130 "resume supervisor received INT; no evaluator was signaled by the supervisor"; }
on_term() { finish "EXECUTION_FAILED" 143 "resume supervisor received TERM; no evaluator was signaled by the supervisor"; }
on_exit() {
    local code=$?
    if (( FINALIZED == 0 )); then
        FINALIZED=1
        write_status "EXECUTION_FAILED" "unexpected resume supervisor exit"
        printf 'classification=EXECUTION_FAILED\nutc=%s\nlocal=%s\ndetail=unexpected resume supervisor exit\nexit_code=%s\n' \
            "$(utc_now)" "$(local_now)" "${code}" > "${FINAL_STATUS}"
    fi
}
trap on_err ERR
trap on_int INT
trap on_term TERM
trap on_exit EXIT

exec 9>"${LOCK}"
if ! flock -n 9; then
    finish "BLOCKED_CHECKPOINT_AMBIGUITY" 73 "another resume supervisor holds ${LOCK}"
fi
if pgrep -af 'evaluate_.*viespeaker_h_1e1p.py.*sampledh_multimodel_20260922' >/dev/null 2>&1; then
    finish "EXECUTION_FAILED" 75 "another sampled-H evaluator process already exists"
fi

: > "${LOG}"
printf 'timestamp_utc,model,status,exit_code,peak_gpu_memory_mib,run_dir,validation_report\n' > "${MODEL_STATUS}"
printf '%s,ARCH_B_GLOBAL_LOCAL,COMPLETED_AND_PARITY_VERIFIED,0,NA,%s,%s\n' \
    "$(utc_now)" "${PROJECT}/experiments/evaluation/ARCH_B_GLOBAL_LOCAL_sampledH_seed20260818_spec20260921_retry1/final_seed20260818_n5000" "${SUPERVISOR}/validation_reports/ARCH_B_GLOBAL_LOCAL.json" >> "${MODEL_STATUS}"
printf '%s,ARCH_A_ORIGINAL,COMPLETED_AND_PARITY_VERIFIED,0,21086,%s,%s\n' \
    "$(utc_now)" "${RESULT_ROOT}/ARCH_A_ORIGINAL/final_seed20260818_n5000" "${SUPERVISOR}/validation_reports/ARCH_A_ORIGINAL.json" >> "${MODEL_STATUS}"
printf '%s,ARCH_C_RELATIVE_LENGTH_GATE,COMPLETED_AND_PARITY_VERIFIED,0,21139,%s,%s\n' \
    "$(utc_now)" "${RESULT_ROOT}/ARCH_C_RELATIVE_LENGTH_GATE/final_seed20260818_n5000" "${SUPERVISOR}/validation_reports/ARCH_C_RELATIVE_LENGTH_GATE.json" >> "${MODEL_STATUS}"
printf 'utc,local,model,check,free_mib,util_percent,compute_processes,eligible\n' > "${GPU_CSV}"
log "new immutable resume supervisor started; resume point LEGACY_B_AAM_ONLY; deadline $(date -d "@${DEADLINE_EPOCH}" '+%Y-%m-%dT%H:%M:%S%z')"

snapshot_ok() { (cd "${SNAPSHOT}" && sha256sum -c snapshot_sha256.txt) >> "${LOG}" 2>&1; }
run_preflight() {
    local model="$1" stage="$2" report="${SUPERVISOR}/preflight_${model}_${stage}.json"
    snapshot_ok || return 1
    CUDA_VISIBLE_DEVICES="" "${PYTHON}" "${SNAPSHOT}/preflight_resume.py" \
        --next-model "${model}" --report "${report}" --validation-dir "${SUPERVISOR}/validation_reports" >> "${LOG}" 2>&1
}
gpu_sample() {
    local model="$1" check="$2" gpu_line free util apps app_count=0 eligible=0
    gpu_line="$(nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | head -n 1)"
    free="$(cut -d, -f1 <<<"${gpu_line}" | tr -d ' ')"
    util="$(cut -d, -f2 <<<"${gpu_line}" | tr -d ' ')"
    apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ -n "$(tr -d '[:space:]' <<<"${apps}")" ]]; then app_count="$(wc -l <<<"${apps}" | tr -d ' ')"; fi
    if (( free >= 22000 && util <= 5 && app_count == 0 )); then eligible=1; fi
    printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "$(utc_now)" "$(local_now)" "${model}" "${check}" "${free}" "${util}" "${app_count}" "${eligible}" >> "${GPU_CSV}"
    log "GPU ${model} check=${check}: free=${free} MiB util=${util}% compute_processes=${app_count} eligible=${eligible}"
    [[ "${eligible}" == 1 ]]
}
wait_for_gpu() {
    local model="$1" consecutive=0 check=0
    write_status "WAITING_FOR_GPU" "waiting for two eligible checks for ${model}"
    while (( $(date +%s) < DEADLINE_EPOCH )); do
        check=$((check + 1))
        if gpu_sample "${model}" "${check}"; then consecutive=$((consecutive + 1)); else consecutive=0; fi
        if (( consecutive >= 2 )); then return 0; fi
        sleep 60
    done
    return 1
}
validate_one() {
    local model="$1" run_dir="$2" checkpoint_sha="$3" enrollment="$4" probe="$5" report="$6"
    "${PYTHON}" "${SNAPSHOT}/validate_model.py" --model "${model}" --run-dir "${run_dir}" \
        --canonical-dir "${PROJECT}/experiments/evaluation/architecture_A_viespeaker_h_1e1p/final_seed20260818_n5000" \
        --checkpoint-sha "${checkpoint_sha}" --enrollment-encoder "${enrollment}" --probe-encoder "${probe}" \
        --parity-checker "${SNAPSHOT}/check_audio_quality_semantic_parity.py" --report "${report}" >> "${LOG}" 2>&1
}
record_model() {
    printf '%s,%s,%s,%s,%s,%s,%s\n' "$(utc_now)" "$1" "$2" "$3" "$4" "$5" "$6" >> "${MODEL_STATUS}"
}
run_model() {
    local model="$1" evaluator="$2" checkpoint="$3" checkpoint_sha="$4" enrollment="$5" probe="$6" module="${7:-}"
    local output="${RESULT_ROOT}/${model}" run_dir="${RESULT_ROOT}/${model}/final_seed20260818_n5000"
    local model_log="${SUPERVISOR}/model_logs/${model}.log" report="${SUPERVISOR}/validation_reports/${model}.json"
    CURRENT_MODEL="${model}"
    if ! wait_for_gpu "${model}"; then finish "GPU_TIMEOUT" 124 "72-hour GPU deadline reached before ${model}"; fi
    write_status "PREFLIGHT" "revalidating frozen artifacts before ${model}"
    if ! run_preflight "${model}" "prelaunch"; then finish "EXECUTION_FAILED" 78 "scientific preflight failed before ${model}"; fi
    if [[ -e "${output}" ]]; then finish "EXECUTION_FAILED" 79 "non-overwrite output exists: ${output}"; fi
    if ! gpu_sample "${model}" "post_preflight_repeat"; then
        log "GPU was acquired during ${model} preflight; returning to the two-check wait"
        if ! wait_for_gpu "${model}"; then finish "GPU_TIMEOUT" 124 "72-hour GPU deadline reached before ${model}"; fi
        if ! gpu_sample "${model}" "post_wait_repeat"; then finish "EXECUTION_FAILED" 80 "GPU gate changed immediately before ${model}"; fi
    fi

    local base=(--mode final --repository "${REPOSITORY}" --checkpoint "${checkpoint}" --musan-dir "${MUSAN}" --pretrained-dir "${PRETRAINED}" --output-root "${output}" --cache-root "${CACHE}" --reference-protocol "${PROTOCOL}" --embedding-batch-size 32 --parquet-batch-size 256 --enrollment-seconds 2 3 4 --probe-seconds 1 1.25 1.5 --bootstrap-replicates 300 --seed 20260818)
    local command
    if [[ "${evaluator}" == "aam_legacy" ]]; then
        command=("${PYTHON}" "${SNAPSHOT}/run_named_aam_legacy.py" --pooling-variant original --model-name "${model}" --local-window-seconds 0.5 --local-overlap 0.5 "${base[@]}")
    else
        command=("${PYTHON}" "${SNAPSHOT}/run_named_legacy.py" "${module}" "${model}" -- "${base[@]}")
    fi
    log "launching ${model}: ${command[*]}"
    write_status "RUNNING" "running ${model}; CUDA required; CPU fallback prohibited"
    : > "${model_log}"
    local code=0 remaining=$(( DEADLINE_EPOCH - $(date +%s) )) evaluator_pid used peak=0
    CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM "${remaining}" "${command[@]}" >> "${model_log}" 2>&1 &
    evaluator_pid=$!
    log "${model} evaluator PID=${evaluator_pid}"
    while ps -p "${evaluator_pid}" -o pid= >/dev/null 2>&1; do
        used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ' || true)"
        if [[ "${used}" =~ ^[0-9]+$ ]] && (( used > peak )); then peak="${used}"; fi
        sleep 5
    done
    if wait "${evaluator_pid}"; then code=0; else code=$?; fi
    if (( code != 0 )); then
        record_model "${model}" EXECUTION_FAILED "${code}" "${peak}" "${run_dir}" "${report}"
        finish "EXECUTION_FAILED" "${code}" "${model} evaluator failed; partial output, if any, is preserved"
    fi
    if validate_one "${model}" "${run_dir}" "${checkpoint_sha}" "${enrollment}" "${probe}" "${report}"; then
        record_model "${model}" COMPLETED_AND_PARITY_VERIFIED 0 "${peak}" "${run_dir}" "${report}"
        log "${model} completed and exact ordered-row parity passed"
    else
        record_model "${model}" COMPLETED_BUT_PARITY_FAILED 1 "${peak}" "${run_dir}" "${report}"
        finish "COMPLETED_BUT_PARITY_FAILED" 82 "${model} failed exact-row or artifact validation; no row intersection performed"
    fi
}

run_model LEGACY_B_AAM_ONLY aam_legacy "${PROJECT}/experiments/mentor_stage/m0_aam_only_full_dataset/best_validation_eer.pt" 4ea92fd165853d3371f92d252f0993e76ca3c22f23a75d9affbf0d27931bf260 student student
run_model LEGACY_C_VECTOR_EMA legacy "${PROJECT}/experiments/mentor_stage/m2_legacy_vector_ema/best_validation_eer.pt" 672fba529dd17195a77fc1e0d9731cefb6e0172e12cc185db904eed2c4271d54 ema_teacher student evaluate_c_vector_ema_viespeaker_h_1e1p
run_model LEGACY_D_FULL_KL_EMA legacy "${PROJECT}/experiments/mentor_stage/m3_legacy_ema_gaussian_kl/best_validation_eer.pt" c01ed0a962426098610b77e5c8b89eb9d1b27a64adbd217765b9ee94637b5f41 ema_teacher student evaluate_d_full_viespeaker_h_1e1p

CURRENT_MODEL="final_analysis"
write_status "FINAL_ANALYSIS" "building six-system exact-row reports"
"${PYTHON}" "${SNAPSHOT}/finalize_results.py" >> "${LOG}" 2>&1
finish "COMPLETED_AND_PARITY_VERIFIED" 0 "resumed sequential evaluation and final paired analysis completed"
