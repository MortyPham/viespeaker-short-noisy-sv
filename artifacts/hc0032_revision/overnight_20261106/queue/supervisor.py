"""Detached dependency-aware HC0032 queue; never trains or interrupts other jobs.

Adapts the server's existing idle-GPU watchdog policy (22,000 MiB, <=5%,
zero compute processes) to this closed objective-model queue. Status files are
mutable operational state; events and scientific outputs remain append-only.
"""
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
RUN=Path(__file__).resolve().parents[1]
Q=RUN/'queue'
PY='/home/ml4u/.conda/envs/asr/bin/python'
CPU='/home/ml4u/ENTER/bin/python'
WAIT_SECONDS=12*3600
POLL=60
MIN_FREE=22000
MAX_UTIL=5
started=time.time()
deadline=started+WAIT_SECONDS
children={}
stage='initializing'
last_error=None
last_gpu=None
ENV={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','HF_HUB_OFFLINE':'1','CUDA_VISIBLE_DEVICES':'0'}
def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def event(kind,**data):
    with (Q/'events.jsonl').open('a') as f: f.write(json.dumps({'time_utc':now(),'pid':os.getpid(),'event':kind,**data},default=str)+'\n')
def training():
    p=RUN/'initialization_audit/decision.json'
    return read(p) if p.exists() else {'training_allowed':False,'reason':'Initialization/recipe audit not finished'}
def task_state():
    p4=RUN/'prompt4_historical/results/run_metadata.json'
    tasks={
        'prompt1_raw_replay':'completed' if (RUN/'prompt1_replay/replay_pass.json').exists() else 'waiting_for_idle_GPU',
        'prompt2_historical_student_enrollment':'completed' if (RUN/'prompt2/completed.json').exists() else 'queued_after_replay_pass',
        'prompt3_matched_training':{'state':'blocked','decision':str(RUN/'initialization_audit/decision.json'),'training_allowed':False,'reason':'Exact initialization and matched recipe not verified; historical selector and augmentation conflicts'},
        'prompt4_historical':'completed' if p4.exists() else 'running_or_pending',
        'prompt4_two_way_sensitivity':'completed' if (RUN/'prompt4_historical/two_way_sensitivity/run_metadata.json').exists() else 'running_or_pending',
        'prompt4_role_comparisons':'completed' if (RUN/'prompt4_roles/results/run_metadata.json').exists() else 'running' if 'p4_roles' in children else 'queued_after_prompt2',
        'prompt4_matched_seeds':'blocked_on_prompt3',
        'prompt5_calibration_inputs':'prepared' if (RUN/'prompt5/prepared/calibration_config.json').exists() else 'preparing',
        'prompt5_primary_operating_point':'completed' if (RUN/'prompt5/run_primary_B/completion.json').exists() else 'queued_after_prompt2',
        'manuscript_handoff':'prepared_with_dynamic_status' if (RUN/'REVIEWER_RESPONSE.md').exists() else 'preparing'}
    if stage in ['prompt1_cuda_smoke','prompt1_raw_replay']:tasks['prompt1_raw_replay']='running'
    if stage=='prompt2_roles':tasks['prompt2_historical_student_enrollment']='running'
    if stage=='prompt5_primary':tasks['prompt5_primary_operating_point']='running'
    if stage in ['stopped_dependent_GPU_work_on_failure','idle_GPU_wait_timeout']:
        for key in ['prompt1_raw_replay','prompt2_historical_student_enrollment','prompt5_primary_operating_point','prompt4_role_comparisons']:
            if tasks[key]!='completed' and tasks[key]!='running':tasks[key]={'state':'blocked','reason':last_error}
    return tasks
def status():
    data={'time_utc':now(),'supervisor_pid':os.getpid(),'stage':stage,'last_error':last_error,
        'deadline_utc':datetime.fromtimestamp(deadline,timezone.utc).isoformat(),
        'policy':{'free_mib_at_least':MIN_FREE,'utilization_at_most_percent':MAX_UTIL,'compute_processes':0,'consecutive_idle_polls':3,'poll_seconds':POLL},
        'tasks':task_state(),'gpu':last_gpu,'active_children':{k:{'pid':p.pid,'returncode':p.poll()} for k,p in children.items()},
        'training_allowed':False,'noise_policy':'Known overlap retained; no historical noise realization changed'}
    temp=Q/'status.json.tmp'
    temp.write_text(json.dumps(data,indent=2,default=str)+'\n'); os.replace(temp,Q/'status.json')
    return data
def transition(name,**data):
    global stage
    stage=name; event('stage',stage=name,**data); status()
def handoff():
    argv=[CPU,str(Q/'render_handoff.py')]
    p=subprocess.run(argv,cwd=ROOT,env=ENV,capture_output=True,text=True)
    event('manuscript_handoff',command=argv,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
def gpu():
    a=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    b=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    row=a.stdout.splitlines()[0].split(',')
    processes=[line.strip() for line in b.stdout.splitlines() if line.strip()]
    return {'index':int(row[0]),'free_mib':int(row[1]),'utilization_percent':int(row[2]),'compute_processes':processes,
        'eligible':int(row[1])>=MIN_FREE and int(row[2])<=MAX_UTIL and not processes}
def verify_release():
    for record in read(Q/'code_release.json'):
        assert sha(record['path'])==record['sha256'],f'Queue code changed: {record["path"]}'
def launch(name,argv):
    verify_release()
    event('command',name=name,argv=argv,cwd=str(ROOT),environment_overrides={k:ENV[k] for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','HF_HUB_OFFLINE','PYTHONDONTWRITEBYTECODE','CUDA_VISIBLE_DEVICES']})
    logfile=(Q/f'{name}.log').open('a')
    p=subprocess.Popen(argv,cwd=ROOT,env=ENV,stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True)
    logfile.close(); children[name]=p; event('started',name=name,child_pid=p.pid); status(); return p
def wait_child(name,fail=True):
    p=children[name]
    while p.poll() is None:
        status(); time.sleep(5)
    event('exited',name=name,child_pid=p.pid,returncode=p.returncode)
    if fail and p.returncode: raise RuntimeError(f'{name} failed with exit code {p.returncode}; see {Q/name}.log')
    return p.returncode
def wait_idle():
    global last_gpu
    stable=0
    while time.time()<deadline:
        last_gpu=gpu(); event('gpu_check',**last_gpu); status()
        stable=stable+1 if last_gpu['eligible'] else 0
        if stable>=3:return
        time.sleep(POLL)
    raise TimeoutError('Overnight idle-GPU deadline reached; existing compute jobs were not interrupted')
def gpu_stage(name,argv):
    global last_gpu
    transition('waiting_for_idle_GPU',next_job=name)
    lockpath=ROOT/'experiments/evaluation/watchdogs/.locks/arch_b_gpu_watchdog_20260921.lock'
    while True:
        wait_idle()
        # Release this lock before any further wait. Existing jobs are never signaled.
        with lockpath.open('a') as lock:
            try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                event('legacy_watchdog_lock_busy'); continue
            last_gpu=gpu()
            if not last_gpu['eligible']:
                event('GPU_changed_before_launch',**last_gpu); continue
            transition(name,argv=argv)
            launch(name,argv); wait_child(name)
            return
def run():
    global last_error
    with (Q/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        event('supervisor_started',argv=sys.argv,deadline_utc=datetime.fromtimestamp(deadline,timezone.utc).isoformat())
        verify_release()
        transition('checking_frozen_inputs')
        if not (Q/'frozen_inputs.json').exists():
            raise RuntimeError('Historical input hashing must complete before queue launch')
        # Cheap metadata revalidation here; workers hash audio/noise as each file is consumed.
        for r in read(Q/'frozen_inputs.json')['inputs']:
            p=Path(r['path']); s=p.stat()
            assert s.st_size==r['bytes'] and s.st_mtime_ns==r['mtime_ns'],f'Frozen input changed: {p}'
        assert read(Q/'waveform_smoke.json')['status']=='passed'
        assert training().get('training_allowed') is False,'This queue has no verified training implementation; never auto-enable training'
        if not (RUN/'prompt1_replay/smoke_pass.json').exists():
            gpu_stage('prompt1_cuda_smoke',[PY,str(Q/'gpu_stages.py'),'smoke'])
        if not (RUN/'prompt1_replay/replay_pass.json').exists():
            gpu_stage('prompt1_raw_replay',[PY,str(Q/'gpu_stages.py'),'replay'])
        assert read(RUN/'prompt1_replay/replay_pass.json')['status']=='passed'
        if not (RUN/'prompt2/completed.json').exists():
            gpu_stage('prompt2_roles',[PY,str(Q/'gpu_stages.py'),'roles'])
        # CPU role analyses can overlap the subsequent primary operating-point GPU work.
        if not (RUN/'prompt4_roles/results/run_metadata.json').exists():
            script=RUN/'prompt4_historical/paired_analysis.py'
            out=RUN/'prompt4_roles/results'
            argv=[CPU,str(script),'--scores-long',str(RUN/'prompt2/scores_long.csv'),'--output-dir',str(out),
                '--cohort','role_matched_historical','--aggregate-replicates','5000','--cell-replicates','1000']
            if out.exists():argv.append('--resume')
            launch('p4_roles',argv)
        if not (RUN/'prompt5/run_primary_B/completion.json').exists():
            gpu_stage('prompt5_primary',[PY,str(RUN/'prompt5/calibration.py'),'run','--device','cuda','--batch-size','8','--bootstrap-replicates','1000'])
        if 'p4_roles' in children:wait_child('p4_roles')
        transition('independent_queue_completed_training_blocked')
        handoff()
if __name__=='__main__':
    try:run()
    except TimeoutError as e:
        last_error=str(e);transition('idle_GPU_wait_timeout');handoff();sys.exit(0)
    except Exception as e:
        last_error=repr(e);event('failure',error=repr(e),traceback=traceback.format_exc())
        transition('stopped_dependent_GPU_work_on_failure')
        handoff()
        # Independent already-running CPU work is intentionally left running.
        traceback.print_exc();sys.exit(1)
