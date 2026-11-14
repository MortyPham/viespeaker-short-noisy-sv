"""One-shot resource-coordinated stage runner. No background waiting by default."""
import argparse, datetime, fcntl, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
R=Path('/media/ml4u/Samsung_T52/ptnk_team/mortyhuaphuoc')
N=Path(__file__).resolve().parents[1]
PY='/home/ml4u/.conda/envs/asr/bin/python'
ENV={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1',
     'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
def load(p): return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def event(kind,**kw):
    item={'utc':stamp(),'pid':os.getpid(),'event':kind,**kw}
    with (N/'queue/events.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
    temp=N/'queue'/f'.status_{time.time_ns()}.tmp'
    with temp.open('x') as f:json.dump(item,f,indent=2)
    os.replace(temp,N/'queue/status.json')
    print(json.dumps(item),flush=True)
def ownership():
    gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,capture_output=True,check=True).stdout
    active=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True,capture_output=True,check=True).stdout.strip()
    free,util=[int(x.strip()) for x in gpu.splitlines()[0].split(',')]
    processes=subprocess.run(['ps','-eo','pid,user,args'],text=True,capture_output=True,check=True).stdout
    # Compare executable tokens; ignore shell command strings and this coordinator.
    duplicates=[]
    for row in processes.splitlines()[1:]:
        cols=row.split(None,2)
        if len(cols)<3 or int(cols[0])==os.getpid():continue
        command=cols[2].split()
        if command and ('python' in Path(command[0]).name or Path(command[0]).name=='bash'):
            if any(Path(x).name in {'gpu_stages.py','train_matched.py','evaluate_matched.py','calibration.py','supervisor.py','execute_plan.py','run_stage.py'} for x in command[1:]):duplicates.append(row)
    event('ownership_check',free_mib=free,utilization_percent=util,compute_processes=active,duplicates=duplicates,
          policy='No concurrent use without established ownership. Idle-only conservative fallback remains unmeasured, not a task memory requirement.')
    if active or duplicates:raise RuntimeError('No authorized idle slot: live compute owner or duplicate research runner; no concurrent-use agreement established')
    if free<22000 or util>5:raise RuntimeError('Conservative idle scheduling fallback not met; task-specific memory remains unmeasured')
    return free,util
def verified_gate(path):
    assert load(path)['status']=='passed',f'Unmet gate: {path}'
def stage(name):
    ownership()
    for info in load(N/'queue/code_release.json'):
        assert sha(info['path'])==info['sha256'],info['path']
    if name=='preflight':return
    if name=='smoke':command=[PY,str(N/'cuda_smoke/smoke.py')]
    elif name=='replay':
        verified_gate(N/'prompt1_replay/smoke_pass.json')
        command=[PY,str(N/'queue/gpu_stages.py'),'replay']
    elif name in ['roles-primary','roles']:
        verified_gate(N/'prompt1_replay/replay_pass.json')
        command=[PY,str(N/'queue/gpu_stages.py'),name]
    elif name=='calibration':
        verified_gate(N/'prompt1_replay/replay_pass.json')
        command=[PY,str(N/'prompt5/calibration.py'),'run','--device','cuda','--batch-size','8',
                 '--bootstrap-replicates','1000','--prepared',str(R/'artifacts/hc0032_revision/overnight_20261106/prompt5/prepared'),
                 '--run-dir',str(N/'prompt5/run_primary_B')]
    elif name=='training':
        verified_gate(N/'prompt1_replay/replay_pass.json')
        assert (N/'prompt2/completed.json').exists(), 'Complete cheaper role deliverables before six training jobs'
        assert (N/'prompt5/run_primary_B/completion.json').exists(), 'Complete small calibration before training'
        assert shutil.disk_usage(N).free>=25*1024**3,'Recheck storage; training planning allowance25GiB'
        command=[PY,str(N/'matched/execute_plan.py')]
    else:raise ValueError(name)
    logfile=N/'logs'/f'{name}_{time.time_ns()}.log'
    # Training children acquire the established execution lock individually.
    lock=None
    if name!='training':
        lock=(R/'experiments/evaluation/watchdogs/.locks/arch_b_gpu_watchdog_20260921.lock').open('a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ownership()
    try:
        with logfile.open('x') as log:
            event('started',stage=name,command=command,environment={k:ENV[k] for k in ['PYTHONDONTWRITEBYTECODE','HF_HUB_OFFLINE','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},log=str(logfile))
            child=subprocess.Popen(command,cwd=R,env=ENV,stdout=log,stderr=subprocess.STDOUT)
            while True:
                try:code=child.wait(timeout=30);break
                except subprocess.TimeoutExpired:event('heartbeat',stage=name,child_pid=child.pid,log=str(logfile))
        if code:raise RuntimeError(f'{name} failed with return code {code}; see {logfile}. Dependent work stopped.')
        event('completed',stage=name,returncode=code,log=str(logfile))
    finally:
        if lock:lock.close()
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['preflight','smoke','replay','roles-primary','calibration','roles','historical','training'])
    args=parser.parse_args()
    # This new lock prevents duplicate coordinators without changing legacy ownership.
    with (N/'queue/coordinator.lock').open('a') as coordinator:
        fcntl.flock(coordinator,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for task in (['smoke','replay','roles-primary','calibration','roles'] if args.stage=='historical' else [args.stage]):stage(task)
        except Exception as error:
            event('blocked_or_failed',stage=args.stage,error=str(error))
            raise SystemExit(75)
