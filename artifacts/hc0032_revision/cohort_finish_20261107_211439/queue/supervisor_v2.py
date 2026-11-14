"""Persistent one-child HC0032 queue; capacity-triggered qualification, then whole seeds."""
import datetime,fcntl,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
N=Path(__file__).resolve().parents[1];B=N/'matched';R=N.parents[2]
PY='/home/ml4u/.conda/envs/asr/bin/python';CPY='/home/ml4u/ENTER/bin/python'
ENV={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','CUDA_VISIBLE_DEVICES':'0','OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
Q=N/'queue';POLL=30;DEADLINE=time.time()+48*3600
minimum_free=8192
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def load(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def event(kind,**kw):
    row={'utc':now(),'supervisor_pid':os.getpid(),'event':kind,'minimum_free_mib':minimum_free,'deadline_utc':datetime.datetime.fromtimestamp(DEADLINE,datetime.timezone.utc).isoformat(),**kw}
    with (Q/'events.jsonl').open('a') as f:f.write(json.dumps(row,default=str)+'\n')
    tmp=Q/f'.status_{time.time_ns()}.tmp';tmp.write_text(json.dumps(row,indent=2,default=str));os.replace(tmp,Q/'status.json');print(json.dumps(row,default=str),flush=True)
def gpu():
    s=subprocess.run(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
    free,util=[int(x.strip()) for x in s.splitlines()[0].split(',')]
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
    return {'free_mib':free,'utilization_percent':util,'compute_processes':apps}
def capacity():
    while time.time()<DEADLINE:
        state=gpu();event('waiting_for_capacity',gpu=state,active_child=None,reason='Supported-extreme backward failed at3GiBallocatorcap; materially larger capacity required. Utilization is not a gate.')
        if state['free_mib']>=minimum_free:return state
        time.sleep(POLL)
    raise TimeoutError('New queue48hourcapacity deadline reached; no foreign job interrupted')
def verify_release():
    for manifest in ['TRAIN_RELEASE.json','QUEUE_RELEASE_v2.json']:
        for x in load(N/manifest):assert sha(x['path'])==x['sha256'],x['path']
def child(name,command):
    verify_release();path=N/'logs'/f'{name}_{time.time_ns()}.log';t=time.perf_counter()
    with path.open('x') as log:
        p=subprocess.Popen(command,cwd=R,env=ENV,stdout=log,stderr=subprocess.STDOUT)
        event('child_started',name=name,command=command,child_pid=p.pid,log=str(path),gpu=gpu())
        while p.poll() is None:
            event('child_heartbeat',name=name,child_pid=p.pid,log=str(path),gpu=gpu());time.sleep(POLL)
    event('child_exited',name=name,returncode=p.returncode,elapsed_seconds=time.perf_counter()-t,log=str(path),active_child=None)
    return p.returncode,path
def qualify(seed):
    global minimum_free
    pointer=Q/f'qualification_seed{seed}.json'
    if pointer.exists():
        folder=Path(load(pointer)['folder'])
        if load(folder/'PROFILE_PASS.json')['status']=='passed' and load(folder/'EQUIVALENCE_PASS.json')['seed']==seed:return folder
    while True:
        capacity();folder=N/'qualification'/f'seed{seed}_{time.time_ns()}'
        code,log=child(f'qualify_seed{seed}',[PY,str(B/'qualify_capacity.py'),'all','--seed',str(seed),'--out',str(folder)])
        if code==0:
            tmp=pointer.with_suffix(f'.{time.time_ns()}.tmp');tmp.write_text(json.dumps({'folder':str(folder),'seed':seed,'source_sha256':sha(B/'engine_v5.py')},indent=2));os.replace(tmp,pointer)
            return folder
        failures=sorted(folder.glob('failure_*.json'))
        failure=load(failures[-1]) if failures else {}
        if failure.get('status')!='OOM':raise RuntimeError(f'Qualification failed scientifically/operationally; not an automatic memory retry: {log}')
        minimum_free=12288 if minimum_free<12288 else 22000 if minimum_free<22000 else 999999
        if minimum_free>24564:raise RuntimeError('Supported recipe still failed near full4090capacity; no further automatic allocation attempts')
        event('capacity_requirement_increased',failure=failure,reason='No repeat at unchanged capacity; new bounded qualification required before training')
def raise_capacity_after_training_failure(error,seed,model):
    global minimum_free
    minimum_free=12288 if minimum_free<12288 else 22000 if minimum_free<22000 else 999999
    if minimum_free>24564:raise RuntimeError('No adequate single4090trainingcapacity after resource retries')
    event('training_resource_retry_wait',seed=seed,model=model,failure=error,reason='Only resume completed immutable checkpoints after material capacity increase; no unchanged-capacity retry')
def analysis():
    paths=[];complete=[]
    for seed in [2027,2028]:
        members=[B/f'evaluation/seed{seed}/{m}/scores_long.csv' for m in 'BCD']
        if all((p.parent/'completion.json').exists() for p in members):paths+=members;complete.append(seed)
    if not paths:return
    out=N/'matched_statistics'/f'blocks{len(complete)}_{time.time_ns()}'
    source=R/'artifacts/hc0032_revision/overnight_20261106/prompt4_historical/paired_analysis.py'
    code,log=child('paired_statistics',[CPY,str(source),'--scores-long',*map(str,paths),'--output-dir',str(out),'--cohort','matched_seeds','--aggregate-replicates','5000','--cell-replicates','1000'])
    if code:raise RuntimeError(f'Matched statistics failed:{log}')
def main():
    event('STARTED',command=[sys.executable,*sys.argv],policy='One GPU child at a time; legacy execution lock in every GPU child; no global B historical replay gate; no foreign process signals')
    for seed in [2027,2028]:
        qualification=qualify(seed)
        remaining=sum(not (B/f'runs/seed{s}/{m}/completion.json').exists() for s in [2027,2028] for m in 'BCD')
        assert shutil.disk_usage(N).free>(remaining*4+2)*1024**3,'Insufficient remaining cohort checkpoint allowance'
        for model in 'BCD':
            config=B/f'configs_v5/{model}_seed{seed}.json'
            if (B/f'runs/seed{seed}/{model}/completion.json').exists():continue
            while True:
                capacity()
                code,log=child(f'train_{model}_seed{seed}',[PY,str(B/'train_fresh.py'),'--config',str(config),'--qualification',str(qualification)])
                if code==0:break
                errors=sorted((B/'logs').glob('train_failure_*.json'))
                error=load(errors[-1]) if errors else {}
                recoverable=error.get('status')=='OOM' or 'Physical headroom lost' in error.get('error','')
                if not recoverable:raise RuntimeError(f'Training stopped on non-resource failure; preserve and investigate:{log}')
                # Never reuse a partially failed state; child resumes only immutable
                # successful-step checkpoints with all optimizer/RNG/EMA state.
                raise_capacity_after_training_failure(error,seed,model)
        for model in 'BCD':
            if (B/f'evaluation/seed{seed}/{model}/completion.json').exists():continue
            capacity();code,log=child(f'evaluate_{model}_seed{seed}',[PY,str(B/'evaluate_fresh.py'),'--config',str(B/f'configs_v5/{model}_seed{seed}.json')])
            if code:raise RuntimeError(f'Evaluation failed:{log}')
        analysis();child('consolidated_report',[PY,str(N/'cohort_report.py')]);event('COMPLETE_SEED_BLOCK',seed=seed)
    event('COMPLETED',complete_matched_blocks=2)
if __name__=='__main__':
    with (Q/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:main()
        except TimeoutError as error:event('EXPIRED',error=str(error),active_child=None)
        except BaseException as error:event('STOPPED',error=repr(error),traceback=traceback.format_exc(),active_child=None);raise
