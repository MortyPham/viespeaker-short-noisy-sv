"""Fixed-capacity HC0032 cohort supervisor: one GPU child; no idle-GPU gate."""
import os,sys,json,datetime,time,subprocess,hashlib,fcntl,shutil,traceback
from pathlib import Path
N=Path(__file__).resolve().parents[1];R=N.parents[2];PY='/home/ml4u/.conda/envs/asr/bin/python';CPY='/home/ml4u/ENTER/bin/python'
ENV={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','CUDA_VISIBLE_DEVICES':'0','OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'}
Q=N/'queue';Q.mkdir(exist_ok=True);DEADLINE=time.time()+48*3600

def load(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def event(kind,**kw):
 row={'utc':now(),'pid':os.getpid(),'event':kind,**kw}
 with (Q/'events.jsonl').open('a') as f:f.write(json.dumps(row,default=str)+'\n')
 tmp=Q/f'.status_{time.time_ns()}.tmp';tmp.write_text(json.dumps(row,indent=2,default=str));os.replace(tmp,Q/'status.json');print(json.dumps(row,default=str),flush=True)
def gpu():
 p=subprocess.run(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True);free,util=[int(x.strip()) for x in p.stdout.splitlines()[0].split(',')]
 apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
 return {'free_mib':free,'utilization_percent':util,'processes':apps}
def verify_release():
 for x in load(N/'RELEASE.json'):assert sha(x['path'])==x['sha256'],x['path']
def capacity():
 while time.time()<DEADLINE:
  g=gpu()
  if g['free_mib']>=4600:return g
  event('waiting_for_original_fixed_slot',gpu=g,required_free_mib=4600,active_child=None);time.sleep(30)
 raise TimeoutError('48h queue deadline; foreign processes unchanged')
def child(name,cmd,gpu_work=True):
 verify_release()
 if gpu_work:capacity()
 log=N/'logs'/f'{name}_{time.time_ns()}.log';began=time.perf_counter()
 with log.open('x') as f:
  p=subprocess.Popen(cmd,cwd=R,env=ENV,stdout=f,stderr=subprocess.STDOUT)
  event('child_started',name=name,command=cmd,child_pid=p.pid,log=str(log),gpu=gpu(),training_active=name.startswith('train_'))
  monitor=None
  if gpu_work:
   monitor=subprocess.Popen([CPY,str(N/'scripts/monitor.py'),str(p.pid),str(N/'logs'/f'{name}_physical_{time.time_ns()}.jsonl')],cwd=R,env=ENV,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  while p.poll() is None:
   event('child_heartbeat',name=name,child_pid=p.pid,log=str(log),gpu=gpu(),training_active=name.startswith('train_'));time.sleep(10)
  if monitor is not None:
   try:monitor.wait(timeout=5)
   except subprocess.TimeoutExpired:monitor.terminate();monitor.wait(timeout=5)
 event('child_exited',name=name,returncode=p.returncode,elapsed_seconds=time.perf_counter()-began,log=str(log),active_child=None)
 if p.returncode:raise RuntimeError(f'{name} failed; preserve successful checkpoints and inspect {log}; do not repeat allocations at unchanged capacity')
 return log
def statistics():
 seeds=[s for s in [2027,2028] if all((N/f'evaluation/seed{s}/{m}/completion.json').exists() for m in 'BCD')]
 if not seeds:return
 scores=[str(N/f'evaluation/seed{s}/{m}/scores_long.csv') for s in seeds for m in 'BCD'];out=N/'matched_statistics'/f'blocks{len(seeds)}_{time.time_ns()}'
 child('paired_statistics',[CPY,str(R/'artifacts/hc0032_revision/overnight_20261106/prompt4_historical/paired_analysis.py'),'--scores-long',*scores,'--output-dir',str(out),'--cohort','matched_seeds','--aggregate-replicates','5000','--cell-replicates','1000'],False)
def main():
 verify_release();event('STARTED',command=[sys.executable,*sys.argv],policy='Fixed capacity; full32; lossless CPU state staging; one CUDA child; no utilization gate',deadline_utc=datetime.datetime.fromtimestamp(DEADLINE,datetime.timezone.utc).isoformat())
 for seed in [2027,2028]:
  qualification=N/f'qualification_staging_{seed}'
  if not (qualification/'PROFILE_PASS.json').exists():child(f'qualify_seed{seed}',[PY,str(N/'scripts/qualify_fixed.py'),'--out',str(qualification),'--seed',str(seed)])
  assert load(qualification/'PROFILE_PASS.json')['status']=='passed' and load(qualification/'EQUIVALENCE_PASS.json')['seed']==seed
  assert load(N/'checkpoint_validation_D/result.json')['passed']
  remaining=sum(not (N/f'runs/seed{s}/{m}/completion.json').exists() for s in [2027,2028] for m in 'BCD')
  assert shutil.disk_usage(N).free>(remaining*4+2)*1024**3,'Insufficient immutable checkpoint allowance'
  for model in 'BCD':
   if (N/f'runs/seed{seed}/{model}/completion.json').exists():continue
   child(f'train_{model}_seed{seed}',[PY,str(N/'scripts/train_fixed.py'),'--config',str(N/f'configs/{model}_seed{seed}.json'),'--qualification',str(qualification)])
  for model in 'BCD':
   if (N/f'evaluation/seed{seed}/{model}/completion.json').exists():continue
   child(f'evaluate_{model}_seed{seed}',[PY,str(N/'scripts/evaluate_fixed.py'),'--config',str(N/f'configs/{model}_seed{seed}.json')])
  statistics();child('consolidated_report',[PY,str(N/'cohort_report.py')],False);event('COMPLETE_SEED_BLOCK',seed=seed)
 event('COMPLETED',complete_seed_blocks=2)
if __name__=='__main__':
 with (Q/'coordinator.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:main()
  except BaseException as error:event('STOPPED',error=repr(error),traceback=traceback.format_exc(),active_child=None);raise
