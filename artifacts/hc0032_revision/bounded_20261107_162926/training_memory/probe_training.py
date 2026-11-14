"""Sequential resource smoke only; never launch full training or retry failures."""
import datetime,json,os,subprocess,time
from pathlib import Path
N=Path(__file__).resolve().parent
B=N/'matched'
PY='/home/ml4u/.conda/envs/asr/bin/python'
started=time.perf_counter()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def gpu():
    return subprocess.run(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip()
def save(name,value):
    with (N/name).open('x') as f:json.dump(value,f,indent=2)
results=[]
for model in ['B','C','D']:
    command=[PY,str(B/'train_matched.py'),'--config',str(B/f'configs/{model}_seed2027.json'),'--mode','smoke']
    before=gpu();t=time.perf_counter();log=N/f'{model}_smoke.log'
    with log.open('x') as f:
        child=subprocess.Popen(command,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','CUDA_VISIBLE_DEVICES':'0','OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},stdout=f,stderr=subprocess.STDOUT)
        print(json.dumps({'utc':now(),'event':'STARTED','model':model,'pid':child.pid,'gpu':before,'command':command}),flush=True)
        while True:
            try:code=child.wait(timeout=30);break
            except subprocess.TimeoutExpired:print(json.dumps({'utc':now(),'event':'HEARTBEAT','model':model,'pid':child.pid,'gpu':gpu()}),flush=True)
    result={'model':model,'seed':2027,'returncode':code,'elapsed_seconds':time.perf_counter()-t,'gpu_before':before,'gpu_after':gpu(),'command':command,'log':str(log)}
    resultfile=B/f'smoke/seed2027/{model}/smoke_pass.json'
    if resultfile.exists():result['smoke']=json.loads(resultfile.read_text())
    else:
        failures=sorted((B/'logs').glob(f'failure_{model}_2027_*.json'))
        result['failures']=[json.loads(p.read_text()) for p in failures]
    results.append(result);save(f'{model}_result.json',result)
    print(json.dumps({'utc':now(),'event':'FINISHED','model':model,'returncode':code,'elapsed_seconds':result['elapsed_seconds']}),flush=True)
    if code:break
save('SUMMARY.json',{'status':'all_smokes_passed' if len(results)==3 and all(r['returncode']==0 for r in results) else 'stopped_on_first_failure',
    'elapsed_seconds':time.perf_counter()-started,'models_attempted':[r['model'] for r in results],'seed':2027,'results':results,'full_training_started':False,'recipe_changed':False,'retry_attempts':0})
