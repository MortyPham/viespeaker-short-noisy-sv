"""Run whole seed blocks sequentially after priority/idle checks. Not auto-started."""
import os
import subprocess
import sys
from common import *
PY='/home/ml4u/.conda/envs/asr/bin/python'
if __name__=='__main__':
    check_priority_and_gpu()
    for seed in [2027,2028]:
        for mode in ['smoke','train','evaluate']:
            for model in ['B','C','D']:
                config=BASE/f'configs/{model}_seed{seed}.json'
                script=BASE/('evaluate_matched.py' if mode=='evaluate' else 'train_matched.py')
                command=[PY,str(script),'--config',str(config)]
                if mode!='evaluate':command+=['--mode',mode]
                with (BASE/f'logs/{mode}_{model}_seed{seed}.log').open('a') as log:
                    subprocess.run(command,cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2','CUDA_VISIBLE_DEVICES':'0'},stdout=log,stderr=subprocess.STDOUT,check=True)
                if mode=='evaluate' and model=='D':subprocess.run([PY,str(BASE/'report_results.py')],cwd=ROOT,check=True)
