"""Frozen historical components and metadata helpers for new controlled blocks."""
import hashlib
import json
import os
import sys
from pathlib import Path
ROOT=Path('/media/ml4u/Samsung_T52/ptnk_team/mortyhuaphuoc')
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'source'))
import train_viespeaker_d_full as hist
import torch

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(path):return json.loads(Path(path).read_text())
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,default=str)
def record(path):return {'path':str(Path(path).absolute()),'sha256':sha(path)}
def tensor_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        x=v.detach().cpu().contiguous();h.update(k.encode());h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode());h.update(x.numpy().tobytes())
    return h.hexdigest()
def verify(info):assert sha(info['path'])==info['sha256'],info['path']
def args_from(config):
    a=hist.parse_args(['--project-root',str(ROOT),'--recording-manifest',str(ROOT/'experiments/shared/d_full_audio_cache/recordings.csv'),
        '--allow-incomplete-speakers','--wandb-mode','disabled'])
    for k,v in config['training'].items():setattr(a,k,v)
    a.model_cache=Path(config['pretrained_bundle']);a.output_dir=Path(config['output_dir'])
    return a
def data(a):
    records=hist.load_recording_manifest(a.recording_manifest,a.cache_dir)
    split=hist.load_speaker_splits(a.speaker_splits)
    selected,train,val=hist.select_eligible_records(records,split,a)
    assert len(train)==3371 and len(val)==90
    return selected,train,val
def batch_hash(batch):
    names=['teacher_audio','student_audio','teacher_lengths','student_lengths','labels']
    payload=tensor_hash({k:batch[k] for k in names})
    meta=[{k:v for k,v in item.items() if k not in ['teacher_audio','student_audio']} for item in batch['metadata']]
    return hashlib.sha256((payload+json.dumps(meta,sort_keys=True)).encode()).hexdigest()
def check_priority_and_gpu():
    import subprocess
    permission=load(BASE.parent/'CONCURRENCY_PERMISSION.json')
    assert permission['user_authorized_bounded_attempt'] is True
    g=subprocess.run(['nvidia-smi','--query-gpu=memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    total,free,util=[int(x.strip()) for x in g.stdout.splitlines()[0].split(',')]
    reserve=load(BASE.parent/'RESOURCE_POLICY.json')['headroom_mib']
    assert free-reserve>=512,'Insufficient available capacity for bounded attempt'
    torch.cuda.set_per_process_memory_fraction((free-reserve)/total,0)
    return {'free_mib':free,'utilization_percent':util,'allocator_budget_mib':free-reserve,'headroom_mib':reserve,'utilization_is_not_a_gate':True}
