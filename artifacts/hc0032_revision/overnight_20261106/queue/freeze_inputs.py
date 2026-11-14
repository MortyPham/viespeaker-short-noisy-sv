"""Hash immutable historical replay/role inputs before allowing GPU launches."""
import csv
import hashlib
import json
import os
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
RUN=Path(__file__).resolve().parents[1]
OLD=ROOT/'artifacts/hc0032_revision/prompt1_20261106_2153'
REG=json.loads((OLD/'final/model_registry.json').read_text())
Q=RUN/'queue'
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def event(**data):
    with (Q/'input_hash_progress.jsonl').open('a') as f: f.write(json.dumps({'time':time.time(),'pid':os.getpid(),**data})+'\n')
def relocate(s): return str(s).replace('/media/ml4u/Samsung_T51/ptnk_team/mortyhuaphuoc',str(ROOT))
paths={}; expected={}
def add(p,kind,hash_value=None):
    paths[str(p)]=kind
    if hash_value: expected[str(p)]=hash_value
for model,cfg in REG['models'].items():
    for k in ['score_file','trial_manifest']:
        add(cfg[k]['path'],k,cfg[k]['sha256'])
    if model!='A': add(cfg['checkpoint']['path'],'checkpoint',cfg['checkpoint']['sha256'])
    else:
        for p in cfg['pretrained_bundle']['files']: add(p['path'],'pretrained',p['sha256'])
    folder=Path(cfg['score_file']['path']).parent
    for name in ['audio_quality.csv','probe_transform_manifest.csv','run_metadata.json','embedding_cache_index.json','protocol_config.json']:
        add(folder/name,'historical_metadata')
    cache=next(iter(cfg['embedding_cache'].values()))['current_path'].split('/embeddings/')[0]
    with (folder/'audio_quality.csv').open() as f:
        for row in csv.DictReader(f): add(Path(cache)/'prepared_audio'/Path(row['path']).with_suffix('.npz'),'prepared_audio')
    for values in cfg['embedding_cache'].values():
        for p in sorted(Path(values['current_path']).glob('*.npz')): add(p,'historical_embedding')
for row in json.loads((OLD/'final/historical_probe_noise_audit.json').read_text())['files']:
    add(row['path'],'noise_file',row['sha256'])
for row in json.loads((OLD/'original_runtime/cached_embedding_check.json').read_text())['input_shards']:
    add(row['path'],'historical_embedding',row['sha256'])
for row in json.loads((Q/'source_snapshot.json').read_text()): add(row['path'],'source',row['sha256'])
for name in ['model_registry.json','protocol_registry.json']:
    add(OLD/'final'/name,'verified_registry')
add(ROOT/'experiments/shared/speaker_splits.csv','training_split','94ffd8724b779ebcee2d361eedaac78f78ab58f1338ddc6be8c9c9afceb6456a')
add(ROOT/'experiments/shared/d_full_audio_cache/recordings.csv','V1_recordings','6465af5727b64f35a164a6e12b59466e30faa71c570f57103f2de8b291638344')
event(status='running',inputs=len(paths))
records=[]
for i,(s,kind) in enumerate(sorted(paths.items())):
    p=Path(s); stat=p.stat(); h=sha(p)
    assert s not in expected or expected[s]==h,f'Historical input changed: {s}'
    records.append({'path':s,'kind':kind,'sha256':h,'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns})
    if (i+1)%250==0: event(status='running',hashed=i+1,total=len(paths))
with (Q/'frozen_inputs.json').open('x') as f:
    json.dump({'status':'verified','time':time.time(),'known_noise_overlap_preserved':True,'inputs':records},f,indent=2)
event(status='completed',inputs=len(records))
print('FROZEN',len(records),'historical inputs',flush=True)
