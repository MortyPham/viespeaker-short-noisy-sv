"""Re-score preserved primary-condition embedding caches, without model inference."""
import ast
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent
registry = json.loads((OUT/'model_registry.json').read_text())
if len(sys.argv) == 2:
    OUT = OUT / sys.argv[1]
    OUT.mkdir(exist_ok=False)
tree=ast.parse((ROOT/'evaluate_aam_only_full_viespeaker_h_1e1p.py').read_text())
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='verification_metrics')
scope=dict(np=np,roc_curve=roc_curve,roc_auc_score=roc_auc_score)
exec(compile(ast.Module(body=[node],type_ignores=[]),'historical_metric','exec'),scope)
inputs=[]
def load(folder):
    result={}
    for shard in sorted(Path(folder).glob('*.npz')):
        with shard.open('rb') as f: h=hashlib.file_digest(f,'sha256').hexdigest()
        inputs.append({'path':str(shard),'sha256':h})
        with np.load(shard,allow_pickle=False) as s:
            assert np.isfinite(s['embeddings']).all()
            assert np.max(np.abs(np.linalg.norm(s['embeddings'],axis=1)-1)) < 1e-5
            for k,v in zip(s['keys'].astype(str),s['embeddings'],strict=True):
                assert k not in result
                result[k]=v.copy()
    assert result
    return result
raw=[]; report=[]
for model,cfg in registry['models'].items():
    cache=cfg['embedding_cache']
    enroll=load(cache['enrollment_3s']['current_path'])
    frame=pd.read_csv(cfg['score_file']['path'],float_precision='round_trip')
    for noise in ['clean','10dB','0dB']:
        condition=f'enroll3s_probe1.25s_{noise}'
        g=frame.loc[frame.condition==condition].copy()
        probes=load(cache[f'1.25s_{noise}']['current_path'])
        scores=np.array([float(np.dot(enroll[e],probes[p])) for e,p in g[['enrollment','probe']].itertuples(index=False,name=None)])
        delta=np.abs(scores-g.score.to_numpy())
        g['paper_id']=model; g['cached_embedding_score']=scores
        raw.append(g)
        report.append({'model':model,'condition':condition,'rows':len(g),'target':int(g.label.sum()),
            'nontarget':int((g.label==0).sum()),'max_absolute_score_delta':float(delta.max()),
            'score_tolerance':1e-7,'within_tolerance':bool((delta<=1e-7).all()),
            **scope['verification_metrics'](g.label.to_numpy(),scores)})
        print(model,condition,delta.max(),flush=True)
with (OUT/'cached_embedding_primary_scores.csv').open('x') as f: pd.concat(raw).to_csv(f,index=False,float_format='%.17g')
with (OUT/'cached_embedding_check.json').open('x') as f: json.dump({'status':'completed',
    'command':[sys.executable,*sys.argv], 'method':'Historical np.dot on preserved normalized float32 embeddings; CPU analysis only',
    'raw_checkpoint_replay':False,'results':report,'input_shards':inputs},f,indent=2)
