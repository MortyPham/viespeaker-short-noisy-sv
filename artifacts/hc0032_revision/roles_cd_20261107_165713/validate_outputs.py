"""CPU-only independent artifact/overlap/archived-cache checks after role extraction."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
N=Path(__file__).resolve().parent
R=N.parents[2]
def load(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def run(model):
    for manifest in ['RELEASE.json','GRID_RELEASE.json']:
        for x in load(N/manifest):assert sha(x['path'])==x['sha256']
    original=load(R/'artifacts/hc0032_revision/prompt1_20261106_2153/final/model_registry.json')['models'][model]
    assert sha(original['checkpoint']['path'])==original['checkpoint']['sha256']
    assert load(N/'replay'/model/'completion.json')['status']=='passed'
    a=pd.read_csv(N/'roles'/model/'primary/scores_long.csv',float_precision='round_trip')
    b=pd.read_csv(N/'roles'/model/'grid27/scores_long.csv',float_precision='round_trip')
    overlap=b.loc[b.condition.str.startswith('enroll3s_probe1.25s_')]
    key=['condition','trial_id','enrollment_role']
    j=a.merge(overlap,on=key,how='outer',validate='one_to_one',indicator=True,suffixes=('_a','_b'))
    assert len(j)==22530 and j['_merge'].eq('both').all()
    for f in ['score','label','enrollment','probe','enrollment_speaker_id','probe_speaker_id','enrollment_checkpoint_sha256','probe_checkpoint_sha256','realization_manifest_sha256']:
        assert j[f+'_a'].equals(j[f+'_b']),f
    journal=[json.loads(line) for line in (N/'batches.jsonl').read_text().splitlines()]
    verified=0
    for x in journal:
        if x['model']==model:assert sha(x['shard']['path'])==x['shard']['sha256'];verified+=1
    archive=pd.read_csv(original['score_file']['path'],float_precision='round_trip')
    matrices={};cache_checks=[]
    inventory={x['path']:x['sha256'] for x in load(N/'queue/frozen_inputs.json')['inputs']}
    def embeddings(condition):
        if condition not in matrices:
            folder=Path(original['embedding_cache'][condition]['current_path']);vectors={}
            for p in sorted(folder.glob('*.npz')):
                assert sha(p)==inventory[str(p)]
                with np.load(p,allow_pickle=False) as d:
                    keys=d['keys'].astype(str).tolist();values=d['embeddings'].copy()
                assert not set(keys)&set(vectors)
                vectors.update(zip(keys,values,strict=True))
            matrices[condition]=vectors
        return matrices[condition]
    for condition,g in archive.groupby('condition'):
        duration=float(g.enrollment_duration_s.iloc[0]);em=embeddings(f'enrollment_{duration:g}s');pm=embeddings(condition.split('_probe')[1])
        values=np.asarray([float(np.dot(em[e],pm[p])) for e,p in g[['enrollment','probe']].itertuples(index=False,name=None)])
        delta=float(np.max(np.abs(values-g.score.to_numpy())))
        assert delta<=1e-5
        cache_checks.append({'condition':condition,'rows':len(g),'max_reconstructed_TS_score_delta':delta})
    summary={'status':'passed','model':model,'primary_grid_overlap_exact_rows':len(j),'fresh_shards_rehashed':verified,
        'archived_TS_reconstruction':cache_checks,'scores_per_role':101610,'scored_conditions':27,'all_finite':bool(np.isfinite(b.score).all()),
        'scope':'Primary fresh raw replay; all27within-model role scores; archivedTS/cache consistency, not fresh all27raw replay'}
    assert summary['all_finite'] and len(b)==406440/2
    with (N/'audit'/f'{model}_FINAL_OUTPUT_VALIDATION.json').open('x') as f:json.dump(summary,f,indent=2)
    print(json.dumps({k:v for k,v in summary.items() if k!='archived_TS_reconstruction'}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('model',choices=['C','D']);args=p.parse_args();run(args.model)
