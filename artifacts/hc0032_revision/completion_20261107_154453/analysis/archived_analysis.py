"""Read-only archived-score audit and descriptive H ROC read-off. No inference."""
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

R = Path('/media/ml4u/Samsung_T52/ptnk_team/mortyhuaphuoc')
N = Path(__file__).resolve().parents[1]
P1 = R/'artifacts/hc0032_revision/prompt1_20261106_2153'
O = R/'artifacts/hc0032_revision/overnight_20261106'
M = R/'artifacts/hc0032_revision/matched_seeds_20261107_0858'
CPUT = R/'artifacts/hc0032_revision/cpu_replay_20261107_0834'
started = time.perf_counter()
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def load(p): return json.loads(Path(p).read_text())
def save(p, obj):
    with Path(p).open('x') as f: json.dump(obj, f, indent=2, default=str)
def table(p, frame):
    with Path(p).open('x') as f: frame.to_csv(f, index=False, float_format='%.17g')
checks = []
def verify(p, expected):
    observed = sha(p)
    checks.append(dict(path=str(p), expected=expected, observed=observed, passed=observed == expected))
    if observed != expected: raise RuntimeError(f'Hash mismatch: {p}')

verify(P1/'final/model_registry.json', '57daf6b3f5f061ebf71c96cda8e69fd7e3ea7f7ae8cb6184f0b34806991d345b')
verify(P1/'final/protocol_registry.json', '20dd549ff43112515988a434c3e45c1d339a7e0aeab8c0ddf0a006aa26135684')
for manifest, key in [(P1/'final/SHA256SUMS.json',None),
                       (O/'prompt4_historical/completion_manifest.json','artifacts'),
                       (M/'preparation/release.json',None)]:
    entries=load(manifest)
    if key: entries=entries[key]
    for item in entries: verify(item['path'],item['sha256'])
registry=load(P1/'final/model_registry.json')['models']
allow=[]
for system,m in registry.items():
    verify(m['score_file']['path'],m['score_file']['sha256'])
    verify(m['trial_manifest']['path'],m['trial_manifest']['sha256'])
    if system!='A': verify(m['checkpoint']['path'],m['checkpoint']['sha256'])
    else:
        for item in m['pretrained_bundle']['files']:
            verify(P1/'pretrained_bundle_smoke'/Path(item['path']).name,item['sha256'])
    allow.append(dict(system=system,enrollment_role=m['enrollment_role'],probe_role=m['probe_role'],
        checkpoint=m.get('checkpoint',m.get('pretrained_bundle')),scores=m['score_file'],
        historical_matched_seed_qualification=False if system!='A' else 'not applicable'))
save(N/'audit/model_role_allowlist.json',allow)
for seed,expected in [(2027,'d9b65ec5b3be1b6bb47202e59e5e642b2081d4b234941b25eb1c2a51174cb729'),
                      (2028,'b6ce0bf501883b6695296800c93713dc74dac82fb05a89aa925d658ebdd5203f')]:
    verify(M/f'initialization/seed{seed}.pt',expected)
for info in load(M/'configs/B_seed2027.json')['inputs']: verify(info['path'], info['sha256'])
verify(O/'prompt5/prepared/calibration_config.json','86d7d285fc9aa90845d9745333483543904ef03a0c73e09094b6069e8efe298c')
verify(O/'prompt5/calibration.py','dbb7a5dfbd5e67c57a96f0a88fe93311b02ecae8ced188c764591a984ff8f83a')
cfg=load(O/'prompt5/prepared/calibration_config.json')
for info in [*cfg['input_files'],*cfg['prepared_files'].values()]: verify(info['path'],info['sha256'])

frozen=pd.read_csv(registry['B']['trial_manifest']['path']).set_index('trial_id',verify_integrity=True)
assert len(frozen)==5000 and frozen.label.value_counts().to_dict()=={0:2500,1:2500}
key=['condition','trial_id','label','enrollment','probe']
allframes={}; metrics=[]; exclusions=[]
def eer(labels,scores):
    far,tar,_=roc_curve(labels,scores,pos_label=1)
    frr=1-tar; d=frr-far; crossings=np.flatnonzero(d[:-1]*d[1:]<=0)
    if len(crossings):
        i=int(crossings[0]); denom=d[i]-d[i+1]; w=d[i]/denom if denom else 0.
        return float(100*(far[i]+w*(far[i+1]-far[i])))
    i=int(np.argmin(abs(d))); return float(50*(frr[i]+far[i]))

for system,m in registry.items():
    frame=pd.read_csv(m['score_file']['path'],float_precision='round_trip')
    assert len(frame)==101610 and frame.condition.nunique()==27
    assert not frame.duplicated(['condition','trial_id']).any()
    assert set(frame.label.unique())=={0,1} and np.isfinite(frame.score).all()
    pairs=frozen.loc[frame.trial_id].reset_index()
    assert frame[['trial_id','label','enrollment','probe']].reset_index(drop=True).equals(pairs[['trial_id','label','enrollment','probe']])
    qpath=Path(m['score_file']['path']).parent/'audio_quality.csv'
    # Hash is verified by the pinned input inventory before this metadata is used.
    inventory={x['path']:x['sha256'] for x in load(O/'queue/frozen_inputs.json')['inputs']}
    verify(qpath,inventory[str(qpath)])
    quality=pd.read_csv(qpath).set_index('path',verify_integrity=True).voiced_seconds
    for condition,g in frame.groupby('condition',sort=True):
        ed=float(g.enrollment_duration_s.iloc[0]); pdur=float(g.probe_duration_s.iloc[0])
        eligible=(frozen.enrollment.map(quality).ge(ed-1e-6)&frozen.probe.map(quality).ge(pdur-1e-6))
        assert set(frozen.index[eligible])==set(g.trial_id), (system,condition)
        for tid,row in frozen.loc[~eligible].iterrows():
            exclusions.append(dict(system=system,condition=condition,trial_id=tid,label=int(row.label),
                enrollment=row.enrollment,probe=row.probe,
                enrollment_insufficient_or_missing=not bool(quality.get(row.enrollment,float('nan'))>=ed-1e-6),
                probe_insufficient_or_missing=not bool(quality.get(row.probe,float('nan'))>=pdur-1e-6)))
        metrics.append(dict(cohort='historical2026_archived',system=system,seed='' if system=='A' else 2026,
            enrollment_role=m['enrollment_role'],probe_role=m['probe_role'],condition=condition,
            eer_percent=eer(g.label.to_numpy(),g.score.to_numpy()),rows=len(g),target=int(g.label.sum()),
            nontarget=int((g.label==0).sum()),excluded=5000-len(g)))
    allframes[system]=frame
    if system!='A':
        # Outer membership equality: no intersection can hide unmatched rows.
        joined=allframes['A'][key].merge(frame[key],on=['condition','trial_id'],how='outer',suffixes=('_a','_b'),indicator=True,validate='one_to_one')
        assert joined['_merge'].eq('both').all()
        for col in ['label','enrollment','probe']: assert joined[col+'_a'].equals(joined[col+'_b'])
met=pd.DataFrame(metrics)
table(N/'analysis/historical_27cell_metrics.csv',met)
table(N/'analysis/excluded_candidates.csv',pd.DataFrame(exclusions))
primary=met[met.condition.str.startswith('enroll3s_probe1.25s_')]
table(N/'analysis/historical_primary_metrics.csv',primary)

# Descriptive threshold selection on H itself: explicitly separate from calibration.
h=allframes['B'].loc[allframes['B'].condition=='enroll3s_probe1.25s_0dB'].copy()
assert len(h)==3755 and int(h.label.sum())==1937 and int((h.label==0).sum())==1818
y=h.label.to_numpy(dtype=np.int64);s=h.score.to_numpy(dtype=np.float64)
order=np.argsort(-s,kind='stable'); sy=s[order]; yy=y[order]
end=np.r_[np.flatnonzero(sy[:-1]!=sy[1:]),len(sy)-1]
tp=np.cumsum(yy)[end];fp=end+1-tp
choices=[(0,0,float(np.nextafter(s.max(),np.inf)))]
choices += [(int(a),int(b),float(t)) for a,b,t in zip(tp,fp,sy[end]) if int(b)*100<=1818]
true_accepts,false_accepts,threshold=max(choices,key=lambda x:(x[0],x[2]))
accepted=s>=threshold
assert int(((y==1)&accepted).sum())==true_accepts and int(((y==0)&accepted).sum())==false_accepts
result=dict(status='VERIFIED descriptive benchmark ROC read-off',cohort='historical2026_archived_H_same_data_selection',
    system='B',seed=2026,enrollment_role='student',probe_role='student',condition='enroll3s_probe1.25s_0dB',
    threshold=threshold,nominal_far=.01,far=false_accepts/1818,tar=true_accepts/1937,frr=1-true_accepts/1937,
    TP=true_accepts,FP=false_accepts,FN=1937-true_accepts,TN=1818-false_accepts,
    target_trials=1937,nontarget_trials=1818,excluded_candidates=1245,
    far_resolution_percentage_points=100/1818,acceptance_rule='score >= threshold',
    tie_rule='Maximize TAR with empirical FAR <= 1%; then highest threshold; entire equal-score groups accepted together; include just-above-maximum rejection option',
    source_score=registry['B']['score_file'],trial_manifest=registry['B']['trial_manifest'],
    checkpoint=registry['B']['checkpoint'],threshold_selection_data='SAME H rows as evaluation',
    limitation='Descriptive ROC illustration only; no development calibration, independent heldout validation, deployable threshold or guaranteed FAR. No interval reported because threshold was selected on the same H sample.')
save(N/'analysis/comment6_descriptive_H_point.json',result)
table(N/'analysis/comment6_descriptive_H_point.csv',pd.DataFrame([{k:v for k,v in result.items() if not isinstance(v,dict)}]))
h['accepted']=accepted;h['enrollment_speaker_id']=h.enrollment.str.split('/').str[0];h['probe_speaker_id']=h.probe.str.split('/').str[0]
h['checkpoint_sha256']=registry['B']['checkpoint']['sha256']
table(N/'analysis/comment6_descriptive_H_decisions.csv',h)
table(N/'analysis/comment6_descriptive_H_ROC.csv',pd.DataFrame(dict(threshold=sy[end],TP=tp,FP=fp,TAR=tp/1937,FAR=fp/1818)))
save(N/'audit/hash_verification.json',dict(status='VERIFIED',checks=checks,verified_entries=len(checks),
    scope='Pinned artifact manifests, allowlisted checkpoints/scores/trials/bundle, preparation and calibration manifests. Bulk audio not consumed by this CPU analysis; verify each waveform hash before inference.'))
save(N/'analysis/completion.json',dict(status='completed',research_question='Revalidate archived exact populations and compute descriptive 0dB ROC point',
    command=[sys.executable,*sys.argv],seed='historical2026; trial20260818',device='CPU statistics only; no model inference',
    elapsed_seconds=time.perf_counter()-started,finished_utc=datetime.now(timezone.utc).isoformat(),
    software={'python':platform.python_version(),'packages':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','torch','torchaudio','speechbrain','pyarrow','soundfile']}},
    output_directory=str(N/'analysis'),initialization='No model loaded; original score/checkpoint identities preserved',checkpoints='Read-only hash verification only',
    active_processes='See audit/host_state_initial.txt and audit/host_state_final.txt',metrics=result))
print(json.dumps(result,indent=2))
