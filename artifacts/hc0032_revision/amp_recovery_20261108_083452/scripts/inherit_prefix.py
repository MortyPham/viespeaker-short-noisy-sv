"""Inherit only the verified pre-overflow successful prefix; preserve original bytes."""
import sys,json,hashlib,torch,datetime,os
import numpy as np
from pathlib import Path
N=Path('/media/ml4u/Samsung_T52/ptnk_team/mortyhuaphuoc/artifacts/hc0032_revision/amp_recovery_20261108_083452');F=Path('/media/ml4u/Samsung_T52/ptnk_team/mortyhuaphuoc/artifacts/hc0032_revision/fixed_capacity_20261108_060219')
def load(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rec(p):return {'path':str(p),'sha256':sha(p)}
assert load(N/'diagnostic_B8344_controlled_v2/result.json')['passed']
assert load(N/'D_retry_and_resume_validation/result.json')['status']=='passed'
assert load(N/'qualification_recovery_2027/PROFILE_PASS.json')['status']=='passed'
original_cfg=load(F/'configs/B_seed2027.json');new_cfg=load(N/'configs/B_seed2027.json')
for key in ['training','inputs','initialization','optimizer','method_settings','selection_rule','test_protocol','pretrained_bundle']:assert original_cfg[key]==new_cfg[key]
parent=load(F/'runs/seed2027/B/resume_pointer.json');assert sha(parent['path'])==parent['sha256']
ck=torch.load(parent['path'],map_location='cpu',weights_only=False);assert ck['global_step']==8000 and ck['best_checkpoint']==parent['path']
source=F/'runs/seed2027/B';out=N/'runs/seed2027/B';out.mkdir(parents=True,exist_ok=False)
metrics=[json.loads(x) for x in (source/'training_metrics.jsonl').read_text().splitlines() if json.loads(x)['step']<=8000]
fingerprints=[json.loads(x) for x in (source/'batch_fingerprints.jsonl').read_text().splitlines() if json.loads(x)['step']<=8000]
assert [x['step'] for x in metrics]==list(range(1,8001));assert [x['step'] for x in fingerprints]==list(range(1,8001))
scales=[x['AMP_scale'] for x in metrics];assert all(a<=b for a,b in zip(scales,scales[1:])), 'Cannot inherit a prefix that silently skipped/retried an update'
newpath=out/'inherited_checkpoint_step08000.pt'
ck['config_sha256']=sha(N/'configs/B_seed2027.json');ck['cohort']=new_cfg['cohort'];ck['best_checkpoint']=str(newpath);ck['parent_checkpoint']=parent
ck['prefix_inheritance']={'updates':8000,'rule':'New retry branch never activated in parent successful prefix; original data/model/optimizer/RNG bytes retained; protocol amendment explicit','protocol_amendment':rec(N/'VALIDATED_AMENDMENT.md'),'tensor_values_changed':False}
with newpath.open('xb') as f:torch.save(ck,f);f.flush();os.fsync(f.fileno())
original=torch.load(parent['path'],map_location='cpu',weights_only=False);roundtrip=torch.load(newpath,map_location='cpu',weights_only=False)
def equal_tree(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal_tree(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
 return a==b
for key in ['student','aam_head','teacher','optimizer','scheduler','scaler','rng_state','loader_generator_state','sampler_state','history','best_noise_eer']:assert equal_tree(original[key],roundtrip[key]),key
del original,roundtrip
for name in ['training_metrics.jsonl','batch_fingerprints.jsonl']:
 with (out/name).open('x') as f:
  for line in (source/name).read_text().splitlines():
   if json.loads(line)['step']<=8000:f.write(line+'\n')
with (out/'resume_pointer.json').open('x') as f:json.dump(rec(newpath),f,indent=2)
with (out/'INHERITED_PREFIX.json').open('x') as f:json.dump({'claim_status':'VERIFIED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent_checkpoint':parent,'new_checkpoint':rec(newpath),'inherited_successful_updates':8000,'parent_metric_file':rec(source/'training_metrics.jsonl'),'parent_fingerprint_file':rec(source/'batch_fingerprints.jsonl'),'discarded_partial_steps':'Original8001..8343remain archived and are replayed under amended common protocol','original_failure':'Step8344nonfinite gradients; no completed comparator exists','config_changes':['output/cohortmetadata','commonAMProllbackretrypolicy','execution_source'],'unchanged': ['alltrainingfields','data/init/protocolhashes','student/head/optimizer/scheduler/scaler/RNGtensorstate','developmentselectionhistory'] },f,indent=2)
print(json.dumps({'inherited_checkpoint':str(newpath),'bytes':newpath.stat().st_size,'successful_prefix':8000}))
