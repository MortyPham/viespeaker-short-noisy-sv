"""Read-only CPU audit of existing HC0032 evidence; writes only this audit dir."""
from pathlib import Path
import os,json,csv,hashlib,datetime,gc,math,statistics
import numpy as np
import pandas as pd
import torch
ROOT=Path('/media/ml4u/Samsung_T52/ptnk_team/mortyhuaphuoc');OUT=Path(__file__).resolve().parent
REV=ROOT/'artifacts/hc0032_revision';F=REV/'amp_recovery_20261108_083452';H=REV/'overnight_20261106/prompt4_historical';CD=REV/'roles_cd_20261107_165713';P1=REV/'prompt1_20261106_2153'
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','Audit must disable CUDA visibility'
seen={}
def load(p):
 p=Path(p);register(p);return json.loads(p.read_text())
def register(p):
 p=Path(p).resolve()
 if str(p) not in seen:
  with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
  seen[str(p)]={'path':str(p),'sha256':h,'bytes':p.stat().st_size}
 return seen[str(p)]
def verify(info):assert register(info['path'])['sha256']==info['sha256'],info['path']
def frame(p):register(p);return pd.read_csv(p,float_precision='round_trip')
def eer(y,s):
 y=np.asarray(y,np.int64);s=np.asarray(s,np.float64);order=np.argsort(-s,kind='stable');y=y[order];s=s[order]
 ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1];tp=np.cumsum(y)[ends];fp=np.cumsum(1-y)[ends]
 far=np.r_[0,fp/fp[-1]];frr=np.r_[1,1-tp/tp[-1]];d=frr-far;cross=np.flatnonzero(d[:-1]*d[1:]<=0);assert len(cross)
 i=int(cross[0]);w=d[i]/(d[i]-d[i+1]) if d[i]!=d[i+1] else 0;return float(100*(far[i]+w*(far[i+1]-far[i])))
def tensor_hash(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):
  x=v.detach().cpu().contiguous();h.update(k.encode());h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode());h.update(x.numpy().tobytes())
 return h.hexdigest()
def grid(df,variant=None):
 values=[]
 for c,g in df.groupby('condition',sort=True):values.append({'condition':c,'eer_percent':eer(g.label,g.score),'rows':len(g),'target':int(g.label.sum()),'nontarget':int((g.label==0).sum()),'exclusions':5000-len(g)})
 return values
result={'audit_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'audit_mode':'CPU saved-data/checkpoint audit only; CUDA_VISIBLE_DEVICES empty; no models executed','fresh':[],'historical':{},'roles':{},'checks':[]}
init_cache={};batchrefs={};baseline=None;realizations=None;commoncfg={}
for seed in [2027,2028]:
 for m in 'BCD':
  cfgpath=F/f'configs/{m}_seed{seed}.json';cfg=load(cfgpath);run=F/f'runs/seed{seed}/{m}';ev=F/f'evaluation/seed{seed}/{m}'
  done=load(run/'completion.json');selected=load(run/'selection.json');evaluation=load(ev/'completion.json');resume=load(run/'resume_pointer.json')
  assert done['global_step']==10000 and done['status']=='completed';assert done['config_sha256']==register(cfgpath)['sha256'];verify(selected['checkpoint']);verify(resume)
  assert done['selected_checkpoint']==selected['checkpoint']==evaluation['selected_checkpoint'];assert evaluation['scored_rows']==101610 and evaluation['conditions']==27 and evaluation['roles']=='student/student'
  for info in cfg['inputs']+[cfg['initialization'],cfg['test_protocol']]:verify(info)
  if seed not in init_cache:
   initial=torch.load(cfg['initialization']['path'],map_location='cpu',weights_only=False);init_cache[seed]={'file':cfg['initialization'],'student_tensor_sha256':tensor_hash(initial['student']),'head_tensor_sha256':tensor_hash(initial['aam_head'])};del initial
  assert init_cache[seed]['student_tensor_sha256']==cfg['initialization']['student_tensor_sha256'] and init_cache[seed]['head_tensor_sha256']==cfg['initialization']['head_tensor_sha256']
  if seed not in commoncfg:commoncfg[seed]=cfg
  else:
   b=commoncfg[seed]
   for k in ['inputs','initialization','augmentation_policy','RNG_policy','amp_policy','selection_rule','test_protocol']:assert b[k]==cfg[k],(m,seed,k)
   for k in cfg['training']:
    if k not in ['lambda_vector','lambda_kl']:assert b['training'][k]==cfg['training'][k],(m,seed,k)
  history=selected['history'];assert [x['step'] for x in history]==list(range(1000,10001,1000));best=min(history,key=lambda x:x['noise_eer']);assert best['noise_eer']==selected['noise_eer']
  ck=torch.load(resume['path'],map_location='cpu',weights_only=False);assert ck['global_step']==10000 and ck['scheduler']['last_epoch']==10000 and ck['sampler_state']['next_start_step']==10000
  assert (ck['teacher'] is None)==(m=='B');steps=[int(x['step']) for x in ck['optimizer']['state'].values()];assert min(steps)==max(steps)==10000
  statefields={k:k in ck for k in ['optimizer','scheduler','scaler','rng_state','loader_generator_state','sampler_state','student','aam_head','teacher']};del ck;gc.collect()
  ck=torch.load(selected['checkpoint']['path'],map_location='cpu',weights_only=False);assert ck['global_step']==best['step'];del ck;gc.collect()
  metricspath=run/'training_metrics.jsonl';register(metricspath);metrics=[json.loads(s) for s in metricspath.read_text().splitlines()];assert [x['step'] for x in metrics]==list(range(1,10001))
  assert all(np.isfinite(x['loss']) and np.isfinite(x['encoder_gradient_norm']) and np.isfinite(x['head_gradient_norm']) for x in metrics)
  fp=run/'batch_fingerprints.jsonl';register(fp);lines=[json.loads(s) for s in fp.read_text().splitlines()];assert [x['step'] for x in lines]==list(range(1,10001));fingerprints=[x['sha256'] for x in lines]
  if seed not in batchrefs:batchrefs[seed]=fingerprints
  else:assert batchrefs[seed]==fingerprints,(seed,m,'training input mismatch')
  for info in evaluation['outputs']:
   if info['path'].endswith('.csv'):verify(info)
  df=frame(ev/'scores_long.csv');assert len(df)==101610 and df.condition.nunique()==27
  keys=df[['condition','trial_id','label','enrollment','probe']].sort_values(['condition','trial_id']).reset_index(drop=True);assert not keys.duplicated(['condition','trial_id']).any()
  if baseline is None:baseline=keys
  else:assert keys.equals(baseline),(seed,m,'test rows differ')
  real=set(df.realization_manifest_sha256);assert len(real)==1
  if realizations is None:realizations=real
  else:assert real==realizations
  assert set(df.enrollment_role)==set(df.probe_role)=={'student'} and set(df.checkpoint_sha256)=={selected['checkpoint']['sha256']}
  cells=grid(df);saved=frame(ev/'cell_metrics.csv');v={x['condition']:x['eer_percent'] for x in cells}
  assert max(abs(x.eer_percent-v[x.condition]) for x in saved.itertuples())<1e-8
  for x in cells:
   if x['condition'].startswith('enroll3s_probe1.25s_'):assert (x['target'],x['nontarget'],x['exclusions'])==(1937,1818,1245)
  member={'seed':seed,'model':m,'training_complete':True,'training_updates':10000,'optimizer_state_tensors':len(steps),'optimizer_steps_each':10000,'resumable_fields_present':statefields,'selected_step':best['step'],'selected_checkpoint':selected['checkpoint'],'config':register(cfgpath),'initialization':cfg['initialization'],'selection_metric':'minimum10dBdevelopmentEER,SS,every1000,earliesttie','development_10dB_EER_percent':100*selected['noise_eer'],'test_complete':True,'mean27_EER_percent':float(np.mean([x['eer_percent'] for x in cells])),'primary':[x for x in cells if x['condition'].startswith('enroll3s_probe1.25s_')],'rows':len(df),'target':int(df.label.sum()),'nontarget':int((df.label==0).sum()),'exclusions':135000-len(df),'matched_training_inputs':True,'retry_attempts':sum(len(x.get('overflow_retries',[])) for x in metrics),'cells':cells}
  result['fresh'].append(member);print('FRESH_VERIFIED',seed,m,member['mean27_EER_percent'],flush=True)
  del df,keys,metrics,lines;gc.collect()
result['initializations']=init_cache;result['fresh_common_config']=commoncfg[2027];result['complete_seed_blocks']=[2027,2028]
reg=load(P1/'final/model_registry.json');histdfs={};histkeys=None;histcells={}
for m in 'ABCD':
 info=reg['models'][m]['score_file'];verify(info);df=frame(info['path']);histdfs[m]=df
 if 'checkpoint' in reg['models'][m]:verify(reg['models'][m]['checkpoint'])
 keys=df[['condition','trial_id','label','enrollment','probe']].sort_values(['condition','trial_id']).reset_index(drop=True)
 if histkeys is None:histkeys=keys
 else:assert histkeys.equals(keys),(m,'historical rows')
 histcells[m]=grid(df)
result['historical']['means']={m:float(np.mean([x['eer_percent'] for x in histcells[m]])) for m in 'ABCD'}
conditions=sorted(histdfs['B'].condition.unique());emap={m:{x['condition']:x['eer_percent'] for x in histcells[m]} for m in 'BCD'};wins={m:0 for m in 'BCD'}
for c in conditions:wins[min('BCD',key=lambda m:emap[m][c])]+=1
assert wins=={'B':24,'C':3,'D':0};result['historical']['wins']=wins;result['historical']['primary']={m:[x for x in histcells[m] if x['condition'].startswith('enroll3s_probe1.25s_')] for m in 'ABCD'}
for name,folder,pattern,lofield,hifield in [('historical_oneway',H/'results','aggregate_chunk_*.npz','ci95_low_pp','ci95_high_pp'),('historical_twoway',H/'two_way_sensitivity','chunk_*.npz','two_way_sensitivity_ci95_low_pp','two_way_sensitivity_ci95_high_pp'),('fresh_oneway',sorted((F/'matched_statistics').glob('blocks2_*'))[-1],'aggregate_chunk_*.npz','ci95_low_pp','ci95_high_pp')]:
 tablepath=folder/('two_way_sensitivity_intervals.csv' if name=='historical_twoway' else 'aggregate_intervals.csv');tab=frame(tablepath);chunks=[]
 for p in sorted(folder.glob(pattern)):
  register(p)
  with np.load(p) as z:chunks.append(z['effects'])
 boot=np.concatenate(chunks);assert boot.shape==(5000,len(tab));q=np.percentile(boot,[2.5,97.5],axis=0)
 assert max(abs(q[0]-tab[lofield].to_numpy()))<1e-10 and max(abs(q[1]-tab[hifield].to_numpy()))<1e-10
 result[name]={'rows':tab.to_dict('records'),'metadata':load(folder/'run_metadata.json'),'replicates':len(boot),'saved_bootstrap_quantiles_reproduced':True,'source':str(tablepath)}
for m in 'CD':
 replay=load(CD/f'replay/{m}/completion.json');assert replay['status']=='passed' and all(x['max_score_delta']==0 for x in replay['checks'])
 result['roles'][m]={'replay':replay,'scopes':{}}
 for scope in ['primary','grid27']:
  done=load(CD/f'roles/{m}/{scope}/completion.json');verify(done['scores']);df=frame(done['scores']['path']);rolekeys=None;cells={}
  for role,g in df.groupby('enrollment_role'):
   k=g[['condition','trial_id','label','enrollment','probe','realization_manifest_sha256']].sort_values(['condition','trial_id']).reset_index(drop=True)
   if rolekeys is None:rolekeys=k
   else:assert rolekeys.equals(k)
   cells[role]=grid(g)
  folder=CD/f'statistics/{m}/{scope}';agg=frame(folder/'aggregate_effect.csv').iloc[0].to_dict();register(folder/'bootstrap.npz')
  with np.load(folder/'bootstrap.npz') as z:boot=z['eer_percent'];conditions_boot=z['conditions'];assert boot.shape==(5000,3 if scope=='primary' else 27,2)
  dif=(boot[:,:,0]-boot[:,:,1]).mean(axis=1);q=np.percentile(dif,[2.5,97.5]);assert abs(q[0]-agg['ci95_low_pp'])<1e-10 and abs(q[1]-agg['ci95_high_pp'])<1e-10
  result['roles'][m]['scopes'][scope]={'aggregate':agg,'absolute_cells':cells,'rows_per_role':len(df)//2,'outer_join_verified':True,'saved_bootstrap_quantiles_reproduced':True,'probe_cache_fields':{c:sorted(map(str,df[c].unique())) for c in ['probe_cache_root','probe_checkpoint_sha256'] if c in df}}
point=load(REV/'completion_20261107_154453/analysis/comment6_descriptive_H_point.json');verify(point['source_score']);g=histdfs['B'].loc[histdfs['B'].condition==point['condition']];accept=g.score.to_numpy()>=point['threshold'];y=g.label.to_numpy();counts={'TP':int(np.sum(accept&(y==1))),'FP':int(np.sum(accept&(y==0))),'FN':int(np.sum(~accept&(y==1))),'TN':int(np.sum(~accept&(y==0)))}
assert all(counts[k]==point[k] for k in counts);assert counts=={'TP':531,'FP':18,'FN':1406,'TN':1800}
# Independently verify same-test max-TAR/highest-threshold tie convention.
scores=g.score.to_numpy();order=np.argsort(-scores,kind='stable');ss=scores[order];yy=y[order];ends=np.r_[np.flatnonzero(ss[:-1]!=ss[1:]),len(ss)-1];tp=np.cumsum(yy)[ends];fp=np.cumsum(1-yy)[ends];valid=fp<=.01*int((yy==0).sum());besttp=tp[valid].max();candidate=ss[ends][valid&(tp==besttp)].max();assert candidate==point['threshold']
result['operating_point']={'saved':point,'counts_recomputed':counts,'optimal_threshold_and_ties_verified':True}
pool=load(H/'pooling_population_audit.json');extras=[]
for m,info in pool['architectures'].items():
 verify(info);df=frame(info['path']);base=histdfs['B'][['condition','trial_id','label','enrollment','probe']];extra=df.merge(base,on=['condition','trial_id'],how='left',indicator=True,suffixes=('_pool','_main'));extra=extra.loc[extra['_merge']=='left_only'];assert len(df)==101619 and len(extra)==9 and set(extra.trial_id)=={203043} and set(extra.label_pool)=={0};assert all(c.startswith('enroll2s_') for c in extra.condition)
 extras.append({'pooling_model':m,'rows':len(df),'extra_rows':9,'trial_id':203043,'label':0,'conditions':list(extra.condition),'enrollment':sorted(set(extra.enrollment_pool)),'probe':sorted(set(extra.probe_pool))})
result['pooling_extras']=extras
for p in [ROOT/'docs/research/AUX_LOSS_PAPER_DRAFT_20260927.md',F/'FINAL_HANDOFF.json',F/'VALIDATED_AMENDMENT.md',F/'runs/seed2027/B/INHERITED_PREFIX.json',REV/'cohort_finish_20261107_211439/audit/B_STAGE_DIAGNOSIS.json',H/'pooling_manuscript_note.md',CD/'paired_statistics.py',H/'paired_analysis.py',H/'two_way_sensitivity.py',F/'scripts/train_recovery.py',F/'scripts/evaluate_recovery.py',F/'scripts/engine_recovery_v2.py',F/'scripts/paired_matched.py',OUT/'evidence/reviewer_requests_user_supplied.txt']:register(p)
with (OUT/'evidence/verified_measurements.json').open('x') as f:json.dump(result,f,indent=2,default=str)
with (OUT/'evidence/inspected_file_hashes.json').open('x') as f:json.dump(list(seen.values()),f,indent=2)
print('AUDIT_COMPLETE',len(seen),'hashed sources',flush=True)
