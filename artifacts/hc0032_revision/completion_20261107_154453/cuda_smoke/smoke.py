"""Representative CUDA parity/timing smoke; complete original batches; no full inference."""
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
os.environ['HF_HUB_OFFLINE']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
ROOT=Path('/media/ml4u/Samsung_T52/ptnk_team/mortyhuaphuoc')
OUT=Path(__file__).resolve().parent
QUEUE=OUT.parent/'queue'
start=time.perf_counter()
spec=importlib.util.spec_from_file_location('frozen_gpu_helpers',QUEUE/'gpu_stages.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch,np,pd=m.torch,m.np,m.pd
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
torch.manual_seed(2026)
assert torch.cuda.is_available(), 'CUDA required; no CPU fallback'
m.assert_inputs()
torch.cuda.reset_peak_memory_stats()
embedding_checks=[]
from torch.nn.utils.rnn import pad_sequence

def save(name,value):
    with (OUT/name).open('x') as f:json.dump(value,f,indent=2,default=str)
def event(kind,**values):
    row={'time_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.perf_counter()-start,'event':kind,**values}
    with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)

for record in m.read(QUEUE/'code_release.json'):
    assert m.digest(record['path'])==record['sha256'],record['path']
for model,c in m.MODELS.items():
    assert m.digest(c['score_file']['path'])==c['score_file']['sha256']
    if model!='A':assert m.digest(c['checkpoint']['path'])==c['checkpoint']['sha256']
primary=m.archived('B');primary=primary.loc[primary.condition=='enroll3s_probe1.25s_clean'].sort_values('trial_id')
chosen=[]
for label in [0,1]:
    group=primary.loc[primary.label==label]
    for quantile in [.1,.7]:chosen.append(int(group.iloc[int(quantile*(len(group)-1))].trial_id))
trials=primary.loc[primary.trial_id.isin(chosen)]
assert len(trials)==4 and int(trials.label.sum())==2
save('sample_plan.json',{'trial_ids':chosen,'trials':trials.to_dict('records'),
    'sampling':'Two prespecified trial-ID quantiles per label; all A/B/C/D and three noise levels',
    'batch_policy':'Complete original sorted eligible BS8 batches containing each selected utterance, never batch only selected keys',
    'tolerances':{'score_absolute':m.SCORE_TOL,'full_replay_eer_pp':m.EER_TOL_PP},
    'device':'cuda','threads':4,'queue_unchanged':True})
event('STARTED',trial_ids=chosen,device='cuda',threads=4)
times=[];results=[];raw=[]
def encode(model,role,side,duration,snr,net):
    keys=m.eligible(model,side,duration)
    wanted=set(trials[side])
    positions={k:i for i,k in enumerate(keys)}
    batches=sorted({positions[k]//8 for k in wanted})
    vectors={};batch_reports=[]
    condition='enrollment_3s' if side=='enrollment' else m.hist.condition_id(duration,snr)
    for batchid in batches:
        batchkeys=keys[batchid*8:(batchid+1)*8]
        t=time.perf_counter()
        waves=[torch.from_numpy(m.hist.make_enrollment(m.prepared(model),k,3.,m.SEED) if side=='enrollment' else m.probe_wave(model,k,1.25,snr)) for k in batchkeys]
        padded=pad_sequence(waves,batch_first=True)
        lengths=torch.tensor([len(w) for w in waves],dtype=torch.float32)/padded.shape[1]
        infer=time.perf_counter()
        with torch.inference_mode():values=torch.nn.functional.normalize(net(padded.to('cuda'),lengths.to('cuda')),dim=-1).cpu().numpy()
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-t
        assert np.isfinite(values).all() and values.shape==(len(batchkeys),192)
        assert np.allclose(np.linalg.norm(values,axis=1),1,atol=1e-5)
        old=m.hist.load_embedding_map(Path(m.MODELS[model]['embedding_cache'][condition]['current_path']))
        delta=max(float(np.max(np.abs(v-old[k]))) for k,v in zip(batchkeys,values,strict=True))
        embedding_checks.append({'model':model,'role':role,'condition':condition,'batchid':batchid,'max_component_delta':delta,'passed':delta<=1e-5})
        vectors.update(zip(batchkeys,values.copy(),strict=True))
        folder=OUT/'embeddings'/model/role/condition;folder.mkdir(parents=True,exist_ok=True)
        with (folder/f'{batchid*8:08d}.npz').open('xb') as f:np.savez_compressed(f,keys=np.asarray(batchkeys),embeddings=values)
        record={'model':model,'role':role,'condition':condition,'batch_start':batchid*8,'count':len(batchkeys),
            'longest_waveform_seconds':padded.shape[1]/16000.,'total_seconds':elapsed,'inference_seconds':time.perf_counter()-infer,
            'full_condition_batches':int(np.ceil(len(keys)/8))}
        times.append(record);batch_reports.append(record)
        event('BATCH_COMPLETED',**record)
    return vectors,batch_reports

for model,cfg in m.MODELS.items():
    roles=set([cfg['enrollment_role'],cfg['probe_role']])
    nets={}
    for role in roles:
        t=time.perf_counter();nets[role]=m.encoder(model,role).to('cuda').eval();event('MODEL_LOADED',model=model,role=role,seconds=time.perf_counter()-t)
    enroll,_=encode(model,cfg['enrollment_role'],'enrollment',3.,None,nets[cfg['enrollment_role']])
    archive=m.archived(model)
    for snr in [None,10.,0.]:
        probes,_=encode(model,cfg['probe_role'],'probe',1.25,snr,nets[cfg['probe_role']])
        condition=m.hist.evaluation_condition_id(3.,1.25,snr)
        group=archive.loc[(archive.condition==condition)&archive.trial_id.isin(chosen)].copy()
        scores=m.score(group,enroll,probes);delta=scores-group.score.to_numpy()
        report={'model':model,'condition':condition,'rows':len(group),'max_absolute_score_delta':float(np.abs(delta).max()),
            'mean_absolute_score_delta':float(np.abs(delta).mean()),'rms_score_delta':float(np.sqrt(np.mean(delta**2))),
            'score_tolerance':m.SCORE_TOL,'passed':bool((np.abs(delta)<=m.SCORE_TOL).all())}
        results.append(report);group['paper_id']=model;group['archived_score']=group.score;group['score']=scores;group['delta']=delta;raw.append(group)
        event('CELL_CHECKED',**report)
    del nets;gc.collect()

all_scores=pd.concat(raw,ignore_index=True)
with (OUT/'smoke_trial_scores.csv').open('x') as f:all_scores.to_csv(f,index=False,float_format='%.17g')
with (OUT/'batch_timings.csv').open('x') as f:pd.DataFrame(times).to_csv(f,index=False)
estimates=[]
for (model,condition),g in pd.DataFrame(times).groupby(['model','condition']):
    estimates.append({'model':model,'condition':condition,'sampled_batches':len(g),'full_batches':int(g.full_condition_batches.iloc[0]),
        'mean_seconds_per_batch':float(g.total_seconds.mean()),'estimated_full_condition_seconds':float(g.total_seconds.mean()*g.full_condition_batches.iloc[0])})
elapsed=time.perf_counter()-start
save('smoke_result.json',{'status':'passed' if (all(r['passed'] for r in results) and all(r['passed'] for r in embedding_checks)) else 'parity_failed',
    'device':'cuda','threads':4,'command':[sys.executable,*sys.argv],'elapsed_seconds':elapsed,'sampled_trial_scores':len(all_scores),
    'max_absolute_score_delta':max(r['max_absolute_score_delta'] for r in results),'score_tolerance':m.SCORE_TOL,
    'cell_results':results,'condition_runtime_estimates':estimates,
    'estimated_full_primary_replay_seconds':sum(x['estimated_full_condition_seconds'] for x in estimates),
    'runtime_caveat':'Projection from selected complete historical BS8 batches; long contiguous voiced crops and host contention can change runtime.',
    'sampled_batch_count':len(times),'software':{k:m.importlib.metadata.version(k) for k in ['torch','torchaudio','speechbrain','numpy','pandas']},
    'trial_manifest_sha256':m.MODELS['B']['trial_manifest']['sha256'],'frozen_inputs_sha256':m.digest(QUEUE/'frozen_inputs.json'),
    'verified_input_files_consumed':len(m.VERIFIED),'checkpoints':{k:v.get('checkpoint',v.get('pretrained_bundle')) for k,v in m.MODELS.items()},
    'initialization':'Allowlisted historical selected states, no training','output_directory':str(OUT),
    'full_primary_replay_started':False,'follow_on_evaluation_started':False})
event('COMPLETED',passed=all(r['passed'] for r in results),elapsed_seconds=elapsed,estimated_full_seconds=sum(x['estimated_full_condition_seconds'] for x in estimates))

save('embedding_checks.json',embedding_checks)
passed=all(r['passed'] for r in results) and all(r['passed'] for r in embedding_checks)
save('cuda_resource_report.json',{'status':'passed' if passed else 'parity_failed','peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'backend':m.metadata('representative_cuda_smoke')})
if passed:
    m.save(OUT.parent/'prompt1_replay/smoke_pass.json',{'status':'passed','representative_smoke':str(OUT/'smoke_result.json'),'embedding_checks':str(OUT/'embedding_checks.json'),'tolerance':1e-5})
else: raise RuntimeError('CUDA parity failed; dependent historical inference stopped; all score diagnostics preserved')
