"""CPU-only frozen-input preparation; no model loading or CUDA allocation."""
import hashlib,importlib.util,json,sys,time
from pathlib import Path
N=Path(__file__).resolve().parent
started=time.perf_counter()
spec=importlib.util.spec_from_file_location('frozen',N/'queue/gpu_stages.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert m.digest(m.OLD/'final/model_registry.json')=='57daf6b3f5f061ebf71c96cda8e69fd7e3ea7f7ae8cb6184f0b34806991d345b'
assert m.digest(m.OLD/'final/protocol_registry.json')=='20dd549ff43112515988a434c3e45c1d339a7e0aeab8c0ddf0a006aa26135684'
for cfg in m.MODELS.values():
    assert m.digest(cfg['score_file']['path'])==cfg['score_file']['sha256']
    assert m.digest(cfg['trial_manifest']['path'])==cfg['trial_manifest']['sha256']
    if 'checkpoint' in cfg:assert m.digest(cfg['checkpoint']['path'])==cfg['checkpoint']['sha256']
ref=m.archived('B');ref=ref.loc[ref.condition=='enroll3s_probe1.25s_clean'].sort_values('trial_id')
chosen=[]
for label in [0,1]:
    g=ref.loc[ref.label==label]
    chosen.extend(int(g.iloc[int(q*(len(g)-1))].trial_id) for q in [.1,.7])
trials=ref.loc[ref.trial_id.isin(chosen)]
records=[]
for model in m.MODELS:
    for side,duration,snr in [('enrollment',3.,None),('probe',1.25,None),('probe',1.25,10.),('probe',1.25,0.)]:
        keys=m.eligible(model,side,duration);starts=sorted({keys.index(k)//8*8 for k in trials[side]})
        for start in starts:
            batchkeys=keys[start:start+8];lengths=[];hashes=[]
            for k in batchkeys:
                w=m.hist.make_enrollment(m.prepared(model),k,duration,m.SEED) if side=='enrollment' else m.probe_wave(model,k,duration,snr)
                lengths.append(len(w));hashes.append(hashlib.sha256(w.tobytes()).hexdigest())
            records.append({'model':model,'side':side,'duration_s':duration,'snr_db':snr,'batch_start':start,'keys':batchkeys,'raw_sample_lengths':lengths,
                'longest_contiguous_seconds':max(lengths)/16000,'relative_lengths':[x/max(lengths) for x in lengths],'waveform_sha256':hashes})
obj={'status':'VERIFIED CPU-only original-batch/input preparation','command':[sys.executable,*sys.argv],'elapsed_seconds':time.perf_counter()-started,
    'models_loaded':0,'GPU_workers_started':0,'trial_ids':chosen,'target_trials':2,'nontarget_trials':2,'batches':records,
    'maximum_contiguous_waveform_seconds':max(x['longest_contiguous_seconds'] for x in records),'verified_consumed_files':len(m.VERIFIED),
    'trial_manifest_sha256':m.MODELS['B']['trial_manifest']['sha256'],'frozen_inventory_sha256':m.digest(N/'queue/frozen_inputs.json'),
    'software':{x:m.importlib.metadata.version(x) for x in ['torch','torchaudio','speechbrain','numpy','pandas']}}
with (N/'audit/batch_plan.json').open('x') as f:json.dump(obj,f,indent=2)
print(json.dumps({k:v for k,v in obj.items() if k!='batches'},indent=2))
