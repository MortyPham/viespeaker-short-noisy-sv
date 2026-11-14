"""Gated historical inference only: raw primary replay, then C/D student enrollment.

Uses a frozen copy of the historical loader/crop/scorer. Cached probes are reused
unchanged for role comparisons. CUDA is mandatory for all full model inference.
"""
import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
ROOT=Path(__file__).resolve().parents[4]
RUN=Path(__file__).resolve().parents[1]
OLD=ROOT/'artifacts/hc0032_revision/prompt1_20261106_2153'
sys.path.insert(0,str(RUN/'source_snapshot'))
import numpy as np
import pandas as pd
import torch
import evaluate_aam_only_full_viespeaker_h_1e1p as hist
torch.set_num_threads(2)
torch.manual_seed(2026)
REG=json.loads((OLD/'final/model_registry.json').read_text())
MODELS=REG['models']
KEY=['condition','trial_id','label','enrollment','probe']
SEED=20260818
SCORE_TOL=1e-5
EER_TOL_PP=1e-6
FROZEN=None
VERIFIED=set()

def digest(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def read(path): return json.loads(Path(path).read_text())
def save(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f: json.dump(data,f,indent=2,default=str)
def writecsv(path,frame):
    with Path(path).open('x') as f: frame.to_csv(f,index=False,float_format='%.17g')
def relocate(s): return str(s).replace('/media/ml4u/Samsung_T51/ptnk_team/mortyhuaphuoc',str(ROOT))
def verify_frozen(path):
    global FROZEN
    if FROZEN is None:
        FROZEN={r['path']:r for r in read(RUN/'queue/frozen_inputs.json')['inputs']}
    path=str(path)
    assert path in FROZEN,f'Input outside frozen inventory: {path}'
    if path not in VERIFIED:
        assert digest(path)==FROZEN[path]['sha256'],f'Frozen input hash changed: {path}'
        VERIFIED.add(path)

# The historical functions still perform all audio operations; these wrappers
# add per-file hash checks before first use without writing historical caches.
_load_prepared=hist.load_prepared
def checked_prepared(root,key):
    verify_frozen(hist.prepared_audio_path(root,key))
    return _load_prepared(root,key)
hist.load_prepared=checked_prepared
_load_noise=hist.load_noise_waveform
def checked_noise(path):
    verify_frozen(path)
    return _load_noise(path)
hist.load_noise_waveform=checked_noise
_load_embeddings=hist.load_embedding_map
def checked_embeddings(folder):
    for p in Path(folder).glob('*.npz'):
        if str(RUN) not in str(p): verify_frozen(p)
    return _load_embeddings(folder)
hist.load_embedding_map=checked_embeddings
def archived(model): return pd.read_csv(MODELS[model]['score_file']['path'],float_precision='round_trip')
def basepath(model): return Path(MODELS[model]['score_file']['path']).parent
def prepared(model):
    folder=next(iter(MODELS[model]['embedding_cache'].values()))['current_path']
    return Path(folder.split('/embeddings/')[0])/'prepared_audio'
def quality(model): return pd.read_csv(basepath(model)/'audio_quality.csv').set_index('path').voiced_seconds
def frozen(): return pd.read_csv(MODELS['B']['trial_manifest']['path'])
def eligible(model,side,duration):
    f=frozen(); q=quality(model)
    return sorted(set(f.loc[f[side].map(q).ge(duration-1e-6),side]))
def manifest_record(model):
    return {'checkpoint':MODELS[model].get('checkpoint',MODELS[model].get('pretrained_bundle')),
        'roles':{'enrollment':MODELS[model]['enrollment_role'],'probe':MODELS[model]['probe_role']}}
def assert_inputs():
    verify_frozen(OLD/'final/model_registry.json')
    verify_frozen(OLD/'final/protocol_registry.json')
    for r in read(RUN/'queue/source_snapshot.json'): assert digest(r['path'])==r['sha256'],r['path']
    for m,c in MODELS.items():
        assert digest(c['score_file']['path'])==c['score_file']['sha256']
        assert digest(c['trial_manifest']['path'])==c['trial_manifest']['sha256']
        if m!='A': assert digest(c['checkpoint']['path'])==c['checkpoint']['sha256']
        else:
            for f in c['pretrained_bundle']['files']: assert digest(f['path'])==f['sha256']
    assert torch.cuda.is_available(),'CUDA required; no CPU fallback'
    assert digest(ROOT/'datasets/VieSpeaker/repository/trials/viespeaker_H.txt')=='298addeecd574cf1b601f69e2c0971df9173b36cd741e6981846f6b4950793b0'

def encoder(model,role):
    bundle=OLD/'pretrained_bundle_smoke'
    # The isolated regular-file bundle was hash-verified by Prompt1.
    for p in MODELS['A']['pretrained_bundle']['files']:
        assert digest(bundle/Path(p['path']).name)==p['sha256']
    net=hist.load_pretrained_encoder(bundle)
    if model!='A':
        ck=torch.load(MODELS[model]['checkpoint']['path'],map_location='cpu',weights_only=False)
        selected=MODELS[model]['checkpoint_metadata']['step']
        assert ck['global_step']==selected
        net.load_state_dict(ck[role],strict=True)
        assert all(torch.equal(v,ck[role][k]) for k,v in net.state_dict().items())
        del ck
    net.eval()
    assert all(not m.training for m in net.modules())
    return net

RECIPES={}
def recipes(model):
    if model not in RECIPES:
        rows=pd.read_csv(basepath(model)/'probe_transform_manifest.csv',dtype=str).fillna('')
        RECIPES[model]={(r['probe'],r['condition']):r for r in rows.to_dict('records')}
    return RECIPES[model]

def probe_wave(model,key,duration,snr):
    condition=hist.condition_id(duration,snr)
    r=recipes(model)[key,condition]
    waveform,intervals=hist.load_prepared(prepared(model),key)
    clean=hist.crop_by_voiced_duration(waveform,intervals,duration,np.random.default_rng(int(r['crop_seed'])))
    if snr is None: return hist.peak_guard(clean)
    rng=np.random.default_rng(int(float(r['noise_seed'])))
    noises=[hist.fit_noise(hist.load_noise_waveform(relocate(path)),len(clean),rng) for path in json.loads(r['noise_files'])]
    mixed=np.mean(noises,axis=0).astype(np.float32)
    cp=float(np.mean(clean**2)+1e-8); npower=float(np.mean(mixed**2)+1e-8)
    return hist.peak_guard(clean+math.sqrt(cp/(npower*10**(float(snr)/10.)))*mixed)

def cache_folder(model,role,condition,stage):
    identity=MODELS[model]['checkpoint']['sha256'] if model!='A' else MODELS['A']['pretrained_bundle']['historical_directory_sha256']
    protocol=MODELS[model]['trial_manifest']['sha256']
    realization=digest(RUN/'queue/frozen_inputs.json')
    return RUN/stage/'fresh_embeddings'/f'{model}_{identity}'/f'protocol_{protocol}'/f'realization_{realization}'/role/condition/'bs8'

def encode(model,role,side,duration,snr,stage,smoke=False):
    condition=f'enrollment_{duration:g}s' if side=='enrollment' else hist.condition_id(duration,snr)
    keys=eligible(model,side,duration)
    if smoke: keys=keys[:8]
    folder=cache_folder(model,role,condition,stage)
    net=encoder(model,role)
    fn=(lambda key:hist.make_enrollment(prepared(model),key,duration,SEED)) if side=='enrollment' else (lambda key:probe_wave(model,key,duration,snr))
    failures=hist.encode_cached(net,keys,fn,folder,f'{stage} {model}/{role} {condition}',8,torch.device('cuda'))
    assert not failures,f'No silent embedding exclusion: {failures}'
    del net; gc.collect(); torch.cuda.empty_cache()
    values=hist.load_embedding_map(folder)
    assert set(values)==set(keys),'Missing or extra embeddings'
    return values,folder

def score(frame,enroll,probe):
    assert set(frame.enrollment)<=set(enroll) and set(frame.probe)<=set(probe)
    return np.array([float(np.dot(enroll[e],probe[p])) for e,p in frame[['enrollment','probe']].itertuples(index=False,name=None)],dtype=np.float64)

def metadata(stage):
    return {'stage':stage,'command':[sys.executable,*sys.argv], 'seed':SEED,'training_seed':2026,
        'device':'cuda','pid':os.getpid(),'time_utc':datetime.now(timezone.utc).isoformat(),
        'software':{k:importlib.metadata.version(k) for k in ['torch','torchaudio','speechbrain','numpy','pandas','scikit-learn']},
        'numeric_backend':{'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'cudnn_benchmark':torch.backends.cudnn.benchmark},
        'code_revision':'No Git repository; immutable source_snapshot.json hashes',
        'trial_manifest_sha256':MODELS['B']['trial_manifest']['sha256'],
        'noise_policy':'Use archived literal probe recipes. Known H noise overlap retained unchanged.',
        'checkpoint_allowlist':{m:manifest_record(m) for m in MODELS},
        'input_hash_inventory':str(RUN/'queue/frozen_inputs.json'),
        'active_processes':subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv'],capture_output=True,text=True).stdout,
        'gpu_process_log':str(RUN/'queue/events.jsonl')}

def smoke():
    assert_inputs()
    out=RUN/'prompt1_replay/smoke_pass.json'
    if out.exists(): return
    reports=[]
    for model in MODELS:
        for side in ['enrollment','probe']:
            role=MODELS[model]['enrollment_role'] if side=='enrollment' else MODELS[model]['probe_role']
            role='teacher' if role=='teacher' else role
            condition='enrollment_3s' if side=='enrollment' else '1.25s_0dB'
            values,folder=encode(model,role,side,3. if side=='enrollment' else 1.25,0.,'prompt1_replay/smoke',True)
            old=hist.load_embedding_map(Path(MODELS[model]['embedding_cache'][condition]['current_path']))
            delta=max(float(np.max(np.abs(v-old[k]))) for k,v in values.items())
            reports.append({'model':model,'role':role,'condition':condition,'max_embedding_absolute_delta':delta,'tolerance':1e-5})
            assert delta<=1e-5,f'Smoke replay mismatch: {reports[-1]}'
    save(out,{'status':'passed',**metadata('prompt1_smoke'),'results':reports})

def replay():
    assert_inputs(); assert (RUN/'prompt1_replay/smoke_pass.json').exists()
    passed=RUN/'prompt1_replay/replay_pass.json'
    if passed.exists(): return
    reports=[]; raw=[]
    for model in MODELS:
        enrollment,efolder=encode(model,MODELS[model]['enrollment_role'],'enrollment',3.,None,'prompt1_replay')
        old=archived(model)
        for snr in [None,10.,0.]:
            probes,pfolder=encode(model,MODELS[model]['probe_role'],'probe',1.25,snr,'prompt1_replay')
            condition=hist.evaluation_condition_id(3.,1.25,snr)
            group=old.loc[old.condition==condition].copy()
            values=score(group,enrollment,probes)
            result=hist.verification_metrics(group.label.to_numpy(),values)
            before=hist.verification_metrics(group.label.to_numpy(),group.score.to_numpy())
            delta=float(np.max(np.abs(values-group.score.to_numpy())))
            row={'paper_id':model,'condition':condition,'rows':len(group),'target':int(group.label.sum()),'nontarget':int((group.label==0).sum()),
                'max_score_delta':delta,'score_tolerance':SCORE_TOL,'eer_delta_pp':result['eer_percent']-before['eer_percent'],
                'eer_tolerance_pp':EER_TOL_PP, **result,'enrollment_embeddings':str(efolder),'probe_embeddings':str(pfolder)}
            reports.append(row); group['archived_score']=group.score; group['score']=values; group['paper_id']=model; raw.append(group)
            # Persist diagnostics before raising, without discarding partial evidence.
            save(RUN/'prompt1_replay'/f'{model}_{condition}_comparison_{time.time_ns()}.json',row)
            writecsv(RUN/'prompt1_replay'/f'{model}_{condition}_scores_{time.time_ns()}.csv',group)
            assert delta<=SCORE_TOL and abs(row['eer_delta_pp'])<=EER_TOL_PP,f'Raw replay diverged: {row}'
    writecsv(RUN/'prompt1_replay/fresh_primary_scores.csv',pd.concat(raw,ignore_index=True))
    save(passed,{'status':'passed',**metadata('prompt1_replay'),'results':reports,'historical_initialization_hash':'UNVERIFIED; does not substitute another model, training remains blocked'})

def longform(frame,model,enrollment_role):
    f=frame.copy(); f['seed']=2026; f['system']=model
    f['enrollment_role']=enrollment_role; f['probe_role']='student'
    f['checkpoint_sha256']=MODELS[model]['checkpoint']['sha256']; f['enrollment_checkpoint_sha256']=f.checkpoint_sha256; f['probe_checkpoint_sha256']=f.checkpoint_sha256
    f['enrollment_speaker_id']=f.enrollment.str.split('/').str[0]
    f['probe_speaker_id']=f.probe.str.split('/').str[0]
    f['selection_role']='student/student' if model=='B' else 'teacher/student'
    f['cohort']='historical2026_role_comparison'
    return f

def roles():
    assert_inputs(); assert (RUN/'prompt1_replay/replay_pass.json').exists()
    completed=RUN/'prompt2/completed.json'
    if completed.exists(): return
    expected=archived('B')[KEY].sort_values(['condition','trial_id']).reset_index(drop=True)
    parts=[]; cells=[]
    for model in ['B','C','D']:
        archive=archived(model)
        assert archive[KEY].sort_values(['condition','trial_id']).reset_index(drop=True).equals(expected)
        parts.append(longform(archive,model,'student' if model=='B' else 'teacher'))
        if model=='B': continue
        for duration in [2.,3.,4.]:
            enroll,folder=encode(model,'student','enrollment',duration,None,'prompt2')
            for condition,group in archive.loc[archive.enrollment_duration_s==duration].groupby('condition',sort=True):
                probe_condition=condition.split('_probe')[1]
                cache=Path(MODELS[model]['embedding_cache'][probe_condition]['current_path'])
                probes=hist.load_embedding_map(cache) # Exact same archived probes for both enrollment roles.
                g=group.copy(); g['score']=score(g,enroll,probes)
                parts.append(longform(g,model,'student'))
    long=pd.concat(parts,ignore_index=True)
    assert len(long)==5*101610
    for (model,role),g in long.groupby(['system','enrollment_role']):
        assert g[KEY].sort_values(['condition','trial_id']).reset_index(drop=True).equals(expected)
        for condition,c in g.groupby('condition'):
            cells.append({'system':model,'seed':2026,'enrollment_role':role,'probe_role':'student','condition':condition,
                'rows':len(c),'target':int(c.label.sum()),'nontarget':int((c.label==0).sum()),'excluded_candidates':5000-len(c),**hist.verification_metrics(c.label.to_numpy(),c.score.to_numpy())})
    cell=pd.DataFrame(cells)
    pivot=cell.pivot(index='condition',columns=['system','enrollment_role'],values='eer_percent')
    comparisons=[('C_ss_minus_B',('C','student'),('B','student')),('D_ss_minus_B',('D','student'),('B','student')),
        ('D_ss_minus_C_ss',('D','student'),('C','student')),('C_ts_minus_C_ss',('C','teacher'),('C','student')),
        ('D_ts_minus_D_ss',('D','teacher'),('D','student')),('C_ts_minus_B',('C','teacher'),('B','student')),('D_ts_minus_B',('D','teacher'),('B','student'))]
    diffs=[]
    for label,a,b in comparisons:
        for condition in pivot.index:
            diffs.append({'comparison':label,'seed':2026,'condition':condition,'eer_difference_pp':float(pivot.loc[condition,a]-pivot.loc[condition,b]),'negative_favors':'first named configuration'})
    writecsv(RUN/'prompt2/scores_long.csv',long)
    writecsv(RUN/'prompt2/cell_metrics.csv',cell)
    writecsv(RUN/'prompt2/effect_sizes.csv',pd.DataFrame(diffs))
    save(completed,{'status':'completed',**metadata('prompt2'),'rows':len(long),'probe_policy':'Archived probe vectors fixed byte-for-byte across enrollment roles',
        'selection_rule':'Historical B selected SS, C/D selected TS; same selected checkpoint for both roles, not a matched training cohort',
        'score_export':str(RUN/'prompt2/scores_long.csv'),'cell_metrics':str(RUN/'prompt2/cell_metrics.csv')})

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['smoke','replay','roles']); args=parser.parse_args()
    (RUN/'prompt1_replay').mkdir(exist_ok=True); (RUN/'prompt2').mkdir(exist_ok=True)
    try: {'smoke':smoke,'replay':replay,'roles':roles}[args.stage]()
    except Exception as error:
        save(RUN/'queue'/f'{args.stage}_failure_{time.time_ns()}.json',{'status':'failed','stage':args.stage,'error':repr(error),'command':[sys.executable,*sys.argv],'pid':os.getpid()})
        raise
