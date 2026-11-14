#!/usr/bin/env python3
"""HC0032 deterministic calibration preparation and CUDA-only operating point.

prepare is CPU metadata/audio-header work only. run must be queued after the
fresh checkpoint replay; independent of full enrollment-role grid; it never selects thresholds on H.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUDIT = ROOT / 'artifacts/hc0032_revision/prompt1_20261106_2153'
sys.path.insert(0, str(HERE.parent/'source_snapshot'))
SEED = 20261106
NOMINAL_FAR = .01

def now(): return datetime.now(timezone.utc).isoformat()
def digest(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def load(path): return json.loads(Path(path).read_text())
def save(path, value):
    with Path(path).open('x') as f: json.dump(value, f, indent=2, default=str)
def fileinfo(path):
    p = Path(path).absolute()
    return {'path': str(p), 'sha256': digest(p), 'bytes': p.stat().st_size}
def software():
    ans = {'python': platform.python_version()}
    for name in ['numpy', 'pandas', 'torch', 'torchaudio', 'speechbrain', 'soundfile']:
        try: ans[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: ans[name] = None
    return ans
def writecsv(path, rows):
    with Path(path).open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def preserve_text(path, text):
    """Resume without replacing an already written result."""
    path=Path(path)
    if path.exists():
        if path.read_text()!=text: raise RuntimeError(f'Conflicting preserved result: {path}')
    else:
        with path.open('x') as f: f.write(text)
def preserve_json(path, payload):
    path=Path(path)
    if path.exists():
        if load(path)!=payload: raise RuntimeError(f'Conflicting preserved JSON: {path}')
    else: save(path,payload)
def check_hash(info):
    if digest(info['path']) != info['sha256']: raise RuntimeError(f'Input hash changed: {info["path"]}')

def prepare(args):
    import numpy as np
    import pandas as pd
    import soundfile as sf
    import evaluate_aam_only_full_viespeaker_h_1e1p as historical
    out = args.prepared
    out.mkdir(parents=True, exist_ok=False)
    registry = load(AUDIT/'final/model_registry.json')
    manifest_path = Path(registry['models']['B']['resolved_config'].get('recording_manifest', ROOT/'experiments/shared/d_full_audio_cache/recordings.csv'))
    manifest_path = ROOT/'experiments/shared/d_full_audio_cache/recordings.csv'
    split_path = ROOT/'experiments/shared/speaker_splits.csv'
    assert digest(manifest_path) == registry['models']['B']['resolved_config']['recording_manifest_sha256']
    assert digest(split_path) == registry['models']['B']['resolved_config']['speaker_split_sha256']
    records = pd.read_csv(manifest_path)
    partitions = pd.read_csv(split_path)
    vaudit = load(AUDIT/'validation_audit.json')
    speakers = sorted(vaudit['excluded_validation'])
    original_assign = pd.read_csv(AUDIT/'reconstructed_validation_assignments.csv')
    selection_speakers = set(original_assign.speaker)
    train = set(registry['models']['B']['resolved_config']['train_speaker_ids'])
    frozen_h = pd.read_csv(Path(registry['models']['B']['trial_manifest']['path']))
    h_ids = set(frozen_h.enrollment.str.split('/').str[0]) | set(frozen_h.probe.str.split('/').str[0])
    assert not set(speakers) & (selection_speakers | train | h_ids)
    assert set(speakers) <= set(partitions.loc[partitions['split']=='validation','speaker_id'])
    # Use the original development MUSAN file pool, removing frozen-H-used files.
    # Frozen H inputs are read only and never replaced.
    old_noise = load(AUDIT/'current_musan_training_validation_files.json')
    h_noise = {x['path'] for x in load(AUDIT/'final/historical_probe_noise_audit.json')['files']}
    pools = {c: [] for c in ['noise','music','speech']}
    for row in old_noise:
        if row['partition']=='validation' and row['path'] not in h_noise:
            pools[row['category']].append(Path(row['path']))
    pools = {k: sorted(v) for k,v in pools.items()}
    assert all(pools.values())
    audio, assignments = {}, []
    for speaker in speakers:
        g = records.loc[(records.speaker_id==speaker) & (records.voiced_seconds>=1.25)].sort_values('source_id')
        enroll = g.loc[g.voiced_seconds>=3].head(5)
        probe = g.loc[~g.source_id.isin(enroll.source_id)].head(10)
        assert len(enroll)==5 and len(probe)==10
        assert not set(enroll.source_id)&set(probe.source_id)
        for role, selected in [('enrollment',enroll),('probe',probe)]:
            for row in selected.to_dict('records'):
                path = ROOT/'experiments/shared/d_full_audio_cache'/row['relative_path']
                header = sf.info(path)
                assert header.samplerate==16000 and header.channels==1
                key = row['source_id']; duration = 3. if role=='enrollment' else 1.25
                if role=='enrollment':
                    recipe={'crop_seed':historical.stable_int(f'enrollment-crop|{SEED}|{key}|{duration:g}'),
                            'category':'clean','noise_seed':None,'noise_files':[]}
                else: recipe=historical.transform_recipe(key,duration,0.,SEED,pools)
                intervals=np.asarray(json.loads(row['vad_intervals']),dtype=np.int64)
                needed=int(round(duration*16000)); lens=np.diff(intervals,axis=1).reshape(-1)
                cumulative=np.concatenate([[0],np.cumsum(lens)])
                start=int(np.random.default_rng(recipe['crop_seed']).integers(0,int(lens.sum())-needed+1))
                end=start+needed
                a=int(np.searchsorted(cumulative[1:],start,side='right'))
                b=int(np.searchsorted(cumulative[1:],end-1,side='right'))
                raw_start=int(intervals[a,0]+start-cumulative[a]); raw_end=int(intervals[b,0]+end-cumulative[b])
                assert 0<=raw_start<raw_end<=header.frames
                audio[f'{role}:{key}']={'key':key,'speaker_id':speaker,'role':role,
                    'path':str(path),'wav_sha256':digest(path),'source_samples':header.frames,
                    'sample_rate':16000,'duration_s':duration,'voiced_seconds':row['voiced_seconds'],
                    'vad_intervals':intervals.tolist(),'raw_start':raw_start,'raw_end':raw_end,
                    'recipe':{k:[str(x) for x in v] if k=='noise_files' else v for k,v in recipe.items()}}
                assignments.append({'speaker_id':speaker,'role':role,'source_id':key,'path':str(path),
                    'duration_s':duration,'raw_start':raw_start,'raw_end':raw_end})
    trials=[]
    ens=[a for a in audio.values() if a['role']=='enrollment']
    prs=[a for a in audio.values() if a['role']=='probe']
    for e in sorted(ens,key=lambda a:a['key']):
        for p in sorted(prs,key=lambda a:a['key']):
            assert e['key'] != p['key']
            trials.append({'trial_id':f'calibration_{len(trials):05d}','enrollment':e['key'],'probe':p['key'],
                'enrollment_speaker_id':e['speaker_id'],'probe_speaker_id':p['speaker_id'],
                'label':int(e['speaker_id']==p['speaker_id']),'condition':'enroll3s_probe1.25s_0dB'})
    assert len(trials)==5000 and sum(t['label'] for t in trials)==500
    writecsv(out/'calibration_trials.csv',trials)
    writecsv(out/'calibration_recordings.csv',assignments)
    save(out/'audio_realizations.json',audio)
    noise_records=[]
    for c,ps in pools.items():
        for p in ps:
            noise_records.append({'category':c,**fileinfo(p)})
    save(out/'calibration_noise_pool.json',noise_records)
    inputs=[fileinfo(p) for p in [manifest_path,split_path,AUDIT/'final/model_registry.json',AUDIT/'validation_audit.json',
        AUDIT/'reconstructed_validation_assignments.csv',AUDIT/'final/historical_probe_noise_audit.json',
        ROOT/'evaluate_aam_only_full_viespeaker_h_1e1p.py']]
    config={'status':'PREPARED; GPU scoring queued behind raw replay and Prompt 2',
        'research_question':'At a development-calibrated nominal1%FAR, what0dB heldout B TAR/FAR is achieved?',
        'command':[sys.executable,*sys.argv],'created_utc':now(),'seed':SEED,'model':'B','training_seed':2026,
        'checkpoint':registry['models']['B']['checkpoint'],'checkpoint_selection':'historical student/student noisy-dev EER, step8000',
        'nominal_far':NOMINAL_FAR,'decision_rule':'score >= threshold',
        'threshold_selector':'Maximize empirical development TAR subject to FAR<=1%; among equal maximal TAR choose highest threshold. Entire tied score groups accepted together. Include a threshold just above maximum score.',
        'enrollment_duration_s':3.,'probe_duration_s':1.25,'probe_snr_db':0.,
        'calibration_speakers':speakers,'calibration_speaker_count':10,
        'historical_selector_speaker_count':len(selection_speakers),'selection_overlap_speakers':[],
        'partition_relationship':'VERIFIED disjoint from current reconstructed90-speaker selector assignments, training3371 and all frozen H candidate filename identities. INFERRED historical selector assignment provenance because original assignment manifest was not persisted.',
        'eligibility':'5 distinct >=3s-voiced enrollment recordings and10 other >=1.25s-voiced probe recordings per calibration speaker; no centroid; one recording per side.',
        'calibration_trials':5000,'target_trials':500,'nontarget_trials':4500,
        'development_far_resolution':1/4500,'expected_false_accept_count_at_nominal_far':45,
        'sparsity_note':'4500 nontarget pairs permit1%FAR in45 error increments, but only10 speaker clusters limit independent calibration information. No precision or population-level guarantee is implied.',
        'noise_policy':'Original development file partition (hash modulo100 in80..89), excluding all frozen H probe-used files; historical crop/noise mixing algorithm unchanged; deterministic new development realizations.',
        'noise_pool_counts':{k:len(v) for k,v in pools.items()},
        'known_historical_H_noise_overlap':{'training_files':326,'development_files':47,'policy':'Preserve original H recipes; new calibration noise pool excludes these H files.'},
        'prepared_files':{name:fileinfo(out/name) for name in ['calibration_trials.csv','calibration_recordings.csv','audio_realizations.json','calibration_noise_pool.json']},
        'input_files':inputs,'software':software(),'code_revision':'No usable Git; exact source hashes recorded',
        'completion_status':'CPU preparation completed; calibration scores/threshold/heldout rates not computed',
        'active_processes':'CPU metadata preparation only; no GPU work or training'}
    save(out/'calibration_config.json',config)
    (out/'PREPARATION.md').write_text('# Prompt5 calibration preparation\n\n**VERIFIED:** Frozen5000-trial development manifest uses10 speakers excluded by the reconstructed historical selector, with500 target and4500 nontarget trials. Enrollment/probe sources are independent. Calibration is disjoint from training and H speaker identities.\n\n**INFERRED:** Historical selector exclusion follows the source/manifest reconstruction; original assignments were not persisted.\n\nThreshold target1%FAR and deterministic tie rule were frozen before scores. The154-file noise pool is disjoint from current training/development-H-used noise files, preserves H realizations, and uses historical crop/mix semantics. GPU scoring and heldout evaluation remain queued behind raw replay and Prompt2.\n')
    print(json.dumps({'prepared':str(out),'speakers':10,'targets':500,'nontargets':4500,'config':str(out/'calibration_config.json')},indent=2))

def threshold_from_development(labels,scores):
    import numpy as np
    labels=np.asarray(labels,dtype=np.int64); scores=np.asarray(scores,dtype=np.float64)
    if set(np.unique(labels))!={0,1} or not np.isfinite(scores).all(): raise ValueError('Invalid calibration scores')
    n1=int(labels.sum()); n0=len(labels)-n1
    # Scores with equal values must be accepted together under score>=threshold.
    order=np.argsort(-scores,kind='stable'); s=scores[order]; y=labels[order]
    end=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
    tp=np.cumsum(y)[end]; fp=end+1-tp
    feasible=fp*100<=n0
    choices=[(0,0,float(np.nextafter(scores.max(),np.inf)))]
    choices += [(int(a),int(b),float(t)) for a,b,t,ok in zip(tp,fp,s[end],feasible) if ok]
    tps,fps,threshold=max(choices,key=lambda x:(x[0],x[2]))
    return {'threshold':threshold,'nominal_far':.01,'threshold_rule':'score >= threshold',
        'tie_rule':'maximal TAR then highest threshold; tied scores included together',
        'target_trials':n1,'nontarget_trials':n0,'true_accepts':tps,'false_accepts':fps,
        'true_rejects':n0-fps,'false_rejects':n1-tps,'far':fps/n0,'tar':tps/n1,'frr':1-tps/n1,
        'empirical_far_resolution':1/n0}

def run(args):
    assert load(HERE.parent/'prompt1_replay/replay_pass.json')['status']=='passed', 'Historical B parity required'
    import numpy as np
    import pandas as pd
    import soundfile as sf
    import torch
    import torch.nn.functional as F
    from torch.nn.utils.rnn import pad_sequence
    import evaluate_aam_only_full_viespeaker_h_1e1p as historical
    if args.device!='cuda' or not torch.cuda.is_available(): raise RuntimeError('CUDA is required; CPU fallback is prohibited')
    torch.set_num_threads(2); torch.manual_seed(SEED)
    cfg=load(args.prepared/'calibration_config.json')
    for info in cfg['input_files']: check_hash(info)
    for info in cfg['prepared_files'].values(): check_hash(info)
    check_hash(cfg['checkpoint'])
    audio=load(args.prepared/'audio_realizations.json')
    for a in audio.values():
        if digest(a['path'])!=a['wav_sha256']: raise RuntimeError('Calibration waveform changed')
    for n in load(args.prepared/'calibration_noise_pool.json'): check_hash(n)
    out=args.run_dir; out.mkdir(parents=True,exist_ok=True)
    if (out/'completion.json').exists():
        done=load(out/'completion.json')
        for info in done['outputs'].values(): check_hash(info)
        print('Already completed:',out); return
    def event(status,**kw):
        with (out/'events.jsonl').open('a') as f: f.write(json.dumps({'utc':now(),'status':status,**kw})+'\n')
    event('RUN_STARTED',command=[sys.executable,*sys.argv],pid=os.getpid(),device='cuda')
    # Never write to the historical pretrained bundle: the audit copy is local.
    bundle=AUDIT/'pretrained_bundle_smoke'
    model=historical.load_pretrained_encoder(bundle)
    checkpoint=torch.load(cfg['checkpoint']['path'],map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['student'],strict=True)
    model=model.to('cuda').eval(); del checkpoint
    noise_cache={}
    def realization(a):
        w,sr=sf.read(a['path'],dtype='float32',always_2d=False)
        assert sr==16000 and w.ndim==1
        recipe=a['recipe']; intervals=np.asarray(a['vad_intervals'],dtype=np.int64)
        clean=historical.crop_by_voiced_duration(w,intervals,a['duration_s'],np.random.default_rng(recipe['crop_seed']))
        assert np.array_equal(clean,w[a['raw_start']:a['raw_end']])
        if a['role']=='enrollment': return historical.peak_guard(clean)
        rng=np.random.default_rng(recipe['noise_seed']); noises=[]
        for p in recipe['noise_files']:
            if p not in noise_cache:
                n,sr=sf.read(p,dtype='float32',always_2d=False)
                noise_cache[p]=historical.to_mono_16k(n,sr)
            noises.append(historical.fit_noise(noise_cache[p],len(clean),rng))
        noise=np.mean(noises,axis=0).astype(np.float32)
        scale=math.sqrt(float(np.mean(clean**2)+1e-8)/float(np.mean(noise**2)+1e-8))
        return historical.peak_guard(clean+scale*noise)
    def encode(batch):
        waves=[torch.from_numpy(realization(a)) for a in batch]
        padded=pad_sequence(waves,batch_first=True)
        lengths=torch.tensor([len(w) for w in waves],dtype=torch.float32)/padded.shape[1]
        with torch.inference_mode():
            value=F.normalize(model(padded.to('cuda'),lengths.to('cuda')),dim=-1).cpu().numpy()
        if value.shape!=(len(batch),192) or not np.isfinite(value).all(): raise RuntimeError('Invalid embedding output')
        return value
    # Small real-data CUDA smoke precedes full calibration scoring.
    smoke=encode([next(a for a in audio.values() if a['role']=='enrollment'),next(a for a in audio.values() if a['role']=='probe')])
    event('CUDA_SMOKE_PASSED',shape=list(smoke.shape),finite=bool(np.isfinite(smoke).all()))
    embeddings={}; cache_root=out/'embedding_cache'/cfg['checkpoint']['sha256']/'student'/digest(args.prepared/'audio_realizations.json')
    for role in ['enrollment','probe']:
        values=sorted([a for a in audio.values() if a['role']==role],key=lambda x:x['key'])
        role_dir=cache_root/role; role_dir.mkdir(parents=True,exist_ok=True)
        for start in range(0,len(values),args.batch_size):
            batch=values[start:start+args.batch_size]; shard=role_dir/f'{start:06d}.npz'
            keys=[a['key'] for a in batch]
            if shard.exists():
                with np.load(shard,allow_pickle=False) as data:
                    assert list(data['keys'].astype(str))==keys
                    vectors=data['embeddings'].copy()
                    assert vectors.shape==(len(batch),192) and np.isfinite(vectors).all()
            else:
                vectors=encode(batch)
                # Publish only a complete shard. Retain its unique staging file
                # so an interrupted write is never mistaken for a completed cache.
                staging=role_dir/f'.{start:06d}.staging_{os.getpid()}_{time.time_ns()}.npz'
                with staging.open('xb') as f:
                    np.savez_compressed(f,keys=np.asarray(keys),embeddings=vectors)
                    f.flush(); os.fsync(f.fileno())
                os.link(staging,shard)
            embeddings.update({f'{role}:{k}':v for k,v in zip(keys,vectors,strict=True)})
            event('BATCH_COMPLETED',role=role,start=start,count=len(batch))
    trials=pd.read_csv(args.prepared/'calibration_trials.csv')
    trials['score']=[float(np.dot(embeddings[f'enrollment:{e}'],embeddings[f'probe:{p}'])) for e,p in trials[['enrollment','probe']].itertuples(index=False,name=None)]
    trials['system']='B'; trials['seed']=2026; trials['checkpoint_sha256']=cfg['checkpoint']['sha256']
    trials['enrollment_role']='student'; trials['probe_role']='student'
    preserve_text(out/'calibration_scores.csv',trials.to_csv(index=False,float_format='%.17g'))
    if (out/'threshold_record.json').exists():
        threshold=load(out/'threshold_record.json')
        check_hash(threshold['calibration_manifest']); check_hash(threshold['calibration_scores'])
        independently_selected=threshold_from_development(trials.label.to_numpy(),trials.score.to_numpy())
        assert all(threshold[k]==v for k,v in independently_selected.items())
    else:
        threshold=threshold_from_development(trials.label.to_numpy(),trials.score.to_numpy())
        threshold.update(calibration_manifest=fileinfo(args.prepared/'calibration_trials.csv'),calibration_scores=fileinfo(out/'calibration_scores.csv'),
            selector='development only; frozen before loading heldout score file',frozen_utc=now())
        save(out/'threshold_record.json',threshold)
    # Only after threshold is on disk may heldout scores be read.
    registry=load(AUDIT/'final/model_registry.json'); source=Path(args.heldout_scores or registry['models']['B']['score_file']['path'])
    if digest(source)!=registry['models']['B']['score_file']['sha256']: raise RuntimeError('Heldout score hash outside allowlist')
    h=pd.read_csv(source,float_precision='round_trip'); h=h.loc[h.condition=='enroll3s_probe1.25s_0dB'].copy()
    assert len(h)==3755 and int(h.label.sum())==1937 and not h.duplicated(['condition','trial_id']).any()
    h['accepted']=h.score>=threshold['threshold']; h['enrollment_speaker_id']=h.enrollment.str.split('/').str[0]
    n1=int(h.label.sum()); n0=len(h)-n1
    tp=int(((h.label==1)&h.accepted).sum()); fp=int(((h.label==0)&h.accepted).sum())
    rates={'threshold':threshold['threshold'],'nominal_development_far':.01,'target_trials':n1,'nontarget_trials':n0,
        'true_accepts':tp,'false_accepts':fp,'false_rejects':n1-tp,'true_rejects':n0-fp,
        'far':fp/n0,'tar':tp/n1,'frr':1-tp/n1,'heldout_scores':fileinfo(source),'condition':'enroll3s_probe1.25s_0dB'}
    groups=[g for _,g in h.groupby('enrollment_speaker_id',sort=True)]
    block=np.array([[int(((g.label==0)&g.accepted).sum()),int((g.label==0).sum()),int(((g.label==1)&g.accepted).sum()),int((g.label==1).sum())] for g in groups])
    rng=np.random.default_rng(SEED); bootstrap=[]; degenerate=0
    while len(bootstrap)<args.bootstrap_replicates:
        sampled=block[rng.integers(0,len(block),len(block))].sum(axis=0)
        if sampled[1]==0 or sampled[3]==0: degenerate+=1; continue
        bootstrap.append([sampled[0]/sampled[1],sampled[2]/sampled[3],1-sampled[2]/sampled[3]])
    intervals={name:np.percentile(np.asarray(bootstrap)[:,i],[2.5,97.5]).tolist() for i,name in enumerate(['far','tar','frr'])}
    rates['bootstrap']={'unit':'enrollment filename speaker','seed':SEED,'valid_replicates':len(bootstrap),'degenerate_draws':degenerate,
        'percentile95_intervals':intervals,'threshold_is_frozen':True,
        'limitation':'Conditional on development threshold; excludes threshold estimation uncertainty and recurring probe-speaker dependence. Intervals unadjusted.'}
    preserve_json(out/'heldout_metrics.json',rates)
    preserve_text(out/'heldout_decisions.csv',h.to_csv(index=False,float_format='%.17g'))
    text=(f'At 3 s enrollment / 1.25 s probe and 0 dB, historical B student/student was calibrated on 10 development speakers '\
        f'excluded by the reconstructed checkpoint selector (500 target / 4,500 nontarget pairs), maximizing TAR subject to nominal FAR <= 1%. '\
        f'The frozen threshold was {threshold["threshold"]:.9g}; development FAR={threshold["far"]:.6%} and TAR={threshold["tar"]:.6%}. '\
        f'On the identical historical H cell (1937 target/1818 nontarget), achieved FAR={rates["far"]:.6%} '\
        f'({fp}/{n0}), TAR={rates["tar"]:.6%} ({tp}/{n1}), and FRR={rates["frr"]:.6%} ({n1-tp}/{n1}). '\
        f'Enrollment-speaker cluster-bootstrap 95% percentile intervals use {len(bootstrap)} draws at the frozen threshold. '\
        'These intervals condition on calibration and may miss dependence through recurring probe speakers. '
        'Development FAR resolution is 1/4500=0.02222 percentage points, but only 10 calibration speaker clusters limit independent information. '
        'Historical H noise realizations, including known pool overlap, were preserved. Selector separation follows reconstructed assignments rather than a persisted historical assignment manifest.\n')
    preserve_text(out/'manuscript_operating_point.txt',text)
    outputs={name:fileinfo(out/name) for name in ['calibration_scores.csv','threshold_record.json','heldout_metrics.json','heldout_decisions.csv','manuscript_operating_point.txt']}
    save(out/'completion.json',{'status':'completed','utc':now(),'research_question':cfg['research_question'],
        'command':[sys.executable,*sys.argv],'pid':os.getpid(),'seed':SEED,'software':software(),'device':'cuda',
        'initialization':cfg['checkpoint'],'config':fileinfo(args.prepared/'calibration_config.json'),'outputs':outputs,
        'gpu_gate':'Versioned resource wrapper and fresh primary replay gate; calibration independent of full role grid; real-data CUDA smoke passed',
        'active_processes':subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv'],text=True,capture_output=True).stdout})
    event('COMPLETED',outputs=outputs); print(json.dumps(rates,indent=2))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','run'])
    parser.add_argument('--prepared',type=Path,default=Path('/media/ml4u/Samsung_T52/ptnk_team/mortyhuaphuoc/artifacts/hc0032_revision/overnight_20261106/prompt5/prepared'))
    parser.add_argument('--run-dir',type=Path,default=HERE/'run_primary_B')
    parser.add_argument('--device',choices=['cuda'],default='cuda')
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--bootstrap-replicates',type=int,default=1000)
    parser.add_argument('--heldout-scores',type=Path)
    args=parser.parse_args()
    if args.bootstrap_replicates<1000: parser.error('At least1000 heldout rate replicates required')
    (prepare if args.mode=='prepare' else run)(args)

if __name__=='__main__': main()
