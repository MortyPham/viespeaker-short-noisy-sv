"""Non-destructive HC0032 Prompt 1 audit; outputs are exclusive-create only.

Runs saved-score analysis, not raw-checkpoint inference. Uses the historical
metric function directly from its AST without importing GPU/audio dependencies.
"""
import ast
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent
EV = ROOT / 'experiments/evaluation'
KEY = ['condition', 'trial_id', 'label', 'enrollment', 'probe']
NAMES = dict(A='ecapa_pretrained_baseline', B='aam_only_full', C='c_vector_ema', D='d_full')
RUNS = dict(B='m0_aam_only_full_dataset', C='m2_legacy_vector_ema', D='m3_legacy_ema_gaussian_kl')
INPUTS = {}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def record(path):
    path = Path(path).absolute()
    v = {'path': str(path), 'exists': path.is_file()}
    if v['exists']: v.update(sha256=sha(path), bytes=path.stat().st_size)
    INPUTS[str(path)] = v
    return v

def read(path):
    record(path)
    return json.loads(Path(path).read_text())

def save(name, obj):
    with (OUT / name).open('x') as f: json.dump(obj, f, indent=2, default=str)

def csv(name, frame):
    with (OUT / name).open('x') as f: frame.to_csv(f, index=False)

def relocated(s):
    return str(s).replace('/media/ml4u/Samsung_T51/ptnk_team/mortyhuaphuoc', str(ROOT))

def historical_dir_hash(path):
    h = hashlib.sha256()
    for p in sorted(p for p in path.rglob('*') if p.is_file()):
        h.update(p.relative_to(path).as_posix().encode())
        with p.open('rb') as f:
            for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

source = ROOT / 'evaluate_aam_only_full_viespeaker_h_1e1p.py'
tree = ast.parse(source.read_text())
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'verification_metrics')
scope = dict(np=np, roc_curve=roc_curve, roc_auc_score=roc_auc_score)
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
metric = scope['verification_metrics']
assert abs(metric(np.array([0,0,1,1]), np.array([0., .1, .9, 1.]))['eer_percent']) < 1e-12
record(source)
checkpoint_meta = read(OUT / 'checkpoint_inspection.json')
models, frames, protocols, quality, recipes = {}, {}, {}, {}, {}
counts, metrics, diffs, eligibility = [], [], [], []
for model, name in NAMES.items():
    base = EV / f'{name}_viespeaker_h_1e1p/final_seed20260818_n5000'
    meta = read(base / 'run_metadata.json')
    protocols[model] = read(base / 'protocol_config.json')
    scorepath = base / f'scores_{name}.csv'
    scoreinfo = record(scorepath)
    frame = pd.read_csv(scorepath, float_precision='round_trip')
    assert not frame.duplicated(['condition', 'trial_id']).any()
    assert np.isfinite(frame.score).all()
    frames[model] = frame
    trialinfo = record(base / 'fixed_trials.csv')
    frozen = pd.read_csv(base / 'fixed_trials.csv')
    match = frame.merge(frozen[['trial_id','label','enrollment','probe']], on=['trial_id'], suffixes=('', '_frozen'), validate='many_to_one')
    assert len(match) == len(frame)
    assert all((match[c] == match[c+'_frozen']).all() for c in ['label','enrollment','probe'])
    record(base / 'audio_quality.csv')
    quality[model] = pd.read_csv(base / 'audio_quality.csv')
    record(base / 'probe_transform_manifest.csv')
    recipes[model] = pd.read_csv(base / 'probe_transform_manifest.csv').fillna('').map(relocated)
    aux = {f: record(base / f) for f in ['audio_preparation_failures.csv','embedding_failures.csv','valid_trials_by_condition.csv','extraction_summary.json','summary_metrics.csv','embedding_cache_index.json']}
    cache = read(base / 'embedding_cache_index.json')
    cache_paths = {k: {'archived_path': v, 'current_path': relocated(v), 'exists': Path(relocated(v)).is_dir()} for k,v in cache.items()}
    for condition, group in frame.groupby('condition', sort=True):
        target = int(group.label.sum())
        counts.append(dict(model=model, condition=condition, rows=len(group), target=target, nontarget=len(group)-target, excluded=5000-len(group)))
        m = metric(group.label.to_numpy(), group.score.to_numpy())
        metrics.append(dict(model=model, condition=condition, **m))
        e, p = float(group.enrollment_duration_s.iloc[0]), float(group.probe_duration_s.iloc[0])
        q = quality[model].set_index('path').voiced_seconds
        eligible = frozen.enrollment.map(q).ge(e-1e-6) & frozen.probe.map(q).ge(p-1e-6)
        expected = set(frozen.loc[eligible, 'trial_id']); actual = set(group.trial_id)
        eligibility.append(dict(model=model, condition=condition, duration_eligible=len(expected), scored=len(actual), eligible_unscored=sorted(expected-actual), scored_ineligible=sorted(actual-expected)))
    models[model] = {'status': 'UNVERIFIED: raw-checkpoint replay and parent identity gates pending',
        'paper_id': model, 'historical_metadata': meta, 'score_file': scoreinfo,
        'trial_manifest': trialinfo, 'archived_inputs': aux, 'embedding_cache': cache_paths,
        'enrollment_role': 'pretrained' if model=='A' else 'student' if model=='B' else 'teacher',
        'probe_role': 'pretrained' if model=='A' else 'student', 'launch_enabled': False}
    if model in RUNS:
        runbase = ROOT / 'experiments/mentor_stage' / RUNS[model]
        ckinfo = checkpoint_meta['models'][model]
        ckfile = record(ckinfo['path'])
        summary = read(runbase/'training_summary.json')
        history = read(runbase/'validation_history.json')
        def noise(h): return h.get('metrics', h).get('noise_eer', h.get('validation/noise_eer'))
        best = min(history, key=noise)
        sig = ckinfo['signature']
        (OUT / f'{model}_training_speakers.txt').write_text('\n'.join(sig.get('train_speaker_ids', []))+'\n')
        models[model].update(checkpoint=ckfile, checkpoint_metadata=ckinfo, training_summary=summary,
            resolved_config=sig, selection={'rule': 'minimum noisy development EER', 'minimum_step': best['step'],
                'minimum_eer': noise(best), 'checkpoint_matches_minimum': ckinfo['step']==best['step'] and abs(ckinfo['best_noise_eer']-noise(best))<1e-14,
                'validation_steps': [h['step'] for h in history]},
            archive_checkpoint_hash_matches=ckfile['sha256']==meta['checkpoint_sha256'])
    else:
        pre = Path(meta['pretrained_dir'])
        models[model]['pretrained_bundle'] = {'path': str(pre), 'historical_directory_sha256': historical_dir_hash(pre),
            'archived_hash_matches': historical_dir_hash(pre)==meta['pretrained_dir_sha256'],
            'files': [record(p) for p in sorted(pre.iterdir()) if p.is_file()]}
    print('Audited', model, len(frame), flush=True)

# Never silently replace the scored population with an intersection.
for model in NAMES:
    a = frames['B'][KEY]; b = frames[model][KEY]
    d = a.merge(b, on=KEY, how='outer', indicator=True, validate='one_to_one')
    for _, row in d.loc[d._merge != 'both'].iterrows():
        diffs.append({'compared_to': 'B', 'model': model, **row.to_dict()})
csv('population_differences.csv', pd.DataFrame(diffs, columns=['compared_to','model']+KEY+['_merge']))
csv('condition_counts.csv', pd.DataFrame(counts))
csv('archived_metrics_recomputed.csv', pd.DataFrame(metrics))
save('eligibility_audit.json', eligibility)

merged = None
for model, frame in frames.items():
    f = frame[KEY+['score']].rename(columns={'score':model})
    merged = f if merged is None else merged.merge(f, on=KEY, how='inner', validate='one_to_one')
csv('diagnostic_common_population.csv', merged[KEY])
common_metrics = []
for condition, g in merged.groupby('condition'):
    for model in NAMES:
        common_metrics.append(dict(condition=condition, model=model, **metric(g.label.to_numpy(), g[model].to_numpy())))
csv('diagnostic_common_metrics.csv', pd.DataFrame(common_metrics))

ms = pd.DataFrame(metrics)
pivot = ms.pivot(index='condition', columns='model', values='eer_percent')
means = pivot.mean().to_dict()
wins = pivot[['B','C','D']].idxmin(axis=1).value_counts().to_dict()
expected = {'clean': [20.031,12.700,12.816,13.113], '10dB':[24.037,15.436,15.694,15.746], '0dB':[32.783,23.817,23.982,24.161]}
table = []
for noise, values in expected.items():
    for model, value in zip(NAMES, values):
        actual = float(pivot.loc[f'enroll3s_probe1.25s_{noise}',model])
        table.append(dict(model=model,noise=noise,expected_percent=value,archived_recomputed_percent=actual,
            delta_pp=actual-value, matches_paper_rounding=abs(actual-value)<=0.0005+1e-12,
            raw_checkpoint_replay_status='BLOCKED: shared GPU occupied; parent provenance unresolved'))
csv('table1_archived_check.csv', pd.DataFrame(table))

splitpath = ROOT/'experiments/shared/speaker_splits.csv'
recpath = ROOT/'experiments/shared/d_full_audio_cache/recordings.csv'
splitinfo, recinfo = record(splitpath), record(recpath)
split = pd.read_csv(splitpath)
assert not split.speaker_id.duplicated().any()
splits = {s:set(g.speaker_id) for s,g in split.groupby('split')}
rec = pd.read_csv(recpath)
speaker_checks = {}
for model in RUNS:
    ids = set(models[model]['resolved_config']['train_speaker_ids'])
    h_ids = set(frames[model].enrollment.str.split('/').str[0]) | set(frames[model].probe.str.split('/').str[0])
    speaker_checks[model] = {'train_speakers':len(ids), 'train_outside_split':sorted(ids-splits['train']),
        'train_validation_overlap':sorted(ids & splits['validation']), 'train_test_overlap':sorted(ids & splits['test']),
        'H_scored_filename_speakers_overlapping_train':sorted(ids & h_ids),
        'signature_split_hash_matches':models[model]['resolved_config'].get('speaker_split_sha256')==splitinfo['sha256'],
        'signature_recordings_hash_matches':models[model]['resolved_config'].get('recording_manifest_sha256')==recinfo['sha256']}
save('speaker_split_audit.json', speaker_checks)

parities = {}
for model in NAMES:
    qa, qb = quality['B'].sort_values('path').reset_index(drop=True), quality[model].sort_values('path').reset_index(drop=True)
    for q in [qa,qb]: q['status'] = q.status.replace({'prepared':'terminal','cached':'terminal'})
    parities[model] = {'fixed_trial_hash_matches_B': models[model]['trial_manifest']['sha256']==models['B']['trial_manifest']['sha256'],
        'quality_semantic_equal':qa.equals(qb),
        'transform_recipes_equal_B': recipes['B'].equals(recipes[model])}

for p in ROOT.glob('*.py'): record(p)
for p in (ROOT/'docs/research').glob('*.md'): record(p)
record(ROOT/'AGENTS.md')
protocol = {'status':'UNVERIFIED: launch gates closed', 'seed':20260818,
    'requested_trial_count':5000, 'expected_complete_grid_rows':101610,
    'observed_grid_rows': {m:len(f) for m,f in frames.items()},
    'common_grid_rows_diagnostic_only':len(merged), 'exact_all_system_membership':not diffs,
    'condition_count':{m:int(f.condition.nunique()) for m,f in frames.items()},
    'all_trial_labels_and_recording_keys_match_frozen':True,
    'parity':parities,'archived_protocols':protocols,
    'training_split':splitinfo,'training_recordings':recinfo,
    'sample_rate':16000, 'vad':'Historical Silero threshold .50, speech/silence 100ms, padding 30ms; exact historical package version UNVERIFIED',
    'scorer':{'source':record(source),'function':'verification_metrics', 'score':'cosine of L2-normalized embeddings',
        'paper_rounding_tolerance_pp':0.0005, 'raw_score_replay_tolerance':'UNVERIFIED: set only after smoke test'},
    'archived_means_percent':means,'expected_means_percent':dict(zip(NAMES,[26.142,18.389,18.585,18.725])),
    'archived_BCD_wins':{m:wins.get(m,0) for m in ['B','C','D']},'expected_BCD_wins':{'B':24,'C':3,'D':0}}
save('protocol_registry.json',protocol)
save('model_registry.json', {'status':'provisional, all launch gates closed', 'models':models,
    'allowlist':{'model_ids':list(NAMES),'training_allowed':False,'evaluation_allowed':False,
        'require_exact_checkpoint_hash':True,'require_exact_protocol_and_scored_population':True,
        'reason':'Resolve population, parent provenance, strict role loading and replay before dependent launches'},
    'excluded':['ARCH_A_ORIGINAL','ARCH_B_GLOBAL_LOCAL','ARCH_C_RELATIVE_LENGTH_GATE','V2 3359-speaker models','subset runs','other backbones','broad-H','VoxCeleb evaluations','deployment checkpoints']})
save('input_inventory.json', INPUTS)
software = {}
for pkg in ['numpy','pandas','scikit-learn','torch','torchaudio','speechbrain','silero-vad']:
    try: software[pkg]=importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError: software[pkg]=None
git = subprocess.run(['git','status','--porcelain=v1'],capture_output=True,text=True)
save('run_metadata.json',{'research_question':'Resolve historical HC0032 A/B/C/D identities and reproduce reference scores',
    'status':'completed independent forensic/archived-score checks; raw-checkpoint replay blocked',
    'command':[sys.executable,*sys.argv], 'cwd':str(ROOT),'output_dir':str(OUT),
    'time_utc':datetime.now(timezone.utc).isoformat(),'seed':'no random computation; historical train 2026, trial 20260818',
    'initialization':'none: CPU metadata and archived-score analysis only',
    'software':software,'python':platform.python_version(),'platform':platform.platform(),
    'git':{'returncode':git.returncode,'stdout':git.stdout,'stderr':git.stderr},
    'code_revision':'UNVERIFIED: no usable Git repository; source hashes in input_inventory.json',
    'active_processes':'See host_preflight.txt; existing processes untouched',
    'checkpoints':{m:models[m].get('checkpoint') for m in RUNS},
    'trial_manifest_sha256':models['B']['trial_manifest']['sha256'],
    'logs':['audit.log','checkpoint_inspection.log','host_preflight.txt'],
    'raw_metrics':'archived_metrics_recomputed.csv; no fresh raw-checkpoint scores',
    'limitations':['Only Prompt 1 supplied; Prompts 2–5 absent','Submitted PDF/reviewer form not supplied',
        'CPU checkpoint inspection does not verify strict model loading in eval mode',
        'Historical parent file hashes absent from training signatures',
        'Individual score exports and diagnostic intersection are explicitly separate']})
print(json.dumps({'rows':protocol['observed_grid_rows'],'common_rows':len(merged),'means':means,'wins':wins,'differences':len(diffs),'parity':parities},indent=2))
