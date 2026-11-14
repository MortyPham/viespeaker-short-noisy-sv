#!/usr/bin/env python3
"""Resumable CPU paired historical/role analyses with exact membership gates.

Writes immutable per-condition/chunk outputs; status is an append-only journal.
Only saved score files are read. No model, GPU or audio inference is performed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_curve

ROOT = Path('/media/ml4u/Samsung_T52/ptnk_team/mortyhuaphuoc')
REGISTRY = ROOT/'artifacts/hc0032_revision/prompt1_20261106_2153/final/model_registry.json'
KEY=['condition','trial_id','label','enrollment','probe']

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def jsonwrite(path,obj):
    with path.open('x') as f: json.dump(obj,f,indent=2,default=str)

def csvwrite(path,frame):
    with path.open('x') as f: frame.to_csv(f,index=False,float_format='%.17g')

def historical_eer(labels,scores,weight=None):
    far,tar,_=roc_curve(labels,scores,pos_label=1,sample_weight=weight)
    d=1-tar-far
    crossing=np.flatnonzero(d[:-1]*d[1:]<=0)
    if len(crossing):
        i=int(crossing[0]); den=d[i]-d[i+1]; w=d[i]/den if den else 0.
        return 100*float(far[i]+w*(far[i+1]-far[i]))
    i=int(np.argmin(np.abs(d))); return 50*float(1-tar[i]+far[i])

class Cell:
    def __init__(self,labels,scores,speakers,cluster_count):
        self.order=np.argsort(scores,kind='stable')[::-1]
        ordered_scores=scores[self.order]
        self.ends=np.r_[np.flatnonzero(np.diff(ordered_scores)),len(labels)-1]
        self.labels=labels[self.order].astype(np.int64)
        self.speakers=speakers[self.order]
        self.cluster_count=cluster_count
    def eer(self,multiplicities):
        w=multiplicities[self.speakers]
        positives=w*self.labels
        tp=np.cumsum(positives); fp=np.cumsum(w-positives)
        if tp[-1]==0 or fp[-1]==0: return np.nan
        far=np.r_[0.,fp[self.ends]/fp[-1]]
        d=np.r_[1.,1.-tp[self.ends]/tp[-1]-far[1:]]
        crosses=np.flatnonzero(d[:-1]*d[1:]<=0)
        if not len(crosses):
            i=int(np.argmin(np.abs(d))); return 50*float(d[i]+2*far[i])
        i=int(crosses[0]); den=d[i]-d[i+1]; weight=d[i]/den if den else 0.
        return 100*float(far[i]+weight*(far[i+1]-far[i]))

def load_inputs(args,out):
    provenance={}; members={}; allframes=[]
    if args.scores_long:
        paths=args.scores_long
        for p in paths:
            df=pd.read_csv(p,float_precision='round_trip')
            needed={'seed','system','enrollment_role','probe_role',*KEY,'score'}
            if not needed.issubset(df): raise ValueError(f'Missing columns: {needed-set(df)}')
            allframes.append(df)
            provenance[str(p)]={'sha256':sha(p),'rows':len(df)}
        long=pd.concat(allframes,ignore_index=True)
        long['variant']=long.system.astype(str)+'/'+long.enrollment_role.astype(str)+'/'+long.probe_role.astype(str)
        for (seed,v),g in long.groupby(['seed','variant']): members[(int(seed),v)]=g
    else:
        reg=json.loads(REGISTRY.read_text())
        provenance[str(REGISTRY)]={'sha256':sha(REGISTRY)}
        for m in ['B','C','D']:
            cfg=reg['models'][m]; p=Path(cfg['score_file']['path'])
            h=sha(p)
            if h!=cfg['score_file']['sha256']: raise ValueError(f'Archived score hash changed: {p}')
            g=pd.read_csv(p,float_precision='round_trip')
            role='student' if m=='B' else 'teacher'
            members[(2026,f'{m}/{role}/student')]=g
            provenance[str(p)]={'sha256':h,'rows':len(g),'checkpoint':cfg['checkpoint'],
                'checkpoint_selection':cfg['selection'],'enrollment_role':role,'probe_role':'student'}
    for k,g in members.items():
        if g.duplicated(['condition','trial_id']).any(): raise ValueError(f'Duplicate keys: {k}')
        if not np.isfinite(g.score).all(): raise ValueError(f'Nonfinite scores: {k}')
        if not set(g.label.unique())=={0,1}: raise ValueError(f'Invalid labels: {k}')
        for c in ['enrollment','probe']:
            if not g[c].str.split('/').str[0].str.fullmatch(r'id\d+').all(): raise ValueError('Speaker ID parse failure')
        if g.condition.nunique()!=27 or len(g)!=101610: raise ValueError(f'Unexpected grid population: {k}')
        if 'enrollment_speaker_id' in g:
            if not (g.enrollment_speaker_id==g.enrollment.str.split('/').str[0]).all(): raise ValueError('Enrollment ID mismatch')
        if 'probe_speaker_id' in g:
            if not (g.probe_speaker_id==g.probe.str.split('/').str[0]).all(): raise ValueError('Probe ID mismatch')
    baseline=next(iter(members.values()))[KEY]
    diagnostics=[]
    for k,g in members.items():
        # Outer join on the true key verifies membership and compares all labels/pairs.
        check=baseline.merge(g[KEY],on=['condition','trial_id'],how='outer',suffixes=('_reference','_candidate'),indicator=True,validate='one_to_one')
        mismatch=check._merge.ne('both')
        for c in ['label','enrollment','probe']: mismatch |= check[c+'_reference'].ne(check[c+'_candidate'])
        if mismatch.any():
            csvwrite(out/f'membership_failure_{k[0]}_{k[1].replace("/","_")}.csv',check.loc[mismatch])
            raise ValueError(f'Exact membership/label/speaker mismatch: {k}; no silent intersection permitted')
        diagnostics.append({'seed':k[0],'variant':k[1],'rows':len(g),'target':int(g.label.sum()),'nontarget':int((g.label==0).sum()),'exact_match':True})
        members[k]=g.sort_values(['condition','trial_id']).reset_index(drop=True)
    # Both arrays are explicitly sorted only AFTER exact joins/label/recording checks.
    seeds=sorted({k[0] for k in members}); variants=sorted({k[1] for k in members})
    for seed in seeds:
        if {v for s,v in members if s==seed}!=set(variants): raise ValueError(f'Incomplete seed block {seed}')
    if not (out/'input_provenance.json').exists(): jsonwrite(out/'input_provenance.json',provenance)
    if not (out/'membership_audit.csv').exists(): csvwrite(out/'membership_audit.csv',pd.DataFrame(diagnostics))
    return members,seeds,variants,provenance

def comparisons(variants):
    candidates=[('C_student_minus_B','C/student/student','B/student/student'),
        ('D_student_minus_B','D/student/student','B/student/student'),
        ('D_student_minus_C_student','D/student/student','C/student/student'),
        ('C_teacher_minus_C_student','C/teacher/student','C/student/student'),
        ('D_teacher_minus_D_student','D/teacher/student','D/student/student'),
        ('C_paper_minus_B','C/teacher/student','B/student/student'),
        ('D_paper_minus_B','D/teacher/student','B/student/student'),
        ('D_paper_minus_C_paper','D/teacher/student','C/teacher/student')]
    return [(name,variants.index(a),variants.index(b)) for name,a,b in candidates if a in variants and b in variants]

def event(out,state,**kw):
    with (out/'status.jsonl').open('a') as f: f.write(json.dumps({'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'pid':os.getpid(),'state':state,**kw})+'\n')
    print(state,kw,flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--scores-long',type=Path,nargs='+')
    p.add_argument('--cohort',choices=['matched_seeds'],default='matched_seeds')
    p.add_argument('--aggregate-replicates',type=int,default=5000)
    p.add_argument('--cell-replicates',type=int,default=1000)
    p.add_argument('--aggregate-seed',type=int,default=2026110604)
    p.add_argument('--cell-seed',type=int,default=20260927)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke-only',action='store_true')
    args=p.parse_args()
    if not args.scores_long:p.error('Matched cohort requires explicit saved score files')
    out=args.output_dir.absolute()
    if out.exists() and not args.resume: raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=args.resume)
    if args.aggregate_replicates<5000: raise ValueError('At least 5000 valid aggregate replicates required')
    event(out,'STARTING',command=[sys.executable,*sys.argv],cohort=args.cohort)
    members,seeds,variants,provenance=load_inputs(args,out)
    assert set(variants)=={'B/student/student','C/student/student','D/student/student'},variants
    assert all(seed in [2027,2028] for seed in seeds),seeds
    baseline=next(iter(members.values()))
    speakers,inverse=np.unique(baseline.enrollment.str.split('/').str[0],return_inverse=True)
    k=len(speakers); condition_names=sorted(baseline.condition.unique()); pairs=comparisons(variants)
    if not pairs: raise ValueError('No supported paired contrasts available')
    cells={}; conditions={}; observed=[]; counts=[]
    for ci,condition in enumerate(condition_names):
        positions=np.flatnonzero(baseline.condition.to_numpy()==condition)
        group=baseline.iloc[positions]; labels=group.label.to_numpy(np.int64)
        enrollment=inverse[positions]
        conditions[ci]=(positions,enrollment,labels)
        counts.append({'condition':condition,'rows':len(group),'target':int(labels.sum()),'nontarget':int((labels==0).sum()),'excluded_candidates':5000-len(group),'enrollment_speakers':len(np.unique(enrollment))})
        for si,seed in enumerate(seeds):
            for vi,variant in enumerate(variants):
                scores=members[(seed,variant)].iloc[positions].score.to_numpy(np.float64)
                cell=Cell(labels,scores,enrollment,k); cells[(ci,si,vi)]=cell
                value=cell.eer(np.ones(k,dtype=np.int64))
                reference=historical_eer(labels,scores)
                if abs(value-reference)>1e-10: raise ValueError(f'Weighted metric mismatch {condition}: {value}/{reference}')
                observed.append({'seed':seed,'variant':variant,'condition':condition,'eer_percent':value})
    # Meaningful independent smoke check: weighted implementation against literal duplicated-row ROC.
    rng_test=np.random.default_rng(5304)
    for ci in [0,13,26]:
        positions,enrollment,labels=conditions[ci]
        for _ in range(10):
            mult=rng_test.multinomial(k,np.full(k,1/k)); weights=mult[enrollment]
            for vi,variant in enumerate(variants):
                scores=members[(seeds[0],variant)].iloc[positions].score.to_numpy(np.float64)
                expected=historical_eer(np.repeat(labels,weights),np.repeat(scores,weights))
                value=cells[(ci,0,vi)].eer(mult)
                if abs(value-expected)>1e-10: raise ValueError('Weighted EER does not match duplicated-row bootstrap')
    event(out,'SMOKE_PASSED',metric_checks=90,global_enrollment_speakers=k)
    if args.smoke_only:
        jsonwrite(out/'smoke_result.json',{'status':'passed','metric_checks':90,'enrollment_speakers':k,'input_membership':'verified by exact outer joins','command':[sys.executable,*sys.argv]})
        return
    observed=pd.DataFrame(observed)
    if not (out/'observed_eers.csv').exists(): csvwrite(out/'observed_eers.csv',observed)
    if not (out/'condition_counts.csv').exists(): csvwrite(out/'condition_counts.csv',pd.DataFrame(counts))
    point=np.empty((len(condition_names),len(seeds),len(pairs)))
    point_rows=[]
    for ci,condition in enumerate(condition_names):
        for si,seed in enumerate(seeds):
            vals=[cells[(ci,si,vi)].eer(np.ones(k,dtype=np.int64)) for vi in range(len(variants))]
            for pi,(name,a,b) in enumerate(pairs):
                effect=vals[a]-vals[b]; point[ci,si,pi]=effect
                point_rows.append({'cohort':args.cohort,'seed':seed,'condition':condition,'contrast':name,'lhs':variants[a],'rhs':variants[b],'difference_pp':effect,**{key:value for key,value in counts[ci].items() if key!='condition'}})
    if not (out/'per_cell_effects.csv').exists(): csvwrite(out/'per_cell_effects.csv',pd.DataFrame(point_rows))
    if not (out/'primary_effects.csv').exists(): csvwrite(out/'primary_effects.csv',pd.DataFrame(point_rows).loc[lambda d:d.condition.str.startswith('enroll3s_probe1.25s_')])
    perseed=[]
    for si,seed in enumerate(seeds):
        for pi,(name,a,b) in enumerate(pairs):
            perseed.append({'cohort':args.cohort,'seed':seed,'contrast':name,'equal_weight_27cell_difference_pp':float(point[:,si,pi].mean())})
    if not (out/'per_seed_effects.csv').exists(): csvwrite(out/'per_seed_effects.csv',pd.DataFrame(perseed))
    # Cellwise historical convention: independent draws within each fixed cell, paired systems.
    # np.integers and one sequential RNG reproduce the saved algorithm's actual draws.
    cellrng=np.random.default_rng(args.cell_seed)
    cell_intervals=[]
    for ci,condition in enumerate(condition_names):
        cluster_ids=np.unique(conditions[ci][1]); local_k=len(cluster_ids)
        draws=cellrng.integers(0,local_k,size=(args.cell_replicates,local_k))
        output=out/f'cell_bootstrap_{ci:02d}.npz'
        if output.exists():
            with np.load(output) as stored: values=stored['effects']; degenerate=int(stored['degenerate'])
        else:
            values=[]; degenerate=0
            for draw in draws:
                mult=np.zeros(k,dtype=np.int64); mult[cluster_ids]=np.bincount(draw,minlength=local_k)
                ers=np.array([[cells[(ci,si,vi)].eer(mult) for vi in range(len(variants))] for si in range(len(seeds))])
                if not np.isfinite(ers).all(): degenerate+=1; continue
                values.append([float((ers[:,a]-ers[:,b]).mean()) for _,a,b in pairs])
            values=np.asarray(values)
            with output.open('xb') as f: np.savez_compressed(f,effects=values,degenerate=degenerate)
        if len(values)<0.9*args.cell_replicates: raise ValueError('Insufficient cellwise valid draws')
        for pi,(name,_,_) in enumerate(pairs):
            lo,hi=np.percentile(values[:,pi],[2.5,97.5])
            cell_intervals.append({'condition':condition,'contrast':name,'difference_pp':float(point[ci,:,pi].mean()),'ci95_low_pp':float(lo),'ci95_high_pp':float(hi),'valid_replicates':len(values),'degenerate_draws':degenerate,'interval':'unadjusted percentile, paired enrollment-speaker clusters per cell'})
        event(out,'CELLWISE_COMPLETE',condition=condition,valid=len(values),degenerate=degenerate)
    if not (out/'cellwise_intervals.csv').exists(): csvwrite(out/'cellwise_intervals.csv',pd.DataFrame(cell_intervals))
    # Joint aggregate: exactly ONE common speaker multiplicity vector per draw, applied
    # to all 27 fixed cells, all models, roles and observed complete seeds.
    aggregate=[]; attempted=0; valid=0; chunk=0; degenerates=[]
    while valid<args.aggregate_replicates:
        path=out/f'aggregate_chunk_{chunk:04d}.npz'
        if path.exists():
            with np.load(path) as stored:
                values=stored['effects']; next_attempt=int(stored['next_attempt']); bad=stored['degenerate_attempts'].tolist()
            aggregate.extend(values.tolist()); valid+=len(values); attempted=next_attempt; degenerates.extend(bad); chunk+=1
            continue
        values=[]; bad=[]
        target=min(100,args.aggregate_replicates-valid)
        while len(values)<target:
            rng=np.random.default_rng(np.random.SeedSequence([args.aggregate_seed,attempted]))
            draw=rng.integers(0,k,k); mult=np.bincount(draw,minlength=k)
            ers=np.empty((len(condition_names),len(seeds),len(variants)))
            for ci in range(len(condition_names)):
                for si in range(len(seeds)):
                    for vi in range(len(variants)): ers[ci,si,vi]=cells[(ci,si,vi)].eer(mult)
            if not np.isfinite(ers).all(): bad.append(attempted)
            else: values.append([float((ers[:,:,a]-ers[:,:,b]).mean()) for _,a,b in pairs])
            attempted+=1
            if attempted>args.aggregate_replicates*2: raise RuntimeError('Too many degenerate joint draws')
        values=np.asarray(values)
        with path.open('xb') as f: np.savez_compressed(f,effects=values,next_attempt=attempted,degenerate_attempts=np.asarray(bad,dtype=np.int64))
        aggregate.extend(values.tolist()); valid+=len(values); degenerates.extend(bad); chunk+=1
        event(out,'AGGREGATE_PROGRESS',valid=valid,attempted=attempted,degenerate=len(degenerates))
    aggregate=np.asarray(aggregate[:args.aggregate_replicates]); summaries=[]
    for pi,(name,a,b) in enumerate(pairs):
        lo,hi=np.percentile(aggregate[:,pi],[2.5,97.5]); seedvalues=point[:,:,pi].mean(axis=0)
        summaries.append({'cohort':args.cohort,'contrast':name,'lhs':variants[a],'rhs':variants[b],
            'equal_weight_27cell_difference_pp':float(point[:,:,pi].mean()),'ci95_low_pp':float(lo),'ci95_high_pp':float(hi),
            'complete_observed_seeds':seeds,'between_seed_sample_sd_pp':float(np.std(seedvalues,ddof=1)) if len(seeds)>1 else None,
            'between_seed_min_pp':float(seedvalues.min()),'between_seed_max_pp':float(seedvalues.max()),
            'valid_joint_replicates':len(aggregate),'degenerate_draws':len(degenerates),
            'interpretation':'test-trial uncertainty conditional on observed complete seeds; not population training-seed uncertainty'})
    if not (out/'aggregate_intervals.csv').exists(): csvwrite(out/'aggregate_intervals.csv',pd.DataFrame(summaries))
    metadata={'status':'completed','research_question':'Paired EER effects and test-trial uncertainty for HC0032 '+args.cohort,
        'command':[sys.executable,*sys.argv],'cwd':str(ROOT),'pid':os.getpid(),'device':'cpu analysis of saved scores only',
        'seed':{'cellwise':args.cell_seed,'joint_aggregate':args.aggregate_seed},'initialization':'none; no model loaded',
        'software':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__},
        'source_sha256':sha(__file__),'code_revision':'UNVERIFIED: no usable Git; current source hash recorded',
        'data_paths_and_hashes':provenance,'trial_manifest_sha256':'97cd1ff475591682c865a0e50b36e3d78d1f86d8c3f14c6e6ce7b4d8076ac83b',
        'output_dir':str(out),'checkpoints':'No checkpoints loaded; provenance records fresh selected checkpoint hashes in saved scores',
        'logs':'stdout.log and status.jsonl','raw_metrics':'observed_eers.csv; per_cell_effects.csv; immutable bootstrap NPZ chunks',
        'active_processes':{'analysis_pid':os.getpid(),'gpu_jobs':'Untouched; CPU-only analysis'},
        'method':'Fixed 27 cells, equal-weight mean paired EER differences; common enrollment-speaker multiplicities across all conditions/models/roles/seeds in each aggregate draw',
        'cellwise_convention':'1000 paired enrollment-speaker draws per condition; saved sequential RNG convention seed 20260927; unadjusted percentile 95% intervals',
        'valid_aggregate_replicates':len(aggregate),'attempted_aggregate_draws':attempted,'degenerate_attempt_ids':degenerates,
        'seed_blocks':seeds,'variants':variants,'limitations':['One-way enrollment clustering may miss dependence through recurring probe speakers',
            'No dyadic/two-way sensitivity estimated','An interval including zero does not establish equivalence',
            'Historical 2026 scores are separate from a matched training cohort','Noise-pool overlap retained as documented in Prompt1 audit']}
    if not (out/'run_metadata.json').exists(): jsonwrite(out/'run_metadata.json',metadata)
    methods='For the observed complete fresh matched seed blocks, paired comparisons used exactly the same 101,610 condition/trial rows, with labels and recording-derived speaker IDs verified by outer joins. EER followed the historical first FAR/FRR-crossing linear interpolation. We report signed EER differences in percentage points, where negative differences favor the first system. The aggregate is the equal-weight mean across the fixed 27 conditions. For its 95% percentile interval, 5,000 valid bootstrap replicates resampled enrollment-speaker clusters with replacement using one common speaker draw for all conditions and systems. Cellwise intervals used the saved 1,000-replicate convention and are unadjusted for multiplicity. These intervals quantify test-trial uncertainty conditional on the observed complete training seeds; they do not quantify population training-seed variability. Enrollment-only clustering may miss dependence through recurring probe speakers. An interval containing zero does not demonstrate equivalence.'
    table=['# Fresh matched HC0032 student/student paired effects','', '**VERIFIED:** fresh matched student/student score analysis; seeds2027/2028kept separate from historical2026. No model inference is performed by this analysis script.','',
        '| Contrast | Mean ΔEER (pp) | Joint 95% CI (pp) |','|---|---:|---:|']
    for s in summaries: table.append(f'| {s["contrast"]} | {s["equal_weight_27cell_difference_pp"]:.4f} | [{s["ci95_low_pp"]:.4f}, {s["ci95_high_pp"]:.4f}] |')
    table.extend(['',methods,'','Full per-cell, primary-noise and per-seed point estimates and machine-readable intervals are saved alongside this table. All B/C/D systems use student enrollment and student probes on identical scored rows. Between-seed sample SD/range describe training-seed variation separately from evaluation bootstrap uncertainty; two seeds do not establish population-wide variability.'])
    if not (out/'manuscript_table_and_methods.md').exists(): (out/'manuscript_table_and_methods.md').write_text('\n'.join(table)+'\n')
    event(out,'COMPLETED',valid_aggregate_replicates=len(aggregate),aggregate_summaries=summaries)

if __name__=='__main__':
    try: main()
    except Exception as exc:
        print(f'FAILED {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        raise
