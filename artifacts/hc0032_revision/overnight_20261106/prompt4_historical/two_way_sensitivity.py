#!/usr/bin/env python3
"""Separately labeled two-way pigeonhole sensitivity; saved scores only.

Independent multinomial enrollment-speaker and probe-speaker cluster draws;
each row receives the product of its two cluster multiplicities. Every draw
is common across all fixed conditions, systems, roles and complete seeds.
"""
import argparse,json,os,sys,time,platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from paired_analysis import Cell,load_inputs,comparisons,event,csvwrite,jsonwrite,sha,historical_eer

def weighted_cell_eer(cell,row_weights):
    w=row_weights[cell.order]
    positive=w*cell.labels
    tp=np.cumsum(positive); fp=np.cumsum(w-positive)
    if tp[-1]==0 or fp[-1]==0: return np.nan
    far=np.r_[0.,fp[cell.ends]/fp[-1]]
    d=np.r_[1.,1.-tp[cell.ends]/tp[-1]-far[1:]]
    crosses=np.flatnonzero(d[:-1]*d[1:]<=0)
    if not len(crosses): raise AssertionError('No weighted EER crossing')
    i=int(crosses[0]); den=d[i]-d[i+1]; w=d[i]/den if den else 0.
    return 100*float(far[i]+w*(far[i+1]-far[i]))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--scores-long',type=Path,nargs='+')
    p.add_argument('--replicates',type=int,default=5000)
    p.add_argument('--seed',type=int,default=2026110641)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args(); out=args.output_dir.absolute()
    out.mkdir(parents=True,exist_ok=args.resume)
    if args.replicates<5000: raise ValueError('Need >=5000 sensitivity replicates')
    members,seeds,variants,provenance=load_inputs(args,out)
    baseline=next(iter(members.values()))
    enroll,enroll_idx=np.unique(baseline.enrollment.str.split('/').str[0],return_inverse=True)
    probes,probe_idx=np.unique(baseline.probe.str.split('/').str[0],return_inverse=True)
    conditions=sorted(baseline.condition.unique()); pairs=comparisons(variants)
    prepared=[]; observed=[]
    for condition in conditions:
        positions=np.flatnonzero(baseline.condition.to_numpy()==condition)
        labels=baseline.iloc[positions].label.to_numpy(np.int64)
        cells=[]
        for seed in seeds:
            seedcells=[]
            for variant in variants:
                scores=members[(seed,variant)].iloc[positions].score.to_numpy()
                cell=Cell(labels,scores,enroll_idx[positions],len(enroll))
                assert abs(weighted_cell_eer(cell,np.ones(len(labels),np.int64))-historical_eer(labels,scores))<1e-10
                seedcells.append(cell)
            cells.append(seedcells)
        prepared.append((positions,labels,cells))
    event(out,'SMOKE_PASSED',enrollment_speakers=len(enroll),probe_speakers=len(probes),method='two-way pigeonhole/product cluster weights')
    # Validate product weights against literal row replication, not same helper.
    rng=np.random.default_rng(641)
    for ci in [0,13,26]:
        positions,labels,cells=prepared[ci]
        a=rng.multinomial(len(enroll),np.full(len(enroll),1/len(enroll)))
        b=rng.multinomial(len(probes),np.full(len(probes),1/len(probes)))
        weights=a[enroll_idx[positions]]*b[probe_idx[positions]]
        for vi,variant in enumerate(variants):
            scores=members[(seeds[0],variant)].iloc[positions].score.to_numpy()
            assert abs(weighted_cell_eer(cells[0][vi],weights)-historical_eer(np.repeat(labels,weights),np.repeat(scores,weights)))<1e-10
    allvalues=[]; attempted=0; degenerates=[]; chunk=0
    while len(allvalues)<args.replicates:
        path=out/f'chunk_{chunk:04d}.npz'
        if path.exists():
            with np.load(path) as x:
                allvalues.extend(x['effects'].tolist()); attempted=int(x['next_attempt']); degenerates.extend(x['degenerate_attempts'].tolist())
            chunk+=1; continue
        values=[]; bad=[]; target=min(100,args.replicates-len(allvalues))
        while len(values)<target:
            rng=np.random.default_rng(np.random.SeedSequence([args.seed,attempted]))
            a=np.bincount(rng.integers(0,len(enroll),len(enroll)),minlength=len(enroll))
            b=np.bincount(rng.integers(0,len(probes),len(probes)),minlength=len(probes))
            ers=np.empty((len(conditions),len(seeds),len(variants)))
            for ci,(positions,labels,cells) in enumerate(prepared):
                weights=a[enroll_idx[positions]]*b[probe_idx[positions]]
                for si in range(len(seeds)):
                    for vi in range(len(variants)): ers[ci,si,vi]=weighted_cell_eer(cells[si][vi],weights)
            if not np.isfinite(ers).all(): bad.append(attempted)
            else: values.append([float((ers[:,:,a]-ers[:,:,b]).mean()) for _,a,b in pairs])
            attempted+=1
            if attempted>args.replicates*2: raise RuntimeError('Too many degenerate draws')
        with path.open('xb') as f: np.savez_compressed(f,effects=np.asarray(values),next_attempt=attempted,degenerate_attempts=np.asarray(bad,np.int64))
        allvalues.extend(values); degenerates.extend(bad); chunk+=1
        event(out,'PROGRESS',valid=len(allvalues),attempted=attempted,degenerate=len(degenerates))
    values=np.asarray(allvalues[:args.replicates]); point=np.empty((len(conditions),len(seeds),len(variants)))
    for ci,(positions,labels,cells) in enumerate(prepared):
        for si in range(len(seeds)):
            for vi in range(len(variants)): point[ci,si,vi]=weighted_cell_eer(cells[si][vi],np.ones(len(positions),np.int64))
    summaries=[]
    for pi,(name,a,b) in enumerate(pairs):
        lo,hi=np.percentile(values[:,pi],[2.5,97.5])
        summaries.append({'contrast':name,'equal_weight_27cell_difference_pp':float((point[:,:,a]-point[:,:,b]).mean()),
            'two_way_sensitivity_ci95_low_pp':float(lo),'two_way_sensitivity_ci95_high_pp':float(hi),
            'valid_replicates':len(values),'degenerate_draws':len(degenerates)})
    if not (out/'two_way_sensitivity_intervals.csv').exists(): csvwrite(out/'two_way_sensitivity_intervals.csv',pd.DataFrame(summaries))
    metadata={'status':'completed','label':'Separately labeled two-way sensitivity, not the primary interval',
        'command':[sys.executable,*sys.argv],'seed':args.seed,'input_provenance':provenance,'source_sha256':sha(__file__),
        'method':'Crossed-role pigeonhole bootstrap: independently sample enrollment-speaker and probe-speaker clusters with replacement, assign each trial the product of the two multiplicities, and reuse those draws across all 27 fixed cells/models/roles/seeds. Recompute weighted EER then equal-weight paired cell mean.',
        'limitations':['Pigeonhole bootstrap can be conservative for some crossed-cluster structures',
            'Same physical speaker appearing in different roles is not forced to have the same multiplicity; not a fully specified dyadic identity bootstrap',
            'Intervals condition on observed complete seed blocks and do not quantify population training variability',
            'Filename-derived speaker clusters retain official-label inconsistencies'],
        'valid_replicates':len(values),'attempted_draws':attempted,'degenerate_attempt_ids':degenerates,
        'seed_blocks':seeds,'device':'CPU saved-score analysis only; GPU untouched',
        'research_question':'Sensitivity of historical paired EER aggregate intervals to recurring probe-speaker dependence',
        'software':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__},
        'initialization':'none; saved-score analysis only','output_dir':str(out),
        'trial_manifest_sha256':'97cd1ff475591682c865a0e50b36e3d78d1f86d8c3f14c6e6ce7b4d8076ac83b',
        'checkpoints':'none loaded; checkpoint references in input_provenance.json',
        'raw_metrics':'immutable bootstrap chunk NPZ files and two_way_sensitivity_intervals.csv',
        'active_processes':{'analysis_pid':os.getpid(),'gpu_jobs':'untouched'},
        'logs':'stdout.log and append-only status.jsonl',
        'code_revision':'UNVERIFIED: no usable Git; current script SHA-256 recorded'}
    if not (out/'run_metadata.json').exists(): jsonwrite(out/'run_metadata.json',metadata)
    methods='This separately labeled sensitivity used a two-way pigeonhole bootstrap. Independent draws resampled enrollment-speaker and probe-speaker clusters, with each trial weighted by the product of its two cluster multiplicities. The same two draws were shared across all fixed conditions, systems, roles and observed seeds. Weighted EERs and equal-weight paired 27-cell effects were recomputed in 5,000 valid replicates. This addresses dependence through recurring probe speakers under a crossed-role clustering approximation; it is not a fully specified dyadic identity model, and may be conservative. Primary results retain the prespecified one-way enrollment-speaker interval.'
    text=['# Two-way clustering sensitivity','',methods,'','| Contrast | Mean ΔEER (pp) | Two-way sensitivity 95% CI (pp) |','|---|---:|---:|']
    for s in summaries: text.append(f'| {s["contrast"]} | {s["equal_weight_27cell_difference_pp"]:.4f} | [{s["two_way_sensitivity_ci95_low_pp"]:.4f}, {s["two_way_sensitivity_ci95_high_pp"]:.4f}] |')
    if not (out/'methods_and_table.md').exists(): (out/'methods_and_table.md').write_text('\n'.join(text)+'\n')
    event(out,'COMPLETED',summary=summaries)
if __name__=='__main__': main()
