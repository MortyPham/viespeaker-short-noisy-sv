"""Paired enrollment-speaker bootstrap for validated within-C/D role scores."""
import argparse,datetime,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
N=Path(__file__).resolve().parent
SEED=20261107
def load(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,default=str)
def csv(p,x):
    with Path(p).open('x') as f:x.to_csv(f,index=False,float_format='%.17g')
def prepared(labels,scores,cluster):
    order=np.argsort(-scores,kind='stable');s=scores[order]
    ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
    return labels[order],cluster[order],ends
def weighted_eer(plan,counts):
    y,cluster,ends=plan;w=counts[cluster]
    tp=np.cumsum(y*w)[ends];fp=np.cumsum((1-y)*w)[ends]
    if tp[-1]==0 or fp[-1]==0:return float('nan')
    keep=np.diff(np.r_[0,tp+fp])>0;tp=tp[keep];fp=fp[keep]
    far=np.r_[0,fp/fp[-1]];frr=np.r_[1,1-tp/tp[-1]];diff=frr-far
    cross=np.flatnonzero(diff[:-1]*diff[1:]<=0)
    if len(cross):
        i=int(cross[0]);denom=diff[i]-diff[i+1];w=diff[i]/denom if denom else 0.
        return float(100*(far[i]+w*(far[i+1]-far[i])))
    i=int(np.argmin(abs(diff)));return float(50*(frr[i]+far[i]))
def reference_eer(y,s,w=None):
    far,tar,_=roc_curve(y,s,sample_weight=w);frr=1-tar;d=frr-far
    cross=np.flatnonzero(d[:-1]*d[1:]<=0)
    if len(cross):
        i=int(cross[0]);denom=d[i]-d[i+1];a=d[i]/denom if denom else 0.
        return float(100*(far[i]+a*(far[i+1]-far[i])))
    i=int(np.argmin(abs(d)));return float(50*(frr[i]+far[i]))
def run(model,scope):
    start=time.perf_counter();source=N/'roles'/model/scope
    done=load(source/'completion.json');assert done['status']=='completed'
    assert sha(source/'scores_long.csv')==done['scores']['sha256']
    replay=load(N/'replay'/model/'completion.json');assert replay['status']=='passed'
    frame=pd.read_csv(source/'scores_long.csv',float_precision='round_trip')
    assert not frame.duplicated(['condition','trial_id','enrollment_role']).any()
    assert set(frame.enrollment_role)=={'teacher','student'} and frame.system.eq(model).all() and frame.probe_role.eq('student').all()
    assert frame.enrollment_checkpoint_sha256.eq(done['checkpoint']['sha256']).all() and frame.probe_checkpoint_sha256.eq(done['checkpoint']['sha256']).all()
    assert np.isfinite(frame.score).all()
    conditions=sorted(frame.condition.unique());assert len(conditions)==(3 if scope=='primary' else 27)
    clusters=sorted(frame.enrollment_speaker_id.unique());cluster_index={s:i for i,s in enumerate(clusters)}
    plans=[];observed=[];membership=[];validation=[]
    for c in conditions:
        groups={role:frame.loc[(frame.condition==c)&(frame.enrollment_role==role)].sort_values('trial_id').reset_index(drop=True) for role in ['teacher','student']}
        a,b=groups['teacher'],groups['student']
        fields=['trial_id','label','enrollment','probe','enrollment_speaker_id','probe_speaker_id','realization_manifest_sha256']
        joined=a[fields].merge(b[fields],on='trial_id',how='outer',validate='one_to_one',indicator=True,suffixes=('_a','_b'))
        assert joined['_merge'].eq('both').all()
        for f in fields[1:]:assert joined[f+'_a'].equals(joined[f+'_b']),f
        membership.append({'condition':c,'rows':len(a),'target':int(a.label.sum()),'nontarget':int((a.label==0).sum()),'excluded_candidates':5000-len(a),'strict_outer_join':'passed'})
        for role in ['teacher','student']:
            g=groups[role];p=prepared(g.label.to_numpy(dtype=np.int64),g.score.to_numpy(),g.enrollment_speaker_id.map(cluster_index).to_numpy())
            plans.append(p);v=weighted_eer(p,np.ones(len(clusters),dtype=np.int64));ref=reference_eer(g.label,g.score)
            assert abs(v-ref)<1e-10,(c,role,v,ref)
            observed.append(v);validation.append({'condition':c,'role':role,'weighted_unit_eer':v,'historical_reference_eer':ref,'absolute_delta_pp':abs(v-ref)})
    # Independent sklearn ROC cross-check of weighted resampling, including zero-weight observations.
    rng_check=np.random.default_rng(SEED+1);p=plans[0]
    first=frame.loc[(frame.condition==conditions[0])&(frame.enrollment_role=='teacher')].sort_values('trial_id')
    for _ in range(20):
        counts=np.bincount(rng_check.integers(0,len(clusters),len(clusters)),minlength=len(clusters))
        v=weighted_eer(p,counts);ref=reference_eer(first.label,first.score,counts[first.enrollment_speaker_id.map(cluster_index).to_numpy()])
        assert np.isfinite(v) and abs(v-ref)<1e-9
    rng=np.random.default_rng(SEED);replicates=[];draws=[];degenerate=0
    while len(replicates)<5000:
        count=np.bincount(rng.integers(0,len(clusters),len(clusters)),minlength=len(clusters))
        value=np.asarray([weighted_eer(p,count) for p in plans])
        if not np.isfinite(value).all():degenerate+=1;continue
        replicates.append(value);draws.append(count)
    boot=np.asarray(replicates).reshape(5000,len(conditions),2);obs=np.asarray(observed).reshape(len(conditions),2)
    # Columns teacher,student; negative TS−SS favors teacher enrollment.
    delta=boot[:,:,0]-boot[:,:,1];effect=obs[:,0]-obs[:,1]
    out=N/'statistics'/model/scope;out.mkdir(parents=True)
    with (out/'bootstrap.npz').open('xb') as f:np.savez_compressed(f,eer_percent=boot,cluster_counts=np.asarray(draws),speaker_ids=np.asarray(clusters),conditions=np.asarray(conditions))
    table=[]
    for i,c in enumerate(conditions):
        lo,hi=np.percentile(delta[:1000,i],[2.5,97.5])
        table.append({'model':model,'seed':2026,'condition':c,**{k:v for k,v in membership[i].items() if k!='condition'},'TS_eer_percent':obs[i,0],'SS_eer_percent':obs[i,1],
            'TS_minus_SS_pp':effect[i],'ci95_low_pp':float(lo),'ci95_high_pp':float(hi),'valid_cell_replicates':1000,'negative_favors':'EMA teacher enrollment','multiplicity':'unadjusted'})
    aggregate=delta.mean(axis=1);lo,hi=np.percentile(aggregate,[2.5,97.5])
    mean={'model':model,'scope':scope,'fixed_conditions':len(conditions),'equal_weight_TS_eer_percent':float(obs[:,0].mean()),'equal_weight_SS_eer_percent':float(obs[:,1].mean()),
        'equal_weight_TS_minus_SS_pp':float(effect.mean()),'ci95_low_pp':float(lo),'ci95_high_pp':float(hi),'valid_aggregate_replicates':5000,'degenerate_draws':degenerate,
        'speaker_clusters':len(clusters),'RNG_seed':SEED,'negative_favors':'EMA teacher enrollment','training_seed_replicates':1}
    csv(out/'cell_effects.csv',pd.DataFrame(table));csv(out/'aggregate_effect.csv',pd.DataFrame([mean]));save(out/'EER_scorer_validation.json',{'status':'passed','unit_weight_cells':validation,'independent_sklearn_weighted_draw_checks':20})
    text=f'''# Historical {model}: enrollment-role comparison ({scope})

**VERIFIED — Methods.** The same selected historical checkpoint (SHA-256 `{done['checkpoint']['sha256']}`) supplied EMA-teacher or student enrollment embeddings. Student probe embeddings were identical across enrollment roles, with the same hash-verified crop/noise realizations, 16kHz preprocessing and complete original sorted eligible BS8batches. Checkpoint selection was retained; no reselection occurred. This is an inference-role ablation conditional on one historical training run, not a matched training-objective comparison. The model passed all three full-primary raw replay cells at unchanged1e−5score tolerance and1e−6percentage-point EER tolerance. Strict outer joins verified all scored trial IDs, official labels, paths and recording-derived speaker IDs; no intersection exclusions were added.

**VERIFIED — Results.** Across {len(conditions)}fixed conditions, equal-weight mean TS and SS EER were {mean['equal_weight_TS_eer_percent']:.6f}% and {mean['equal_weight_SS_eer_percent']:.6f}%. TS−SS was {mean['equal_weight_TS_minus_SS_pp']:+.6f}percentage points (paired95%percentile interval [{lo:+.6f},{hi:+.6f}]). Negative differences favor teacher enrollment. Each primary cell contains1,937target/1,818nontarget rows and1,245excluded candidates; per-cell tables report every claimed condition's exact counts and EERs.

**VERIFIED — Uncertainty.** We used5,000valid aggregate bootstrap replicates, resampling enrollment-filename speaker clusters with replacement using one common draw across all fixed conditions and both roles. Cellwise intervals use the first1,000valid common draws and are unadjusted for multiplicity. Seed={SEED}; degenerate draws={degenerate}; observed enrollment-speaker clusters={len(clusters)}. The signed aggregate equally weights conditions. These intervals condition on one observed historical training seed; they do not quantify training-seed variability, and enrollment-only clustering can miss recurring-probe dependence. An interval containing zero does not establish equivalence. Literal historical H noise realizations were retained, including the documented overlap with reconstructed training/development noise pools.

**Scope.** {'Only3s enrollment/1.25s probe at clean,10dB,0dB is complete in this table; the27-cell aggregate is not claimed.' if scope=='primary' else 'All27frozen conditions are complete. Primary replay validated three cells; remaining historical TSscores/cache consistency were checked without claiming fresh raw replay of all27cells.'} B remains excluded from fresh B-dependent comparisons while its replay mismatch is unresolved. These within-{model} comparisons do not require B scores.
'''
    with (out/'MANUSCRIPT.md').open('x') as f:f.write(text)
    save(out/'completion.json',{'status':'completed','model':model,'scope':scope,'command':[sys.executable,*sys.argv],'elapsed_seconds':time.perf_counter()-start,
        'scores':{'path':str(source/'scores_long.csv'),'sha256':sha(source/'scores_long.csv')},'aggregate':mean,'checkpoint':done['checkpoint'],
        'software':{'numpy':np.__version__,'pandas':pd.__version__},'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'GPU_work':'None; CPU statistics only'})
    print(json.dumps(mean),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('model',choices=['C','D']);p.add_argument('scope',choices=['primary','grid27']);a=p.parse_args();run(a.model,a.scope)
