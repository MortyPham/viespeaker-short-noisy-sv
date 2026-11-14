"""Frozen 27-cell evaluation of selected controlled checkpoints, student/student."""
import argparse
import fcntl
import importlib.util
import os
import sys
import time
os.environ['HF_HUB_OFFLINE']='1'
from common import *

def run(config_path, enrollment_role='student'):
    cfg=load(config_path)
    for r in load(BASE/'preparation/release.json'):verify(r)
    selected=load(Path(cfg['output_dir'])/'selection.json')['checkpoint'];verify(selected)
    assert load(Path(cfg['output_dir'])/'completion.json')['global_step']==10000
    ss_out=BASE/f'evaluation/seed{cfg["training"]["seed"]}/{cfg["paper_system"]}'
    out=ss_out if enrollment_role=='student' else ss_out.parent/(ss_out.name+'_TS')
    assert enrollment_role=='student' or cfg['paper_system'] in ['C','D']
    out.mkdir(parents=True,exist_ok=True)
    if (out/'completion.json').exists():
        for r in load(out/'completion.json')['outputs']:verify(r)
        print('Evaluation already complete',out);return
    q=BASE.parent/'queue'
    spec=importlib.util.spec_from_file_location('frozen_inference',q/'gpu_stages.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    import numpy as np
    import pandas as pd
    for r in cfg['inputs']:verify(r)
    assert torch.cuda.is_available(),'CUDA required for full evaluation'
    ck=torch.load(selected['path'],map_location='cpu',weights_only=False)
    assert ck['config_sha256']==sha(config_path) and ck['global_step']>=1000
    assert ck['system']=={'B':'AAM_ONLY','C':'C_VECTOR_EMA','D':'D_FULL'}[cfg['paper_system']]
    model=hist.load_pretrained_encoder(args_from(cfg));model.load_state_dict(ck['student'],strict=True)
    enrollment_model=model
    if enrollment_role=='teacher':
        enrollment_model=hist.load_pretrained_encoder(args_from(cfg));enrollment_model.load_state_dict(ck['teacher'],strict=True)
    del ck
    enrollment_model=enrollment_model.to('cuda').eval()
    model=model.to('cuda').eval()
    smoke_keys=m.eligible('B','enrollment',3.)[:8]
    errors=m.hist.encode_cached(enrollment_model,smoke_keys,lambda key:m.hist.make_enrollment(m.prepared('B'),key,3.,m.SEED),
        out/'smoke/enrollment_3s','Selected-checkpoint evaluation CUDA smoke',8,torch.device('cuda'))
    assert not errors and len(m._load_embeddings(out/'smoke/enrollment_3s'))==len(smoke_keys)
    if not (out/'smoke_pass.json').exists():save(out/'smoke_pass.json',{'status':'passed','selected_checkpoint':selected,'rows':len(smoke_keys),'device':'cuda'})
    expected=m.archived('B').sort_values(['condition','trial_id']).reset_index(drop=True)
    # Use every historically eligible key in original BS8 order, retaining all crops/noise realizations.
    input_identity=sha(q/'frozen_inputs.json');cache=out/'embeddings'/selected['sha256']/cfg['test_protocol']['sha256']/input_identity/enrollment_role
    probe_cache=ss_out/'embeddings'/selected['sha256']/cfg['test_protocol']['sha256']/input_identity/'student'
    if enrollment_role=='teacher':
        ss_completion=load(ss_out/'completion.json')
        assert ss_completion['selected_checkpoint']==selected
        for info in ss_completion['outputs']:verify(info)
    maps={}
    began=time.perf_counter()
    for duration in [2.,3.,4.]:
        condition=f'enrollment_{duration:g}s';folder=cache/condition/'bs8'
        keys=m.eligible('B','enrollment',duration)
        errors=m.hist.encode_cached(enrollment_model,keys,lambda key,d=duration:m.hist.make_enrollment(m.prepared('B'),key,d,m.SEED),folder,
            f'{cfg["paper_system"]} SS seed{cfg["training"]["seed"]} {condition}',8,torch.device('cuda'))
        assert not errors,errors
        maps[condition]=m._load_embeddings(folder)
        assert set(maps[condition])==set(keys)
    for duration in [1.,1.25,1.5]:
        for snr in [None,10.,0.]:
            condition=m.hist.condition_id(duration,snr);folder=probe_cache/condition/'bs8'
            keys=m.eligible('B','probe',duration)
            if enrollment_role=='student':
                errors=m.hist.encode_cached(model,keys,lambda key,d=duration,s=snr:m.probe_wave('B',key,d,s),folder,
                    f'{cfg["paper_system"]} SS seed{cfg["training"]["seed"]} {condition}',8,torch.device('cuda'))
                assert not errors,errors
            else:
                assert folder.is_dir(), 'TS requires completed same-checkpoint SS probe cache'
            maps[condition]=m._load_embeddings(folder)
            assert set(maps[condition])==set(keys)
    groups=[];metrics=[]
    for condition,g in expected.groupby('condition',sort=True):
        ed=float(g.enrollment_duration_s.iloc[0]);probe=condition.split('_probe')[1]
        f=g.copy();f['score']=m.score(g,maps[f'enrollment_{ed:g}s'],maps[probe])
        f['system']=cfg['paper_system'];f['seed']=cfg['training']['seed'];f['cohort']=cfg['cohort']
        f['enrollment_role']=enrollment_role;f['probe_role']='student'
        f['realization_manifest_sha256']=input_identity;f['batch_policy']='complete original sorted eligible BS8'
        f['checkpoint_sha256']=selected['sha256']
        f['probe_cache_root']=str(probe_cache)
        f['probe_role']='student';f['checkpoint_sha256']=selected['sha256']
        f['enrollment_speaker_id']=f.enrollment.str.split('/').str[0];f['probe_speaker_id']=f.probe.str.split('/').str[0]
        f['enrollment_checkpoint_sha256']=selected['sha256'];f['probe_checkpoint_sha256']=selected['sha256']
        groups.append(f)
        metrics.append({'condition':condition,'system':cfg['paper_system'],'seed':cfg['training']['seed'],'rows':len(f),
            'target':int(f.label.sum()),'nontarget':int((f.label==0).sum()),'exclusions':5000-len(f),**m.hist.verification_metrics(f.label.to_numpy(),f.score.to_numpy())})
    scores=pd.concat(groups).sort_values(['condition','trial_id']).reset_index(drop=True)
    assert len(scores)==101610 and scores[m.KEY].equals(expected[m.KEY])
    for name,frame in [('scores_long.csv',scores),('cell_metrics.csv',pd.DataFrame(metrics))]:
        with (out/name).open('x') as f:frame.to_csv(f,index=False,float_format='%.17g')
    save(out/'completion.json',{'status':'completed','command':[sys.executable,*sys.argv],'device':'cuda',
        'selected_checkpoint':selected,'config':record(config_path),'seed':cfg['training']['seed'],'system':cfg['paper_system'],
        'roles':enrollment_role+'/student','trial_manifest':cfg['test_protocol'],'frozen_audio_inventory':record(q/'frozen_inputs.json'),
        'scored_rows':101610,'conditions':27,'elapsed_seconds':time.perf_counter()-began,
        'outputs':[record(out/'scores_long.csv'),record(out/'cell_metrics.csv')]+[record(p) for p in sorted(cache.rglob('*.npz'))],
        'limitations':'Same historical H noise recipes, including known training/development noise-pool overlap; two-seed training dispersion is descriptive'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);p.add_argument('--enrollment-role',choices=['student','teacher'],default='student');a=p.parse_args()
    with (ROOT/'experiments/evaluation/watchdogs/.locks/arch_b_gpu_watchdog_20260921.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_priority_and_gpu();run(a.config,a.enrollment_role)
