"""Bounded concurrent historical parity and primary enrollment-role evaluation.

No batch-size, preprocessing, numerical tolerance or shared job changes.
"""
import signal
import argparse, datetime, fcntl, gc, hashlib, importlib.util, json, os, subprocess, sys, time, traceback
from pathlib import Path
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['HF_HUB_OFFLINE']='1'
N=Path(__file__).resolve().parent
R=N.parents[2]
START=time.perf_counter()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def load(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2,default=str)
def event(kind,**x):
    row={'utc':now(),'elapsed_seconds':time.perf_counter()-START,'pid':os.getpid(),'event':kind,**x}
    with (N/'events.jsonl').open('a') as f:f.write(json.dumps(row,default=str)+'\n')
    temp=N/f'.status_{time.time_ns()}.tmp'
    with temp.open('x') as f:json.dump(row,f,indent=2,default=str)
    os.replace(temp,N/'STATUS.json')
    print(json.dumps(row,default=str),flush=True)
def gpu():
    result=subprocess.run(['nvidia-smi','--query-gpu=memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,capture_output=True,check=True)
    total,free,util=[int(v.strip()) for v in result.stdout.splitlines()[0].split(',')]
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True,capture_output=True,check=True).stdout.strip()
    return {'total_mib':total,'free_mib':free,'utilization_percent':util,'compute_processes':apps}

def main(stage):
    if stage=='smoke':
        def deadline(signum,frame):raise TimeoutError('Bounded CUDA smoke exceeded600seconds')
        signal.signal(signal.SIGALRM,deadline);signal.alarm(600)
    permission=load(N/'CONCURRENCY_PERMISSION.json')
    assert permission['user_authorized_bounded_attempt'] is True,'User authorization not established'
    policy=load(N/'RESOURCE_POLICY.json')
    state=gpu();budget=state['free_mib']-policy['headroom_mib']
    assert budget>=policy['minimum_attempt_budget_mib'],'Insufficient bounded capacity even for smoke'
    spec=importlib.util.spec_from_file_location('frozen',N/'queue/gpu_stages.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    global torch,np,pd
    torch,np,pd=m.torch,m.np,m.pd
    from torch.nn.utils.rnn import pad_sequence
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    assert torch.cuda.is_available(),'CUDA required; no CPU fallback'
    torch.cuda.set_per_process_memory_fraction(budget/state['total_mib'],0)
    torch.cuda.reset_peak_memory_stats()
    torch.backends.cudnn.allow_tf32=False
    for item in load(N/'RELEASE.json'):assert sha(item['path'])==item['sha256'],item['path']
    m.assert_inputs()
    metadata=m.metadata('concurrent_'+stage)
    metadata.update(resource_policy=load(N/'RESOURCE_POLICY.json'),launch_gpu=state,allocator_budget_mib=budget,
        concurrency_permission=permission,initialization='Allowlisted historical selected checkpoints; no training',output_directory=str(N),
        source_release_sha256=sha(N/'RELEASE.json'),research_question='Historical CUDA raw parity followed by primary C/D student-vs-EMA enrollment with fixed student probes')
    save(N/f'{stage}_run_metadata.json',metadata)
    event('STARTED',stage=stage,allocator_budget_mib=budget,gpu=state)
    active={'model':None,'role':None,'condition':None,'batch_start':None}
    batches=[];component=[];raw=[];checks=[]
    def peaks():return {'allocated_mib':torch.cuda.max_memory_allocated()/2**20,'reserved_mib':torch.cuda.max_memory_reserved()/2**20}
    def release(net):
        net.cpu();gc.collect();torch.cuda.empty_cache()
    def encoder(model,role):
        active.update(model=model,role=role,condition=None,batch_start=None)
        before=gpu();assert before['free_mib']>=policy['headroom_mib']+128,'Headroom lost before model load'
        net=m.encoder(model,role).to('cuda').eval()
        event('MODEL_LOADED',model=model,role=role,gpu=before,**peaks())
        return net
    def extract(model,role,side,duration,snr,net,selected=None,compare=False):
        condition=f'enrollment_{duration:g}s' if side=='enrollment' else m.hist.condition_id(duration,snr)
        keys=m.eligible(model,side,duration)
        starts=list(range(0,len(keys),8)) if selected is None else sorted({(keys.index(k)//8)*8 for k in selected})
        identity=m.MODELS[model].get('checkpoint',{}).get('sha256',m.MODELS['A']['pretrained_bundle']['historical_directory_sha256'])
        folder=N/('smoke' if stage=='smoke' else 'replay' if stage=='replay' else 'roles_primary')/'embeddings'/identity/role/condition/'bs8'
        folder.mkdir(parents=True,exist_ok=True)
        old=m.hist.load_embedding_map(Path(m.MODELS[model]['embedding_cache'][condition]['current_path'])) if compare else None
        vectors={}
        for batch_start in starts:
            active.update(model=model,role=role,condition=condition,batch_start=batch_start)
            # Every waveform/batch is unchanged, including unscored companions and relative lengths.
            batch_keys=keys[batch_start:batch_start+8];t=time.perf_counter()
            shard=folder/f'{batch_start:08d}.npz'
            assert not shard.exists(),'Use a new attempt directory; no implicit overwrite/resume'
            before=gpu()
            assert before['free_mib']>=policy['headroom_mib'],'Shared capacity lost; stop before next batch'
            waves=[torch.from_numpy(m.hist.make_enrollment(m.prepared(model),k,duration,m.SEED) if side=='enrollment' else m.probe_wave(model,k,duration,snr)) for k in batch_keys]
            padded=pad_sequence(waves,batch_first=True)
            lengths=torch.tensor([len(w) for w in waves],dtype=torch.float32)/padded.shape[1]
            with torch.inference_mode():
                value=torch.nn.functional.normalize(net(padded.to('cuda'),lengths.to('cuda')),dim=-1).cpu().numpy()
            torch.cuda.synchronize()
            assert value.shape==(len(batch_keys),192) and np.isfinite(value).all()
            assert np.allclose(np.linalg.norm(value,axis=1),1,atol=1e-5)
            with shard.open('xb') as f:np.savez_compressed(f,keys=np.asarray(batch_keys),embeddings=value)
            if compare:
                d=np.stack([value[i]-old[k] for i,k in enumerate(batch_keys)])
                component.append({'model':model,'role':role,'condition':condition,'batch_start':batch_start,
                    'max_component_delta':float(abs(d).max()),'mean_component_delta':float(abs(d).mean()),'rms_component_delta':float(np.sqrt((d*d).mean())),'tolerance':1e-5})
            vectors.update(zip(batch_keys,value.copy(),strict=True))
            record={'model':model,'role':role,'condition':condition,'batch_start':batch_start,'rows':len(batch_keys),
                'longest_waveform_seconds':padded.shape[1]/16000,'elapsed_seconds':time.perf_counter()-t,'gpu_before':before,
                'shard':str(shard),'shard_sha256':sha(shard),**peaks()}
            batches.append(record)
            with (N/f'{stage}_batches.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if len(batches)%10==0 or stage=='smoke':event('BATCH_COMPLETED',**record)
            del waves,padded,lengths,value
        return vectors
    try:
        if stage in ['smoke','replay']:
            ref=m.archived('B');ref=ref.loc[ref.condition=='enroll3s_probe1.25s_clean'].sort_values('trial_id')
            selected=[]
            for label in [0,1]:
                g=ref.loc[ref.label==label]
                selected.extend(int(g.iloc[int(q*(len(g)-1))].trial_id) for q in [.1,.7])
            if stage=='replay':assert load(N/'smoke_result.json')['status']=='passed'
            save(N/f'{stage}_sample_plan.json',{'stage':stage,'trial_ids':selected if stage=='smoke' else 'all primary','batch_size':8,'batch_policy':'complete original sorted eligible batches','seed':20260818,'score_tolerance':1e-5,'embedding_tolerance':1e-5,'full_EER_tolerance_pp':1e-6})
            for model,cfg in m.MODELS.items():
                if model!='B':continue
                archive=m.archived(model)
                chosen=ref.loc[ref.trial_id.isin(selected)] if stage=='smoke' else ref
                net=encoder(model,cfg['enrollment_role'])
                enroll=extract(model,cfg['enrollment_role'],'enrollment',3.,None,net,set(chosen.enrollment) if stage=='smoke' else None,stage=='smoke')
                release(net);del net
                net=encoder(model,cfg['probe_role'])
                for snr in [None,10.,0.]:
                    probes=extract(model,cfg['probe_role'],'probe',1.25,snr,net,set(chosen.probe) if stage=='smoke' else None,stage=='smoke')
                    condition=m.hist.evaluation_condition_id(3.,1.25,snr)
                    group=archive.loc[archive.condition==condition].copy()
                    if stage=='smoke':group=group.loc[group.trial_id.isin(selected)].copy()
                    expected=ref.loc[ref.trial_id.isin(selected)] if stage=='smoke' else ref
                    assert group[['trial_id','label','enrollment','probe']].sort_values('trial_id').reset_index(drop=True).equals(expected[['trial_id','label','enrollment','probe']].sort_values('trial_id').reset_index(drop=True))
                    values=m.score(group,enroll,probes);delta=values-group.score.to_numpy()
                    before=m.hist.verification_metrics(group.label.to_numpy(),group.score.to_numpy())
                    after=m.hist.verification_metrics(group.label.to_numpy(),values)
                    report={'model':model,'condition':condition,'rows':len(group),'target':int(group.label.sum()),'nontarget':int((group.label==0).sum()),
                        'max_absolute_score_delta':float(abs(delta).max()),'mean_absolute_score_delta':float(abs(delta).mean()),'rms_score_delta':float(np.sqrt((delta*delta).mean())),
                        'eer_delta_pp':after['eer_percent']-before['eer_percent'],'fresh_eer_percent':after['eer_percent'],
                        'passed':bool((abs(delta)<=1e-5).all() and (stage=='smoke' or abs(after['eer_percent']-before['eer_percent'])<=1e-6))}
                    checks.append(report);group['archived_score']=group.score;group['score']=values;group['score_delta']=delta;group['system']=model;raw.append(group)
                    with (N/stage/f'{model}_{condition}_scores.csv').open('x') as f:group.to_csv(f,index=False,float_format='%.17g')
                    event('CELL_CHECKED',**report)
                release(net);del net
            frame=pd.concat(raw,ignore_index=True)
            with (N/f'{stage}_scores.csv').open('x') as f:frame.to_csv(f,index=False,float_format='%.17g')
            passed=all(r['passed'] for r in checks) and all(r['max_component_delta']<=1e-5 for r in component)
            save(N/f'{stage}_component_checks.json',component)
            result={'status':'passed' if passed else 'parity_failed','stage':stage,'elapsed_seconds':time.perf_counter()-START,'scores':len(frame),'batches':len(batches),'checks':checks,
                'max_score_delta':max(x['max_absolute_score_delta'] for x in checks),'max_component_delta':max([x['max_component_delta'] for x in component],default=None),
                'input_files_consumed_and_verified':len(m.VERIFIED),'initial_gpu':state,'final_gpu':gpu(),'allocator_budget_mib':budget,**peaks()}
            save(N/f'{stage}_result.json',result)
            event('COMPLETED',**result)
            if not passed:raise RuntimeError('Parity discrepancy preserved; dependent historical inference blocked')
        elif stage=='roles':
            assert load(N/'replay_result.json')['status']=='passed'
            reference=m.archived('B');reference=reference.loc[reference.condition.str.startswith('enroll3s_probe1.25s_')].sort_values(['condition','trial_id']).reset_index(drop=True)
            parts=[];cells=[];probe_sources=[]
            for model in ['B','C','D']:
                archive=m.archived(model);archive=archive.loc[archive.condition.str.startswith('enroll3s_probe1.25s_')].copy()
                assert archive[m.KEY].sort_values(['condition','trial_id']).reset_index(drop=True).equals(reference[m.KEY])
                parts.append(m.longform(archive,model,'student' if model=='B' else 'teacher'))
                if model=='B':continue
                net=encoder(model,'student')
                enroll=extract(model,'student','enrollment',3.,None,net)
                release(net);del net
                for condition,g in archive.groupby('condition',sort=True):
                    probe_condition=condition.split('_probe')[1]
                    folder=Path(m.MODELS[model]['embedding_cache'][probe_condition]['current_path'])
                    probes=m.hist.load_embedding_map(folder)
                    probe_sources.append({'model':model,'condition':condition,'folder':str(folder),'files':[{'path':str(p),'sha256':sha(p)} for p in sorted(folder.glob('*.npz'))]})
                    ss=g.copy();ss['score']=m.score(g,enroll,probes)
                    parts.append(m.longform(ss,model,'student'))
            frame=pd.concat(parts,ignore_index=True)
            assert len(frame)==5*11265
            for (model,role),g in frame.groupby(['system','enrollment_role']):
                assert g[m.KEY].sort_values(['condition','trial_id']).reset_index(drop=True).equals(reference[m.KEY])
                assert np.isfinite(g.score).all()
                for condition,c in g.groupby('condition'):
                    cells.append({'system':model,'enrollment_role':role,'probe_role':'student','seed':2026,'condition':condition,'rows':len(c),'target':int(c.label.sum()),'nontarget':int((c.label==0).sum()),'excluded':5000-len(c),**m.hist.verification_metrics(c.label.to_numpy(),c.score.to_numpy())})
            with (N/'roles_primary/scores_long.csv').open('x') as f:frame.to_csv(f,index=False,float_format='%.17g')
            with (N/'roles_primary/cell_metrics.csv').open('x') as f:pd.DataFrame(cells).to_csv(f,index=False,float_format='%.17g')
            save(N/'roles_primary/probe_provenance.json',probe_sources)
            result={'status':'completed','stage':stage,'elapsed_seconds':time.perf_counter()-START,'rows':len(frame),'conditions':3,'target_per_condition':1937,'nontarget_per_condition':1818,'excluded_per_condition':1245,
                'same_student_probes':'Hash-verified archived same-run selected-checkpoint probes, identical across enrollment roles','checkpoints':{x:m.MODELS[x]['checkpoint'] for x in ['B','C','D']},'initial_gpu':state,'final_gpu':gpu(),**peaks()}
            save(N/'roles_result.json',result);event('COMPLETED',**result)
    except BaseException as error:
        details={'status':'OOM' if isinstance(error,torch.cuda.OutOfMemoryError) else 'failed','error':repr(error),'traceback':traceback.format_exc(),
            'stage':stage,'active':active,'elapsed_seconds':time.perf_counter()-START,'initial_gpu':state,'gpu_at_failure':gpu(),'allocator_budget_mib':budget,**peaks()}
        save(N/f'failure_{stage}_{time.time_ns()}.json',details)
        event('FAILED',**details)
        # Own resources are released when this worker exits; no foreign process is signaled.
        raise
    finally:
        gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','replay','roles']);args=p.parse_args()
    with (R/'experiments/evaluation/watchdogs/.locks/arch_b_gpu_watchdog_20260921.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main(args.stage)
