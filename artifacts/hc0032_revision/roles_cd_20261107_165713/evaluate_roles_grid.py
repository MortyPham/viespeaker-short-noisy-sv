"""Independent C/D replay gates and within-model enrollment-role ablation."""
import argparse,datetime,fcntl,gc,hashlib,importlib.util,json,os,subprocess,sys,time,traceback
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
N=Path(__file__).resolve().parent
R=N.parents[2]
START=time.perf_counter()
def load(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def info(p):return {'path':str(p),'sha256':sha(p)}
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2,default=str)
def csv(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:x.to_csv(f,index=False,float_format='%.17g')
def event(kind,**x):
    item={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'worker_elapsed_seconds':time.perf_counter()-START,'pid':os.getpid(),'event':kind,**x}
    with (N/'events.jsonl').open('a') as f:f.write(json.dumps(item,default=str)+'\n')
    tmp=N/f'.status_{time.time_ns()}.tmp'
    with tmp.open('x') as f:json.dump(item,f,indent=2,default=str)
    os.replace(tmp,N/'STATUS.json');print(json.dumps(item,default=str),flush=True)
def gpu():
    result=subprocess.run(['nvidia-smi','--query-gpu=memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    total,free,util=[int(x.strip()) for x in result.stdout.splitlines()[0].split(',')]
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
    return dict(total_mib=total,free_mib=free,utilization_percent=util,compute_processes=apps)
spec=importlib.util.spec_from_file_location('frozen',N/'queue/gpu_stages.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch,np,pd=m.torch,m.np,m.pd
from torch.nn.utils.rnn import pad_sequence
torch.set_num_threads(2);torch.set_num_interop_threads(1)
ACTIVE={};BATCHES=0
def peak():return dict(peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
def validate_rows(frame,reference):
    assert not frame.duplicated(['condition','trial_id']).any()
    joined=frame[m.KEY].merge(reference[m.KEY],on=['condition','trial_id'],how='outer',validate='one_to_one',suffixes=('_x','_y'),indicator=True)
    assert joined['_merge'].eq('both').all(),'Missing or extra rows; no intersection permitted'
    for col in ['label','enrollment','probe']:assert joined[col+'_x'].equals(joined[col+'_y']),col
    assert np.isfinite(frame.score).all()
def original_map(model,condition):
    folder=Path(m.MODELS[model]['embedding_cache'][condition]['current_path'])
    side='enrollment' if condition.startswith('enrollment') else 'probe'
    duration=float(condition.removeprefix('enrollment_').removesuffix('s')) if side=='enrollment' else float(condition.split('s_')[0])
    keys=m.eligible(model,side,duration);vectors={};records=[]
    for p in sorted(folder.glob('*.npz')):
        m.verify_frozen(p)
        with np.load(p,allow_pickle=False) as data:
            batchkeys=data['keys'].astype(str).tolist();value=data['embeddings'].copy()
        start=int(p.stem);assert batchkeys==keys[start:start+8],'Archived batch membership mismatch'
        assert value.shape==(len(batchkeys),192) and np.isfinite(value).all() and np.allclose(np.linalg.norm(value,axis=1),1,atol=1e-5)
        assert not set(batchkeys)&set(vectors),'Duplicate archived embedding key'
        vectors.update(zip(batchkeys,value,strict=True));records.append(info(p))
    assert set(vectors)==set(keys)
    provenance=N/'audit'/f'{model}_{condition}_cache.json'
    if not provenance.exists():save(provenance,{'model':model,'condition':condition,'files':records,'keys':len(vectors),'source':'frozen full-checkpoint/role/protocol BS8 cache'})
    return vectors
def encode(model,role,side,duration,snr,net,purpose):
    global BATCHES
    condition=f'enrollment_{duration:g}s' if side=='enrollment' else m.hist.condition_id(duration,snr)
    keys=m.eligible(model,side,duration)
    folder=N/purpose/'embeddings'/m.MODELS[model]['checkpoint']['sha256']/m.MODELS[model]['trial_manifest']['sha256']/sha(N/'queue/frozen_inputs.json')/role/condition/'bs8'
    folder.mkdir(parents=True,exist_ok=True);vectors={}
    for start in range(0,len(keys),8):
        ACTIVE.update(model=model,role=role,condition=condition,batch_start=start,purpose=purpose)
        batchkeys=keys[start:start+8];p=folder/f'{start:08d}.npz'
        assert not p.exists(),'Fresh attempt required; no unverified resume/overwrite'
        state=gpu();assert state['free_mib']>=1536,'Physical headroom lost; stop before next batch'
        t=time.perf_counter()
        waves=[torch.from_numpy(m.hist.make_enrollment(m.prepared(model),k,duration,m.SEED) if side=='enrollment' else m.probe_wave(model,k,duration,snr)) for k in batchkeys]
        padded=pad_sequence(waves,batch_first=True);lengths=torch.tensor([len(w) for w in waves],dtype=torch.float32)/padded.shape[1]
        with torch.inference_mode():value=torch.nn.functional.normalize(net(padded.to('cuda'),lengths.to('cuda')),dim=-1).cpu().numpy()
        torch.cuda.synchronize()
        assert value.shape==(len(batchkeys),192) and np.isfinite(value).all() and np.allclose(np.linalg.norm(value,axis=1),1,atol=1e-5)
        with p.open('xb') as f:np.savez_compressed(f,keys=np.asarray(batchkeys),embeddings=value)
        vectors.update(zip(batchkeys,value.copy(),strict=True));BATCHES+=1
        record={**ACTIVE,'batch_rows':len(batchkeys),'elapsed_seconds':time.perf_counter()-t,'longest_waveform_seconds':padded.shape[1]/16000,'gpu_before':state,'shard':info(p),**peak()}
        with (N/'batches.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        if start%160==0:event('BATCH_COMPLETED',batches_total=BATCHES,**record)
        del waves,padded,lengths,value
    return vectors
def with_encoder(model,role,fn):
    state=gpu();assert state['free_mib']>=1536+512,'Insufficient headroom for one encoder'
    ACTIVE.update(model=model,role=role)
    net=m.encoder(model,role).to('cuda').eval()
    event('MODEL_LOADED',model=model,role=role,gpu=state,**peak())
    try:return fn(net)
    finally:
        net.cpu();del net;gc.collect();torch.cuda.empty_cache()
def primary_replay(model):
    started=time.perf_counter();out=N/'replay'/model;out.mkdir()
    archive=m.archived(model);reference=archive.loc[archive.condition.str.startswith('enroll3s_probe1.25s_')].copy()
    assert len(reference)==11265
    enroll=with_encoder(model,'teacher',lambda net:encode(model,'teacher','enrollment',3.,None,net,'replay'))
    raw=[];checks=[]
    def probes(net):
        for snr in [None,10.,0.]:
            values=encode(model,'student','probe',1.25,snr,net,'replay')
            condition=m.hist.evaluation_condition_id(3.,1.25,snr)
            g=reference.loc[reference.condition==condition].copy();assert len(g)==3755 and int(g.label.sum())==1937
            scores=m.score(g,enroll,values);delta=scores-g.score.to_numpy()
            before=m.hist.verification_metrics(g.label.to_numpy(),g.score.to_numpy());after=m.hist.verification_metrics(g.label.to_numpy(),scores)
            g['archived_score']=g.score;g['score']=scores;g['score_delta']=delta;validate_rows(g,reference.loc[reference.condition==condition])
            csv(out/f'{condition}_scores.csv',g);raw.append(g)
            report={'model':model,'condition':condition,'rows':len(g),'target':1937,'nontarget':1818,'excluded_candidates':1245,
                'max_score_delta':float(abs(delta).max()),'mean_absolute_score_delta':float(abs(delta).mean()),'rms_score_delta':float(np.sqrt(np.mean(delta*delta))),
                'archived_eer_percent':before['eer_percent'],'fresh_eer_percent':after['eer_percent'],'eer_delta_pp':after['eer_percent']-before['eer_percent'],
                'score_tolerance':1e-5,'eer_tolerance_pp':1e-6,'passed':bool((abs(delta)<=1e-5).all() and abs(after['eer_percent']-before['eer_percent'])<=1e-6)}
            checks.append(report);save(out/f'{condition}_check.json',report);event('REPLAY_CELL',**report)
    with_encoder(model,'student',probes)
    csv(out/'scores.csv',pd.concat(raw,ignore_index=True))
    passed=all(x['passed'] for x in checks)
    result={'status':'passed' if passed else 'parity_failed','model':model,'checks':checks,'elapsed_seconds':time.perf_counter()-started,
        'checkpoint':m.MODELS[model]['checkpoint'],'roles':'teacher/student','rows':11265,'initialization':'Same selected historical checkpoint; no reselection',**peak()}
    save(out/'completion.json',result);event('PRIMARY_REPLAY_FINISHED',**result)
    return passed
def roles(model,grid=False):
    assert load(N/'replay'/model/'completion.json')['status']=='passed'
    scope='grid27' if grid else 'primary';out=N/'roles'/model/scope;out.mkdir(parents=True)
    archive=m.archived(model)
    if not grid:archive=archive.loc[archive.condition.str.startswith('enroll3s_probe1.25s_')].copy()
    parts=[];cells=[];all_student={}
    if grid:
        journal=[json.loads(line) for line in (N/'batches.jsonl').read_text().splitlines()]
        original=[x for x in journal if x['model']==model and x['role']=='student' and x['condition']=='enrollment_3s' and x['purpose']=='roles']
        assert original, 'Completed primary student enrollment required'
        for x in original:assert sha(x['shard']['path'])==x['shard']['sha256']
        folder=Path(original[0]['shard']['path']).parent
        assert len(list(folder.glob('*.npz')))==len(original)
        all_student[3.]=m._load_embeddings(folder)
        assert set(all_student[3.])==set(m.eligible(model,'enrollment',3.))
        save(N/'audit'/f'{model}_fresh3s_grid_reuse.json',{'status':'verified','shards':[x['shard'] for x in original],'purpose':'Reuse same fresh primary SS enrollment vectors for all remaining probe conditions'})
    def extract_student(net):
        for duration in ([2.,4.] if grid else [3.]):all_student[duration]=encode(model,'student','enrollment',duration,None,net,'roles')
    with_encoder(model,'student',extract_student)
    cached_probes={};cached_teacher={}
    for condition,g in archive.groupby('condition',sort=True):
        pdur=condition.split('_probe')[1];duration=float(g.enrollment_duration_s.iloc[0])
        if pdur not in cached_probes:cached_probes[pdur]=original_map(model,pdur)
        teacher_condition=f'enrollment_{duration:g}s'
        if teacher_condition not in cached_teacher:cached_teacher[teacher_condition]=original_map(model,teacher_condition)
        probes=cached_probes[pdur]
        ts=g.copy();reconstructed_ts=m.score(g,cached_teacher[teacher_condition],probes)
        assert np.max(abs(reconstructed_ts-g.score.to_numpy()))<=1e-5,'Archived TS cache/scoring mismatch: shared pipeline guard'
        ts=m.longform(ts,model,'teacher')
        ss=g.copy()
        enrol=all_student[duration]
        ss['score']=m.score(g,enrol,probes);ss=m.longform(ss,model,'student')
        for f in [ts,ss]:
            validate_rows(f,g);f['scope']=scope;f['realization_manifest_sha256']=sha(N/'queue/frozen_inputs.json')
            f['probe_cache_policy']='Identical hash-verified archived same-checkpoint student probes across enrollment roles'
            f['batch_policy']='Complete original sorted side-eligible BS8 batches; no change'
            parts.append(f)
            cells.append({'model':model,'seed':2026,'condition':condition,'enrollment_role':f.enrollment_role.iloc[0],'probe_role':'student','rows':len(f),
                'target':int(f.label.sum()),'nontarget':int((f.label==0).sum()),'excluded_candidates':5000-len(f),**m.hist.verification_metrics(f.label.to_numpy(),f.score.to_numpy())})
    frame=pd.concat(parts,ignore_index=True);assert len(frame)==2*(101610 if grid else 11265)
    for role,g in frame.groupby('enrollment_role'):validate_rows(g,archive)
    csv(out/'scores_long.csv',frame);csv(out/'cell_metrics.csv',pd.DataFrame(cells))
    result={'status':'completed','model':model,'scope':scope,'rows':len(frame),'conditions':27 if grid else 3,'checkpoint':m.MODELS[model]['checkpoint'],
        'probes':'Identical archived student probes; no role-specific probe recomputation','selection':'Historical TS checkpoint retained, no reselection','scores':info(out/'scores_long.csv'),**peak()}
    save(out/'completion.json',result);event('ROLE_SCORES_COMPLETED',**result)
def run(grid):
    for item in load(N/'GRID_RELEASE.json'):assert sha(item['path'])==item['sha256'],item['path']
    audit=load(N/'audit/SMOKE_VERIFICATION.json');assert audit['status']=='passed_C_D'
    m.assert_inputs();assert torch.cuda.is_available()
    before=gpu();torch.cuda.init();after=gpu()
    budget=after['free_mib']-1536-512;assert budget>=512,'Insufficient capacity after context overhead and headroom'
    torch.cuda.set_per_process_memory_fraction(budget/(torch.cuda.get_device_properties(0).total_memory/2**20),0)
    torch.cuda.reset_peak_memory_stats()
    save(N/('grid_run.json' if grid else 'primary_run.json'),{**m.metadata('C_D_independent_roles'),'research_question':load(N/'PLAN.json')['question'],
        'command':[sys.executable,*sys.argv],'output_directory':str(N),'before_context':before,'after_context':after,'observed_context_overhead_mib':before['free_mib']-after['free_mib'],
        'extra_library_margin_mib':512,'physical_headroom_mib':1536,'pytorch_budget_mib':budget,'initialization':'Allowlisted historical selected states; no training','release':info(N/'GRID_RELEASE.json')})
    event('STARTED',scope='grid27' if grid else 'primary',gpu=after,pytorch_budget_mib=budget)
    for model in ['C','D']:
        if grid:
            if (N/'roles'/model/'primary/completion.json').exists():roles(model,True)
        elif primary_replay(model):roles(model)
        else:event('MODEL_BLOCKED',model=model,reason='Own full-primary replay failed; no enrollment-role inference')
    event('WORKER_FINISHED',scope='grid27' if grid else 'primary',gpu=gpu(),**peak())
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--grid',action='store_true');args=p.parse_args()
    with (R/'experiments/evaluation/watchdogs/.locks/arch_b_gpu_watchdog_20260921.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(args.grid)
        except BaseException as error:
            report={'status':'OOM' if isinstance(error,torch.cuda.OutOfMemoryError) else 'failed','error':repr(error),'traceback':traceback.format_exc(),'active':ACTIVE,'elapsed_seconds':time.perf_counter()-START,'gpu':gpu(),**peak()}
            save(N/f'failure_{time.time_ns()}.json',report);event('FAILED',**report)
            raise
