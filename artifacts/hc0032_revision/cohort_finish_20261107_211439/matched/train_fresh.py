"""Sequential fresh cohort job; full32matched updates and complete resume state."""
import argparse,fcntl,shutil,traceback
from engine_v5 import *
def run(config_path,qualification):
    cfg=load(config_path);verify_inputs(cfg)
    for item in load(RUN/'TRAIN_RELEASE.json'):verify(item)
    assert load(qualification/'EQUIVALENCE_PASS.json')['status']=='passed'
    assert load(qualification/'PROFILE_PASS.json')['status']=='passed'
    assert load(qualification/'EQUIVALENCE_PASS.json')['seed']==cfg['training']['seed']
    assert shutil.disk_usage(RUN).free>5*1024**3,'Insufficient checkpoint storage allowance'
    a,records,train,ds,aug,assignments,consumed=make_data(cfg)
    import csv
    with (BASE/'preparation/validation_assignments.csv').open() as f:
        frozen={(r['speaker'],r['role'],r['source_id'],r['path']) for r in csv.DictReader(f)}
    actual={(speaker,role,r.source_id,r.path) for speaker,roles in assignments.items() for role,items in roles.items() for r in items}
    assert actual==frozen and not set(train)&set(assignments),'Development assignment/population mismatch'
    assert a.steps==10000 and a.speakers_per_batch==16 and a.pairs_per_speaker==2
    expected_roles={'B':(0.,0.),'C':(1.,0.),'D':(1.,.01)}
    assert (a.lambda_vector,a.lambda_kl)==expected_roles[cfg['paper_system']]
    state=build(cfg,a,train);model=cfg['paper_system'];out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    if (out/'completion.json').exists():
        assert load(out/'completion.json')['config_sha256']==sha(config_path);return
    reference=BASE/f'runs/seed{a.seed}/B/batch_fingerprints.jsonl'
    baseline={}
    if model!='B':
        assert load(reference.parent/'completion.json')['global_step']==10000
        for line in reference.read_text().splitlines():
            row=__import__('json').loads(line)
            if row['step'] in baseline:assert baseline[row['step']]==row['sha256']
            baseline[row['step']]=row['sha256']
    student,head,teacher,opt,sched,scaler=state
    start=0;best=float('inf');bestpath=None;history=[];resume=None
    gen=torch.Generator().manual_seed(hist.stable_int(f'HC0032:loader:{a.seed}'))
    if (out/'resume_pointer.json').exists():
        pointer_record=load(out/'resume_pointer.json');verify(pointer_record)
        resume=torch.load(pointer_record['path'],map_location='cpu',weights_only=False)
        assert resume['config_sha256']==sha(config_path)
        student.load_state_dict(resume['student'],strict=True);head.load_state_dict(resume['aam_head'],strict=True)
        if teacher is not None:teacher.load_state_dict(resume['teacher'],strict=True)
        opt.load_state_dict(resume['optimizer']);sched.load_state_dict(resume['scheduler']);scaler.load_state_dict(resume['scaler'])
        gen.set_state(resume['loader_generator_state']);start=resume['global_step'];best=resume['best_noise_eer'];bestpath=resume['best_checkpoint'];history=resume['history']
    dl,gen=loader(ds,train,a,start,a.steps-start,gen)
    if resume is not None:restore_rng(resume['rng_state']);del resume
    attempt=time.time_ns();t=time.perf_counter();costs=[]
    event(out,'STARTED',command=[sys.executable,*sys.argv],seed=a.seed,model=model,config=record(config_path),qualification=str(qualification),initialization=cfg['initialization'],attempt=attempt,start_step=start,
        recipe='V1full32FP16documentedMUSANrecipe; unchangedobjectives; validatedlosslessmemoryexecution',gpu=gpu())
    for index,batch in enumerate(dl,start=start+1):
        fingerprint=batch_hash(batch)
        if model!='B':assert baseline[index]==fingerprint,f'Matched input mismatch at{index}'
        with (out/'batch_fingerprints.jsonl').open('a') as f:f.write(json.dumps({'utc':now(),'attempt':attempt,'step':index,'sha256':fingerprint})+'\n')
        if index%20==1 and gpu()['free_mib']<1024:
            raise RuntimeError('Physical headroom lost; stop and resume from last completed checkpoint after capacity changes')
        metrics,_=step(state,batch,cfg,a,index,True)
        costs.append(metrics['elapsed_seconds'])
        with (out/'training_metrics.jsonl').open('a') as f:f.write(json.dumps({'utc':now(),'attempt':attempt,**metrics})+'\n')
        if index%20==0:event(out,'PROGRESS',seed=a.seed,model=model,step=index,loss=metrics['loss'],elapsed_seconds=time.perf_counter()-t,metrics=metrics)
        selected_now=False
        if index%1000==0:
            torch.manual_seed(hist.stable_int(f'HC0032:validation:{a.seed}'))
            with memory_forward(student,None):v=hist.evaluate_verification(student,student,assignments,aug,a,torch.device('cuda'))
            v['step']=index;history.append(v);event(out,'VALIDATION',metrics=v)
            if float(v['noise_eer'])<best:best=float(v['noise_eer']);selected_now=True
            student.train();head.train()
        if index==1 or index%1000==0:
            # Immutable checkpoints published only at successful update boundaries.
            candidate=out/f'checkpoint_step{index:05d}_{time.time_ns()}.pt'
            if selected_now:bestpath=str(candidate)
            payload={'system':{'B':'AAM_ONLY','C':'C_VECTOR_EMA','D':'D_FULL'}[model],'student':student.state_dict(),'aam_head':head.state_dict(),'teacher':None if teacher is None else teacher.state_dict(),
                'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'scaler':scaler.state_dict(),'global_step':index,'signature':{**cfg['training'],'train_speaker_ids':train},'cohort':cfg['cohort'],
                'config_sha256':sha(config_path),'initialization':cfg['initialization'],'history':history,'best_noise_eer':best,'best_checkpoint':bestpath,'rng_state':rng_state(),
                'loader_generator_state':gen.get_state(),'sampler_state':{'seed':a.seed,'next_start_step':index,'speakers_per_batch':16,'pairs_per_speaker':2},'execution_source':record(BASE/'engine_v5.py')}
            with candidate.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
            del payload
            pointer(out/'resume_pointer.json',record(candidate));event(out,'CHECKPOINT',checkpoint=record(candidate),step=index,selected=selected_now)
    assert bestpath is not None and len(history)==10 and history[-1]['step']==10000
    save(out/'selection.json',{'checkpoint':record(bestpath),'rule':cfg['selection_rule'],'noise_eer':best,'history':history})
    save(out/'completion.json',{'status':'completed','global_step':10000,'seed':a.seed,'model':model,'config_sha256':sha(config_path),'selected_checkpoint':record(bestpath),
        'initialization':cfg['initialization'],'command':[sys.executable,*sys.argv],'elapsed_seconds':time.perf_counter()-t,'mean_measured_step_seconds':sum(costs)/len(costs) if costs else None,'data_input_files_consumed_parent':len(consumed),
        'RNG_and_resume':'Python/NumPy/TorchCPU/CUDA, loader generator, sampler nextstep, optimizer/scheduler/scaler/EMA/student/head saved. Item-local data RNG and per-step forward fork RNG make worker seeds irrelevant to scientific inputs.',
        'scientific_scope':'Fresh matched cohort; historical2026not reused; no test-based checkpoint selection'})
    event(out,'COMPLETED',seed=a.seed,model=model,global_step=10000)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--qualification',type=Path,required=True);args=p.parse_args()
    with (ROOT/'experiments/evaluation/watchdogs/.locks/arch_b_gpu_watchdog_20260921.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:gate=gpu_guard();run(args.config,args.qualification)
        except BaseException as error:
            save(BASE/'logs'/f'train_failure_{time.time_ns()}.json',{'status':'OOM' if isinstance(error,torch.cuda.OutOfMemoryError) else 'failed','error':repr(error),'traceback':traceback.format_exc(),'command':[sys.executable,*sys.argv],'gpu':gpu()})
            raise
