"""Sequential CUDA-only B/C/D controlled training; historical components reused.

Scientific checkpoints are immutable. Only resume_pointer.json is a mutable
operational pointer. Smoke starts from the same initialization in a separate
directory and never supplies weights to a full run.
"""
import argparse
import copy
import fcntl
import gc
import math
import os
import subprocess
import sys
import time
from datetime import datetime,timezone
os.environ['HF_HUB_OFFLINE']='1'
from common import *
torch.set_num_threads(2)

def now():return datetime.now(timezone.utc).isoformat()
def journal(out,name,**data):
    with (out/f'{name}.jsonl').open('a') as f:f.write(json.dumps({'utc':now(),**data},default=str)+'\n')
def pointer(path,obj):
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temporary.open('x') as f:json.dump(obj,f,indent=2)
    os.replace(temporary,path)
def baseline_fingerprints(path):
    result={}
    if path.exists():
        for line in path.read_text().splitlines():
            r=__import__('json').loads(line)
            if r['step'] in result:assert result[r['step']]==r['sha256']
            result[r['step']]=r['sha256']
    return result

def run(cfg,config_path,mode):
    import importlib.metadata
    environment=load(BASE/'preparation/software_environment.json')
    for name,version in environment['packages'].items():assert importlib.metadata.version(name)==version,(name,version)
    for item in load(BASE/'preparation/release.json'):verify(item)
    for item in cfg['inputs']:verify(item)
    for item in load(BASE/'preparation/source_snapshot.json'):verify(item)
    verify(cfg['initialization'])
    registry=load(ROOT/'artifacts/hc0032_revision/prompt1_20261106_2153/final/model_registry.json')
    for item in registry['models']['A']['pretrained_bundle']['files']:
        assert sha(Path(cfg['pretrained_bundle'])/Path(item['path']).name)==item['sha256']
    parity=load(BASE/'preparation/parity_checks.json');assert parity['status']=='passed'
    a=args_from(cfg);device=torch.device('cuda');assert torch.cuda.is_available(),'CUDA required; no CPU fallback'
    model=cfg['paper_system'];seed=a.seed
    out=BASE/f'smoke/seed{seed}/{model}' if mode=='smoke' else Path(cfg['output_dir'])
    out.mkdir(parents=True,exist_ok=True)
    final=out/('smoke_pass.json' if mode=='smoke' else 'completion.json')
    if final.exists():
        assert load(final)['config_sha256']==sha(config_path)
        print('Already completed',final);return
    if mode=='train':
        for member in ['B','C','D']:
            gate=load(BASE/f'smoke/seed{seed}/{member}/smoke_pass.json')
            assert gate['status']=='passed' and gate['actual_speakers_per_batch']==16 and gate['actual_batch_size']==32
            assert gate['config_sha256']==sha(BASE/f'configs/{member}_seed{seed}.json')
    records,train,val=data(a)
    assert train==load(BASE/'preparation/populations.json')['train_speaker_ids']
    augmenter=hist.MusanAugmenter(a.musan_dir,a)
    assignments=hist.create_validation_assignments(records,val,a)
    dataset=hist.SpeakerPairDataset(records,{s:i for i,s in enumerate(train)},augmenter,a)
    audio_hashes={}
    import csv
    with (BASE/'preparation/audio_inputs.csv').open() as f:
        for row in csv.DictReader(f):audio_hashes[row['path']]=row['sha256']
    for row in load(BASE/'preparation/noise_inputs.json'):audio_hashes[row['path']]=row['sha256']
    consumed=set();original_audio=hist.load_audio
    def checked_audio(path,sr):
        p=str(path);assert p in audio_hashes,p
        if p not in consumed:assert sha(p)==audio_hashes[p],p;consumed.add(p)
        return original_audio(path,sr)
    hist.load_audio=checked_audio
    hist.seed_everything(seed)
    initial=torch.load(cfg['initialization']['path'],map_location='cpu',weights_only=False)
    student=hist.load_pretrained_encoder(a).cpu().eval()
    student.load_state_dict(initial['student'],strict=True)
    assert tensor_hash(student.state_dict())==cfg['initialization']['student_tensor_sha256']
    head=hist.AAMSoftmaxHead(192,len(train),a.aam_margin,a.aam_scale)
    head.load_state_dict(initial['aam_head'],strict=True)
    assert tensor_hash(head.state_dict())==cfg['initialization']['head_tensor_sha256']
    # No dummy training-mode forward may modify initial BatchNorm buffers.
    student=student.to(device);head=head.to(device)
    teacher=None if model=='B' else copy.deepcopy(student).eval().requires_grad_(False)
    del initial
    optimizer=torch.optim.AdamW([{'params':student.parameters(),'lr':a.encoder_lr},{'params':head.parameters(),'lr':a.head_lr}],
        weight_decay=a.weight_decay,betas=(.9,.999),eps=1e-8)
    def multiplier(step):
        if step<a.warmup_steps:return max(step,1)/max(a.warmup_steps,1)
        return .5*(1+math.cos(math.pi*min(max((step-a.warmup_steps)/(a.steps-a.warmup_steps),0),1)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,multiplier)
    scaler=torch.amp.GradScaler('cuda',enabled=True,init_scale=1024.)
    start=0;best=float('inf');best_checkpoint=None;history=[]
    if (out/'resume_pointer.json').exists():
        latest=load(out/'resume_pointer.json');verify(latest)
        ck=torch.load(latest['path'],map_location=device,weights_only=False)
        assert ck['config_sha256']==sha(config_path)
        student.load_state_dict(ck['student'],strict=True);head.load_state_dict(ck['aam_head'],strict=True)
        if teacher is not None:teacher.load_state_dict(ck['teacher'],strict=True)
        optimizer.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);scaler.load_state_dict(ck['scaler'])
        start=ck['global_step'];best=ck['best_noise_eer'];best_checkpoint=ck['best_checkpoint'];history=ck['history'];del ck
    reference_dir=BASE/f'{"smoke" if mode=="smoke" else "runs"}/seed{seed}/B'
    baseline=baseline_fingerprints(reference_dir/'batch_fingerprints.jsonl') if model!='B' else {}
    limit=cfg['smoke_steps'] if mode=='smoke' else a.steps
    sampler=hist.SpeakerBalancedBatchSampler(train,limit-start,16,2,seed,start_step=start)
    options={'num_workers':a.num_workers,'pin_memory':True}
    if a.num_workers:options.update(persistent_workers=True,prefetch_factor=2)
    loader=torch.utils.data.DataLoader(dataset,batch_sampler=sampler,collate_fn=hist.collate_pairs,**options)
    torch.backends.cudnn.benchmark=False
    student.train();head.train()
    torch.cuda.reset_peak_memory_stats();elapsed_start=time.perf_counter();timings=[]
    journal(out,'events',event='STARTED',command=[sys.executable,*sys.argv],pid=os.getpid(),seed=seed,model=model,
        config_sha256=sha(config_path),initialization=cfg['initialization'],mode=mode,device='cuda',start_step=start)
    for offset,batch in enumerate(loader,start=start+1):
        fingerprint=batch_hash(batch)
        if model!='B':assert baseline.get(offset)==fingerprint,f'Input mismatch vs B seed{seed} step{offset}'
        journal(out,'batch_fingerprints',step=offset,sha256=fingerprint)
        t=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        sa=batch['student_audio'].to(device);sl=batch['student_lengths'].to(device);labels=batch['labels'].to(device)
        te=None
        if teacher is not None:
            with torch.random.fork_rng(devices=[0]),torch.no_grad(),torch.amp.autocast('cuda',dtype=torch.float16):
                torch.manual_seed(hist.stable_int(f'HC0032:teacher:{seed}:{offset}'))
                te=torch.nn.functional.normalize(teacher(batch['teacher_audio'].to(device),batch['teacher_lengths'].to(device)).float(),dim=-1)
        with torch.random.fork_rng(devices=[0]),torch.amp.autocast('cuda',dtype=torch.float16):
            torch.manual_seed(hist.stable_int(f'HC0032:student:{seed}:{offset}'))
            se=torch.nn.functional.normalize(student(sa,sl).float(),dim=-1)
            aam,_=head(se,labels);vector=se.new_zeros(());kl=se.new_zeros(())
            if model in ['C','D']:vector=hist.vector_alignment_loss(te,se)
            if model=='D':kl=hist.symmetric_diagonal_gaussian_kl(te,se,a.kl_epsilon)
            loss=aam+a.lambda_vector*vector+a.lambda_kl*kl
        assert torch.isfinite(loss),'Nonfinite objective'
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        encoder_norm=torch.nn.utils.clip_grad_norm_(student.parameters(),a.gradient_clip)
        head_norm=torch.nn.utils.clip_grad_norm_(head.parameters(),a.gradient_clip)
        assert torch.isfinite(encoder_norm) and torch.isfinite(head_norm),'Nonfinite gradients'
        before=scaler.get_scale();scaler.step(optimizer);scaler.update()
        assert scaler.get_scale()>=before,'AMP skipped update; abort to preserve matched update/schedule count'
        scheduler.step()
        if teacher is not None:hist.ema_update(teacher,student,a.ema_alpha)
        torch.cuda.synchronize();timings.append(time.perf_counter()-t)
        journal(out,'training_metrics',step=offset,loss=float(loss),aam=float(aam),vector=float(vector),kl=float(kl),
            step_seconds=timings[-1],peak_allocated_bytes=torch.cuda.max_memory_allocated(),lr_encoder=optimizer.param_groups[0]['lr'])
        if offset%20==0 or mode=='smoke':print(model,seed,offset,float(loss),timings[-1],flush=True)
        if mode=='train' and offset%a.eval_every==0:
            torch.manual_seed(hist.stable_int(f'HC0032:validation:{seed}'))
            metrics=hist.evaluate_verification(student,student,assignments,augmenter,a,device)
            metrics['step']=offset;history.append(metrics);journal(out,'validation_metrics',**metrics)
            candidate=out/f'checkpoint_step{offset:05d}_{time.time_ns()}.pt'
            if float(metrics['noise_eer'])<best:best=float(metrics['noise_eer']);best_checkpoint=str(candidate)
            payload={'system':{'B':'AAM_ONLY','C':'C_VECTOR_EMA','D':'D_FULL'}[model],
                'student':student.state_dict(),'teacher':None if teacher is None else teacher.state_dict(),
                'aam_head':head.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
                'global_step':offset,'best_noise_eer':best,'best_checkpoint':best_checkpoint,'history':history,
                'signature':{**cfg['training'],'train_speaker_ids':train},'config_sha256':sha(config_path),
                'initialization':cfg['initialization'],'cohort':cfg['cohort']}
            staging=out/f'.staging_{offset}_{time.time_ns()}.pt'
            with staging.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
            os.link(staging,candidate);pointer(out/'resume_pointer.json',record(candidate))
            journal(out,'events',event='CHECKPOINT_COMPLETE',checkpoint=record(candidate),step=offset,best_checkpoint=best_checkpoint)
            student.train();head.train()
    result={'status':'passed' if mode=='smoke' else 'completed','utc':now(),'seed':seed,'model':model,'mode':mode,
        'device':'cuda','config_sha256':sha(config_path),'global_step':limit,'actual_speakers_per_batch':16,'actual_batch_size':32,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'mean_step_seconds':sum(timings)/len(timings) if timings else None,'elapsed_seconds':time.perf_counter()-elapsed_start,
        'initialization':cfg['initialization'],'parent_consumed_input_files_hash_verified':len(consumed),
        'worker_input_policy':'Every distinct audio/noise path is SHA-256 checked per loader process before first use; worker counts are process-local',
        'GPU_name':torch.cuda.get_device_name(0),'software':environment,
        'code_revision':'See preparation/source_snapshot.json and preparation/release.json',
        'selection_rule':cfg['selection_rule'],'active_processes':subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv'],capture_output=True,text=True).stdout}
    if mode=='train':
        result['selected_checkpoint']=record(best_checkpoint);result['best_noisy_development_eer']=best
        save(out/'selection.json',{'checkpoint':record(best_checkpoint),'rule':cfg['selection_rule'],'noise_eer':best,'history':history})
    save(final,result)
    journal(out,'events',event='COMPLETED',completion=str(final));print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);p.add_argument('--mode',choices=['smoke','train'],required=True);args=p.parse_args()
    cfg=load(args.config)
    assert cfg['training']['seed'] in [2027,2028]
    assert (cfg['training']['lambda_vector'],cfg['training']['lambda_kl'])=={'B':(0.,0.),'C':(1.,0.),'D':(1.,.01)}[cfg['paper_system']]
    lock_path=ROOT/'experiments/evaluation/watchdogs/.locks/arch_b_gpu_watchdog_20260921.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check_priority_and_gpu()
        try:run(cfg,args.config,args.mode)
        except Exception as error:
            save(BASE/f'logs/failure_{cfg["paper_system"]}_{cfg["training"]["seed"]}_{time.time_ns()}.json',
                {'status':'failed','error':repr(error),'command':[sys.executable,*sys.argv]})
            raise
