"""Full-batch matched training primitives; lossless saved-tensor CPU storage."""
import copy,contextlib,datetime,gc,importlib.metadata,math,os,random,subprocess,time
from common import *
import numpy as np
torch.set_num_threads(2)
RUN=BASE.parent
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def gpu():
    s=subprocess.run(['nvidia-smi','--query-gpu=memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
    total,free,util=[int(x.strip()) for x in s.splitlines()[0].split(',')]
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
    return {'total_mib':total,'free_mib':free,'utilization_percent':util,'compute_processes':apps}
def event(out,kind,**kw):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    row={'utc':now(),'pid':os.getpid(),'event':kind,**kw}
    with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row,default=str)+'\n')
    print(json.dumps(row,default=str),flush=True)
def pointer(path,obj):
    tmp=Path(path).with_name(Path(path).name+f'.{time.time_ns()}.tmp')
    with tmp.open('x') as f:json.dump(obj,f,indent=2,default=str)
    os.replace(tmp,path)
def gpu_guard():
    assert torch.cuda.is_available(),'CUDA required; no CPU fallback'
    before=gpu()
    # Reserve context/library memory OUTSIDE the allocator, using the prior ~0.5GiB footprint.
    budget=before['free_mib']-1536-512
    assert budget>=512,'No bounded CUDA slot after headroom/context allowance'
    torch.cuda.set_per_process_memory_fraction(budget/(torch.cuda.get_device_properties(0).total_memory/2**20),0)
    probe=torch.empty(1,device='cuda');del probe;torch.cuda.empty_cache()
    return {'before':before,'after_context':gpu(),'allocator_budget_mib':budget,'physical_headroom_mib':1536,'nonallocator_allowance_mib':512,'utilization_gate':None}
def rng_state():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state_all()}
def restore_rng(x):
    random.setstate(x['python']);np.random.set_state(x['numpy']);torch.set_rng_state(x['torch_cpu']);torch.cuda.set_rng_state_all(x['torch_cuda'])
def cpu_state(module):return {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}
def bytes_in(x):
    if torch.is_tensor(x):return x.numel()*x.element_size()
    if isinstance(x,dict):return sum(bytes_in(v) for v in x.values())
    if isinstance(x,(list,tuple)):return sum(bytes_in(v) for v in x)
    return 0
def verify_inputs(cfg):
    for item in cfg['inputs']:verify(item)
    verify(cfg['initialization'])
    for item in load(BASE/'preparation/source_snapshot.json'):verify(item)
    for item in load(BASE/'preparation/release.json'):verify(item)
    expected=load(BASE/'preparation/software_environment.json')['packages']
    for name,version in expected.items():assert importlib.metadata.version(name)==version,(name,version)
    registry=load(ROOT/'artifacts/hc0032_revision/prompt1_20261106_2153/final/model_registry.json')
    for item in registry['models']['A']['pretrained_bundle']['files']:
        assert sha(Path(cfg['pretrained_bundle'])/Path(item['path']).name)==item['sha256']
def make_data(cfg):
    a=args_from(cfg);records,train,val=data(a)
    assert train==load(BASE/'preparation/populations.json')['train_speaker_ids']
    hashes={}
    import csv
    with (BASE/'preparation/audio_inputs.csv').open() as f:
        for row in csv.DictReader(f):hashes[row['path']]=row['sha256']
    for row in load(BASE/'preparation/noise_inputs.json'):hashes[row['path']]=row['sha256']
    consumed=set();original=hist.load_audio
    def checked(path,sr):
        path=str(path);assert path in hashes,path
        if path not in consumed:assert sha(path)==hashes[path],path;consumed.add(path)
        return original(path,sr)
    hist.load_audio=checked
    aug=hist.MusanAugmenter(a.musan_dir,a)
    ds=hist.SpeakerPairDataset(records,{s:i for i,s in enumerate(train)},aug,a)
    assignments=hist.create_validation_assignments(records,val,a)
    return a,records,train,ds,aug,assignments,consumed
def build(cfg,a,train):
    hist.seed_everything(a.seed)
    initial=torch.load(cfg['initialization']['path'],map_location='cpu',weights_only=False)
    student=hist.load_pretrained_encoder(a).cpu().eval();student.load_state_dict(initial['student'],strict=True)
    head=hist.AAMSoftmaxHead(192,len(train),a.aam_margin,a.aam_scale);head.load_state_dict(initial['aam_head'],strict=True)
    assert tensor_hash(student.state_dict())==cfg['initialization']['student_tensor_sha256']
    assert tensor_hash(head.state_dict())==cfg['initialization']['head_tensor_sha256']
    del initial
    student=student.to('cuda').train();head=head.to('cuda').train()
    teacher=None if cfg['paper_system']=='B' else copy.deepcopy(student).eval().requires_grad_(False)
    opt=torch.optim.AdamW([{'params':student.parameters(),'lr':a.encoder_lr},{'params':head.parameters(),'lr':a.head_lr}],weight_decay=a.weight_decay,betas=(.9,.999),eps=1e-8)
    def multiplier(step):
        if step<a.warmup_steps:return max(step,1)/max(a.warmup_steps,1)
        return .5*(1+math.cos(math.pi*min(max((step-a.warmup_steps)/(a.steps-a.warmup_steps),0),1)))
    sched=torch.optim.lr_scheduler.LambdaLR(opt,multiplier)
    scaler=torch.amp.GradScaler('cuda',enabled=True,init_scale=1024.)
    torch.backends.cudnn.benchmark=False
    return student,head,teacher,opt,sched,scaler
def step(state,batch,cfg,a,index,offload=True,capture=False):
    student,head,teacher,opt,sched,scaler=state
    opt.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
    before={'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),'optimizer_bytes':bytes_in(opt.state)}
    sa=batch['student_audio'].to('cuda');sl=batch['student_lengths'].to('cuda');labels=batch['labels'].to('cuda');te=None
    if teacher is not None:
        with torch.random.fork_rng(devices=[0]),torch.no_grad(),torch.amp.autocast('cuda',dtype=torch.float16):
            torch.manual_seed(hist.stable_int(f'HC0032:teacher:{a.seed}:{index}'))
            te=torch.nn.functional.normalize(teacher(batch['teacher_audio'].to('cuda'),batch['teacher_lengths'].to('cuda')).float(),dim=-1)
    storage=torch.autograd.graph.save_on_cpu(pin_memory=False) if offload else contextlib.nullcontext()
    with storage:
        with torch.random.fork_rng(devices=[0]),torch.amp.autocast('cuda',dtype=torch.float16):
            torch.manual_seed(hist.stable_int(f'HC0032:student:{a.seed}:{index}'))
            se=torch.nn.functional.normalize(student(sa,sl).float(),dim=-1)
            aam,logits=head(se,labels);vector=se.new_zeros(());kl=se.new_zeros(())
            if teacher is not None:vector=hist.vector_alignment_loss(te,se)
            if cfg['paper_system']=='D':kl=hist.symmetric_diagonal_gaussian_kl(te,se,a.kl_epsilon)
            loss=aam+a.lambda_vector*vector+a.lambda_kl*kl
        assert torch.isfinite(loss),'Nonfinite loss'
        scaler.scale(loss).backward()
    scaler.unscale_(opt)
    grads={f'student.{k}':v.grad.detach().cpu().clone() for k,v in student.named_parameters() if v.grad is not None} if capture else None
    if capture:grads.update({f'head.{k}':v.grad.detach().cpu().clone() for k,v in head.named_parameters() if v.grad is not None})
    encoder_norm=torch.nn.utils.clip_grad_norm_(student.parameters(),a.gradient_clip);head_norm=torch.nn.utils.clip_grad_norm_(head.parameters(),a.gradient_clip)
    assert torch.isfinite(encoder_norm) and torch.isfinite(head_norm),'Nonfinite gradients'
    scale=scaler.get_scale();scaler.step(opt);scaler.update();assert scaler.get_scale()>=scale,'AMP skipped update'
    sched.step()
    if teacher is not None:hist.ema_update(teacher,student,a.ema_alpha)
    torch.cuda.synchronize()
    result={'step':index,'loss':float(loss),'aam':float(aam),'vector':float(vector),'kl':float(kl),'encoder_gradient_norm':float(encoder_norm),'head_gradient_norm':float(head_norm),
        'elapsed_seconds':time.perf_counter()-t,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
        'before':before,'optimizer_bytes_after':bytes_in(opt.state),'student_padded_seconds':batch['student_audio'].shape[1]/16000,'teacher_padded_seconds':batch['teacher_audio'].shape[1]/16000,'AMP_scale':scaler.get_scale()}
    snapshot=None
    if capture:
        snapshot={'gradients':grads,'student':cpu_state(student),'head':cpu_state(head),'teacher':None if teacher is None else cpu_state(teacher),
            'rng':rng_state(),'scaler':scaler.state_dict(),'scheduler':sched.state_dict()}
    # The function boundary releases all computation outputs before the next forward.
    del sa,sl,labels,te,se,aam,logits,vector,kl,loss,encoder_norm,head_norm
    opt.zero_grad(set_to_none=True)
    result['allocated_after_clear_mib']=torch.cuda.memory_allocated()/2**20
    return result,snapshot
def loader(ds,train,a,start,count,generator=None):
    gen=generator or torch.Generator().manual_seed(hist.stable_int(f'HC0032:loader:{a.seed}'))
    sampler=hist.SpeakerBalancedBatchSampler(train,count,16,2,a.seed,start_step=start)
    options={'num_workers':a.num_workers,'pin_memory':True,'generator':gen}
    if a.num_workers:options.update(persistent_workers=True,prefetch_factor=2)
    return torch.utils.data.DataLoader(ds,batch_sampler=sampler,collate_fn=hist.collate_pairs,**options),gen
def checkpoint(out,state,cfg,config_path,index,history,best,bestpath,generator):
    student,head,teacher,opt,sched,scaler=state
    candidate=Path(out)/f'checkpoint_step{index:05d}_{time.time_ns()}.pt'
    payload={'system':{'B':'AAM_ONLY','C':'C_VECTOR_EMA','D':'D_FULL'}[cfg['paper_system']], 'student':student.state_dict(),'aam_head':head.state_dict(),'teacher':None if teacher is None else teacher.state_dict(),
        'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'scaler':scaler.state_dict(),'global_step':index,'signature':cfg['training'],'cohort':cfg['cohort'],
        'config_sha256':sha(config_path),'initialization':cfg['initialization'],'history':history,'best_noise_eer':best,'best_checkpoint':bestpath,
        'rng_state':rng_state(),'loader_generator_state':generator.get_state(),'sampler_state':{'seed':cfg['training']['seed'],'next_start_step':index,'speakers_per_batch':16,'pairs_per_speaker':2},
        'execution_memory':'lossless saved tensors on CPU; no recomputation, unchanged full32batch and objectives'}
    with candidate.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
    pointer(Path(out)/'resume_pointer.json',record(candidate));return record(candidate)
def cleanup(state):
    if state is not None:
        student,head,teacher,opt,sched,scaler=state
        opt.zero_grad(set_to_none=True);student.cpu();head.cpu()
        if teacher is not None:teacher.cpu()
        opt.state.clear()
    gc.collect();torch.cuda.empty_cache()
