"""Lossless state staging; unchanged full32GPU computations and objective math."""
import sys
sys.path.insert(0,'/media/ml4u/Samsung_T52/ptnk_team/mortyhuaphuoc/artifacts/hc0032_revision/cohort_finish_20261107_211439/matched')
import engine as reference_engine
import engine_v5 as previous
from engine_v5 import *

def gpu_guard():
    before=gpu();assert torch.cuda.is_available()
    budget=before['free_mib']-512-512
    assert budget>=512
    torch.cuda.set_per_process_memory_fraction(budget/(torch.cuda.get_device_properties(0).total_memory/2**20),0)
    x=torch.empty(1,device='cuda');del x;torch.cuda.empty_cache()
    return {'before':before,'after_context':gpu(),'allocator_budget_mib':budget,'physical_headroom_mib':512,'nonallocator_allowance_mib':512,'policy':'Fixed capacity, bounded lossless state staging; no utilization gate'}

def stage_optimizer_cpu(opt):
    for state in opt.state.values():
        for key,value in state.items():
            if torch.is_tensor(value) and value.is_cuda:state[key]=value.to('cpu')

def restore_optimizer_cuda(opt):
    for parameter,state in opt.state.items():
        for key,value in state.items():
            # AdamW noncapturable scalar step is CPU in the original implementation.
            if torch.is_tensor(value) and key!='step':state[key]=value.to(parameter.device)

def staged_step(state,batch,cfg,a,index,offload=True,capture=False):
    student,head,teacher,opt,sched,scaler=state
    opt.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
    before={'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),'optimizer_bytes':bytes_in(opt.state)}
    if teacher is not None:teacher.to('cuda')
    sa=batch['student_audio'].to('cuda');sl=batch['student_lengths'].to('cuda');labels=batch['labels'].to('cuda');te=None
    if teacher is not None:
        with torch.random.fork_rng(devices=[0]),torch.no_grad(),torch.amp.autocast('cuda',dtype=torch.float16):
            torch.manual_seed(hist.stable_int(f'HC0032:teacher:{a.seed}:{index}'))
            te=torch.nn.functional.normalize(teacher(batch['teacher_audio'].to('cuda'),batch['teacher_lengths'].to('cuda')).float(),dim=-1)
    if teacher is not None:teacher.cpu()
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
    restore_optimizer_cuda(opt)
    scale=scaler.get_scale();scaler.step(opt);scaler.update();assert scaler.get_scale()>=scale,'AMP skipped update'
    sched.step()
    if teacher is not None:
        teacher.to('cuda');hist.ema_update(teacher,student,a.ema_alpha);teacher.cpu()
    stage_optimizer_cpu(opt)
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
def step(state,batch,cfg,a,index,offload=True,capture=False):
    if not offload:return reference_engine.step(state,batch,cfg,a,index,False,capture)
    with memory_forward(state[0],state[2]):
        return staged_step(state,batch,cfg,a,index,True,capture)
