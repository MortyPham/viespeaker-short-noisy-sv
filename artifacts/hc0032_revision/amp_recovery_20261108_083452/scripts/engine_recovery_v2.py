"""Common full-batch FP16 overflow recovery; no skipped scientific update."""
import sys
sys.path.insert(0,'/media/ml4u/Samsung_T52/ptnk_team/mortyhuaphuoc/artifacts/hc0032_revision/fixed_capacity_20261108_060219/scripts')
import engine_fixed as _staged_engine
from engine_fixed import *
RECOVERY_ROOT=Path(__file__).resolve().parents[1]

def buffers(state):
    return [None if m is None else {k:v.detach().cpu().clone() for k,v in m.named_buffers()} for m in state[:3]]
def restore_buffers(state,saved):
    with torch.no_grad():
        for m,values in zip(state[:3],saved):
            if m is None:continue
            for k,v in m.named_buffers():v.copy_(values[k].to(v.device))
def step(state,batch,cfg,a,index,offload=True,capture=False):
    pre_buffers=buffers(state);pre_rng=rng_state();pre_scaler=state[5].state_dict().copy();retries=[]
    for attempt in range(9):
        try:
            metrics,snapshot=_staged_engine.step(state,batch,cfg,a,index,offload,capture)
            metrics['overflow_retries']=retries
            return metrics,snapshot
        except AssertionError as error:
            if str(error)!='Nonfinite gradients' or attempt==8:raise
            old_scale=state[5].get_scale();new_scale=old_scale/2
            if new_scale<1:raise
            # Failure occurs before optimizer, scheduler, or EMA update. The failed
            # forward's buffers/RNG must be restored before the identical batch retry.
            restore_buffers(state,pre_buffers);restore_rng(pre_rng)
            state[3].zero_grad(set_to_none=True);stage_optimizer_cpu(state[3])
            if state[2] is not None:state[2].cpu()
            # update(new_scale) clears per-optimizer unscale state. Restore the
            # successful-prefix scaler metadata but restart its growth counter.
            state[5].update(new_scale=new_scale)
            scaled={**pre_scaler,'scale':new_scale,'_growth_tracker':0}
            state[5].load_state_dict(scaled)
            row={'step':index,'seed':a.seed,'model':cfg['paper_system'],'attempt':attempt+1,'old_scale':old_scale,'new_scale':new_scale,'batch_sha256':batch_hash(batch),'policy':'Restore all model buffers/RNG; identical full batch; no optimizer/scheduler/EMA advance on failed attempt'}
            retries.append(row);event(RECOVERY_ROOT/'retry_events','AMP_OVERFLOW_RETRY',**row)
            gc.collect();torch.cuda.empty_cache()
