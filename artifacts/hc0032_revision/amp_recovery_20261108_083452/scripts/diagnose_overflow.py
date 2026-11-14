import sys,fcntl,traceback
sys.path.insert(0,'/media/ml4u/Samsung_T52/ptnk_team/mortyhuaphuoc/artifacts/hc0032_revision/amp_recovery_20261108_083452/scripts')
import engine_recovery as e
from engine_recovery import *
import qualify_capacity as q
OUT=RECOVERY_ROOT/'diagnostic_B8344'
def restore(payload,cfg,a,train):
    state=build(cfg,a,train)
    state[0].load_state_dict(payload['student']);state[1].load_state_dict(payload['aam_head'])
    if state[2] is not None:state[2].load_state_dict(payload['teacher']);state[2].cpu()
    state[3].load_state_dict(payload['optimizer']);stage_optimizer_cpu(state[3]);state[4].load_state_dict(payload['scheduler']);state[5].load_state_dict(payload['scaler']);restore_rng(payload['rng_state'])
    return state
def run():
    cfg=load(Path('/media/ml4u/Samsung_T52/ptnk_team/mortyhuaphuoc/artifacts/hc0032_revision/fixed_capacity_20261108_060219/configs/B_seed2027.json'));verify_inputs(cfg)
    a,records,train,ds,aug,assignments,used=make_data(cfg)
    ptr=load(Path('/media/ml4u/Samsung_T52/ptnk_team/mortyhuaphuoc/artifacts/hc0032_revision/fixed_capacity_20261108_060219/runs/seed2027/B/resume_pointer.json'));verify(ptr);payload=torch.load(ptr['path'],map_location='cpu',weights_only=False);assert payload['global_step']==8000
    state=restore(payload,cfg,a,train);gen=torch.Generator();gen.set_state(payload['loader_generator_state'])
    dl,gen=loader(ds,train,a,8000,344,gen);restore_rng(payload['rng_state']);del payload
    originals={x['step']:x for x in [__import__('json').loads(s) for s in Path('/media/ml4u/Samsung_T52/ptnk_team/mortyhuaphuoc/artifacts/hc0032_revision/fixed_capacity_20261108_060219/runs/seed2027/B/training_metrics.jsonl').read_text().splitlines()]}
    fingerprints={x['step']:x['sha256'] for x in [__import__('json').loads(s) for s in Path('/media/ml4u/Samsung_T52/ptnk_team/mortyhuaphuoc/artifacts/hc0032_revision/fixed_capacity_20261108_060219/runs/seed2027/B/batch_fingerprints.jsonl').read_text().splitlines()]}
    for index,batch in enumerate(dl,8001):
        assert batch_hash(batch)==fingerprints[index]
        if index<8344:
            metrics,_=previous.step(state,batch,cfg,a,index,True)
            for key in ['loss','aam','vector','kl','encoder_gradient_norm','head_gradient_norm','AMP_scale']:assert metrics[key]==originals[index][key],(index,key,metrics[key],originals[index][key])
            if index%50==0:event(OUT,'REPLAY_PREFIX_EXACT',step=index)
            continue
        before={'student':cpu_state(state[0]),'aam_head':cpu_state(state[1]),'teacher':None,'optimizer':copy.deepcopy(state[3].state_dict()),'scheduler':copy.deepcopy(state[4].state_dict()),'scaler':state[5].state_dict().copy(),'rng_state':rng_state()}
        path=OUT/'before_step08344.pt'
        with path.open('xb') as f:torch.save(before,f)
        metrics,actual=e.step(state,batch,cfg,a,index,True,True);assert metrics['overflow_retries'],'Original failure was not reproduced'
        cleanup(state);del state;gc.collect();torch.cuda.empty_cache()
        before['scaler']['scale']=metrics['AMP_scale'];before['scaler']['_growth_tracker']=0
        reference=restore(before,cfg,a,train);ref_metrics,expected=previous.step(reference,batch,cfg,a,index,True,True)
        result=q.compare(actual,expected);result.update(loss_absolute_delta=abs(metrics['loss']-ref_metrics['loss']),metrics=metrics,reference_metrics=ref_metrics,original_successful_prefix_exact_steps=343,failing_step=index,original_resume=ptr,prestep_checkpoint=record(path),batch_sha256=batch_hash(batch))
        result['passed']=result['status']=='passed' and result['loss_absolute_delta']==0
        save(OUT/'result.json',result);event(OUT,'RECOVERY_VERIFIED',result=result);cleanup(reference);assert result['passed']
if __name__=='__main__':
    OUT.mkdir(parents=True)
    with (ROOT/'experiments/evaluation/watchdogs/.locks/arch_b_gpu_watchdog_20260921.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            gate=gpu_guard();event(OUT,'STARTED',command=[sys.executable,*sys.argv],resource_gate=gate);run()
        except BaseException as error:
            save(OUT/'failure.json',{'error':repr(error),'traceback':traceback.format_exc(),'gpu':gpu()});raise
