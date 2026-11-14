import sys,fcntl,traceback
sys.path.insert(0,'/media/ml4u/Samsung_T52/ptnk_team/mortyhuaphuoc/artifacts/hc0032_revision/amp_recovery_20261108_083452/scripts')
import engine_recovery_v2 as e
from engine_recovery_v2 import *
import qualify_capacity as q
OUT=RECOVERY_ROOT/'D_retry_and_resume_validation'
def restore(payload,cfg,a,train):
    state=build(cfg,a,train);state[0].load_state_dict(payload['student']);state[1].load_state_dict(payload['head']);state[2].load_state_dict(payload['teacher']);state[2].cpu()
    state[3].load_state_dict(payload['optimizer']);stage_optimizer_cpu(state[3]);state[4].load_state_dict(payload['scheduler']);state[5].load_state_dict(payload['scaler']);restore_rng(payload['rng'])
    return state
def snap(state):
    return {'student':cpu_state(state[0]),'head':cpu_state(state[1]),'teacher':cpu_state(state[2]),'optimizer':copy.deepcopy(state[3].state_dict()),'scheduler':copy.deepcopy(state[4].state_dict()),'scaler':state[5].state_dict().copy(),'rng':rng_state()}
def run():
    cfg=load(Path('/media/ml4u/Samsung_T52/ptnk_team/mortyhuaphuoc/artifacts/hc0032_revision/fixed_capacity_20261108_060219/configs/D_seed2027.json'));verify_inputs(cfg);a,records,train,ds,aug,assignments,used=make_data(cfg)
    batches=[hist.collate_pairs([ds[x] for x in specs]) for specs in hist.SpeakerBalancedBatchSampler(train,2,16,2,a.seed)]
    state=build(cfg,a,train);before=snap(state)
    original=torch.nn.utils.clip_grad_norm_;calls=[0]
    def injected(*args,**kwargs):
        value=original(*args,**kwargs);calls[0]+=1
        return value.new_tensor(float('inf')) if calls[0]==1 else value
    torch.nn.utils.clip_grad_norm_=injected
    try:metrics,actual=e.step(state,batches[0],cfg,a,1,True,True)
    finally:torch.nn.utils.clip_grad_norm_=original
    assert len(metrics['overflow_retries'])==1
    # Save an actual successful retry checkpoint, then verify resumed nextupdate.
    payload=snap(state);path=OUT/'after_retry_step00001.pt'
    with path.open('xb') as f:torch.save(payload,f)
    second,continuous=e.step(state,batches[1],cfg,a,2,True,True)
    cleanup(state);del state;gc.collect();torch.cuda.empty_cache()
    reference=restore(before,cfg,a,train);reference[5].load_state_dict({**before['scaler'],'scale':512.,'_growth_tracker':0})
    reference_metrics,expected=e._staged_engine.step(reference,batches[0],cfg,a,1,True,True)
    comparison=q.compare(actual,expected);comparison['loss_absolute_delta']=abs(metrics['loss']-reference_metrics['loss']);assert comparison['status']=='passed' and comparison['loss_absolute_delta']==0
    cleanup(reference);del reference;gc.collect();torch.cuda.empty_cache()
    resumed=restore(torch.load(path,map_location='cpu',weights_only=False),cfg,a,train);resumed_metrics,resumed_snapshot=e.step(resumed,batches[1],cfg,a,2,True,True)
    resume=q.compare(continuous,resumed_snapshot);resume['loss_absolute_delta']=abs(second['loss']-resumed_metrics['loss']);assert resume['status']=='passed' and resume['loss_absolute_delta']==0
    save(OUT/'result.json',{'status':'passed','synthetic_failed_attempt':'Injected nonfinite gradient norm before optimizer/scheduler/EMA, after real forward/backward','real_waveforms':True,'full_batch_size':32,'retry_vs_one_halfscale_update':comparison,'checkpoint_resume':resume,'retry_metrics':metrics,'checkpoint':record(path)})
    cleanup(resumed);event(OUT,'COMPLETED',result=load(OUT/'result.json'))
if __name__=='__main__':
    OUT.mkdir(parents=True)
    with (ROOT/'experiments/evaluation/watchdogs/.locks/arch_b_gpu_watchdog_20260921.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:gate=gpu_guard();event(OUT,'STARTED',command=[sys.executable,*sys.argv],resource_gate=gate);run()
        except BaseException as error:save(OUT/'failure.json',{'error':repr(error),'traceback':traceback.format_exc(),'gpu':gpu()});raise
