"""Independent two-step actualDfull32checkpoint/resume parity; never a seed replicate."""
import fcntl,traceback
from engine_v5 import *
from validate_memory_v5 import compare
OUT=RUN/'checkpoint_validation_D'
def test():
    cfg=load(BASE/'configs_v5/D_seed2027.json');verify_inputs(cfg)
    a,records,train,ds,aug,assignments,used=make_data(cfg)
    batches=[hist.collate_pairs([ds[x] for x in specs]) for specs in hist.SpeakerBalancedBatchSampler(train,2,16,2,a.seed)]
    first=build(cfg,a,train);m1,_=step(first,batches[0],cfg,a,1,True)
    gen=torch.Generator().manual_seed(hist.stable_int(f'HC0032:loader:{a.seed}'))
    ck=checkpoint(OUT,first,cfg,BASE/'configs_v5/D_seed2027.json',1,[],float('inf'),None,gen)
    m2,uninterrupted=step(first,batches[1],cfg,a,2,True,True)
    cleanup(first);del first;gc.collect();torch.cuda.empty_cache()
    restored=build(cfg,a,train)
    saved=torch.load(ck['path'],map_location='cpu',weights_only=False)
    assert saved['sampler_state']['next_start_step']==1 and saved['global_step']==1
    restored[0].load_state_dict(saved['student'],strict=True);restored[1].load_state_dict(saved['aam_head'],strict=True);restored[2].load_state_dict(saved['teacher'],strict=True)
    restored[3].load_state_dict(saved['optimizer']);restored[4].load_state_dict(saved['scheduler']);restored[5].load_state_dict(saved['scaler'])
    gen.set_state(saved['loader_generator_state']);restore_rng(saved['rng_state']);del saved
    m3,resumed=step(restored,batches[1],cfg,a,2,True,True)
    result=compare(uninterrupted,resumed);result.update(loss_absolute_delta=abs(m2['loss']-m3['loss']),checkpoint=ck,
        metrics=[m1,m2,m3],actual_batch_size=32,model='D',seed=2027,scope='Two-step qualification/checkpoint mechanics only; not full training or a completed seed block',
        checked_state=['student','head','teacher/EMA','optimizer','scheduler','GradScaler','PythonNumPyTorchCPU/CUDARNG','sampler nextstep','loader generator'])
    result['passed']=result['status']=='passed' and result['loss_absolute_delta']==0
    save(OUT/'result.json',result);cleanup(restored)
    assert result['passed'],'Checkpoint resume failed strict equivalence'
    event(OUT,'COMPLETE',result=result)
if __name__=='__main__':
    OUT.mkdir()
    with (ROOT/'experiments/evaluation/watchdogs/.locks/arch_b_gpu_watchdog_20260921.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:gate=gpu_guard();event(OUT,'STARTED',gate=gate,command=[sys.executable,*sys.argv]);test()
        except BaseException as error:
            save(OUT/'failure.json',{'error':repr(error),'traceback':traceback.format_exc(),'gpu':gpu(),'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20});raise
