import sys,fcntl,traceback
sys.path.insert(0,'/media/ml4u/Samsung_T52/ptnk_team/mortyhuaphuoc/artifacts/hc0032_revision/amp_recovery_20261108_083452/scripts')
import engine_recovery_v2 as e
from engine_recovery_v2 import *
import qualify_capacity as q
OUT=RECOVERY_ROOT/'diagnostic_B8344_controlled_v2'
def restore(payload,cfg,a,train):
    state=build(cfg,a,train)
    state[0].load_state_dict(payload['student']);state[1].load_state_dict(payload['aam_head'])
    if state[2] is not None:state[2].load_state_dict(payload['teacher']);state[2].cpu()
    state[3].load_state_dict(payload['optimizer']);stage_optimizer_cpu(state[3]);state[4].load_state_dict(payload['scheduler']);state[5].load_state_dict(payload['scaler']);restore_rng(payload['rng_state'])
    return state
def run():
    cfg=load(Path('/media/ml4u/Samsung_T52/ptnk_team/mortyhuaphuoc/artifacts/hc0032_revision/fixed_capacity_20261108_060219/configs/B_seed2027.json'));verify_inputs(cfg)
    a,records,train,ds,aug,assignments,used=make_data(cfg)
    specs=next(iter(hist.SpeakerBalancedBatchSampler(train,1,16,2,a.seed,start_step=8343)))
    batch=hist.collate_pairs([ds[x] for x in specs]);expected={x['step']:x['sha256'] for x in [json.loads(s) for s in Path('/media/ml4u/Samsung_T52/ptnk_team/mortyhuaphuoc/artifacts/hc0032_revision/fixed_capacity_20261108_060219/runs/seed2027/B/batch_fingerprints.jsonl').read_text().splitlines()]};assert batch_hash(batch)==expected[8344]
    original=torch.load(RECOVERY_ROOT/'diagnostic_B8344_v3/before_step08344.pt',map_location='cpu',weights_only=False)
    trials=[]
    for scale in [32768.,65536.,131072.,262144.]:
        before=copy.deepcopy(original);before['scaler']['scale']=scale;before['scaler']['_growth_tracker']=0
        state=restore(before,cfg,a,train);metrics,actual=e.step(state,batch,cfg,a,8344,True,True)
        trials.append({'initial_scale':scale,'metrics':metrics});event(OUT,'CONTROLLED_SCALE_TRIAL',initial_scale=scale,metrics=metrics)
        if not metrics['overflow_retries']:
            cleanup(state);del state;gc.collect();torch.cuda.empty_cache();continue
        # Use the same module objects and allocation pool for a contemporaneous
        # independent single-attempt update from the exact same pre-step tensor state.
        before=copy.deepcopy(original)
        state[0].load_state_dict(before['student']);state[1].load_state_dict(before['aam_head']);state[3].load_state_dict(before['optimizer']);stage_optimizer_cpu(state[3]);state[4].load_state_dict(before['scheduler']);state[5].load_state_dict({**before['scaler'],'scale':metrics['AMP_scale'],'_growth_tracker':0});restore_rng(before['rng_state'])
        reference_metrics,expected_snapshot=e._staged_engine.step(state,batch,cfg,a,8344,True,True)
        result=q.compare(actual,expected_snapshot);result.update(loss_absolute_delta=abs(metrics['loss']-reference_metrics['loss']),metrics=metrics,reference_metrics=reference_metrics,trials=trials,batch_sha256=batch_hash(batch),failing_step=8344,actual_original_failure_reproduced=False,original_failure_scale=16384.,controlled_overflow_initial_scale=scale,scope='Controlled actual AMP overflow on the frozen original failed batch; original trained-prefix replay has numeric differences and did not fail at16384.')
        result['passed']=result['status']=='passed' and result['loss_absolute_delta']==0
        save(OUT/'result.json',result);event(OUT,'RECOVERY_VERIFIED',result=result);cleanup(state);assert result['passed'];return
    raise AssertionError('No controlled overflow within bounded scale trials')
if __name__=='__main__':
    OUT.mkdir(parents=True)
    with (ROOT/'experiments/evaluation/watchdogs/.locks/arch_b_gpu_watchdog_20260921.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            gate=gpu_guard();event(OUT,'STARTED',command=[sys.executable,*sys.argv],resource_gate=gate);run()
        except BaseException as error:
            save(OUT/'failure.json',{'error':repr(error),'traceback':traceback.format_exc(),'gpu':gpu()});raise
