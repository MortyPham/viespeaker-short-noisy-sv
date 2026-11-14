"""Actual32-example semantic equivalence and representative CUDA capacity gates."""
import argparse,fcntl,hashlib,pickle,traceback
from engine_v5 import *
OUT=RUN/'validation_v5'
def snapshot_rng_equal(a,b):
    return a['python']==b['python'] and a['numpy'][0]==b['numpy'][0] and np.array_equal(a['numpy'][1],b['numpy'][1]) and a['numpy'][2:]==b['numpy'][2:] and torch.equal(a['torch_cpu'],b['torch_cpu']) and all(torch.equal(x,y) for x,y in zip(a['torch_cuda'],b['torch_cuda'],strict=True))
def compare(a,b):
    result={};passed=True
    for group in ['gradients','student','head','teacher']:
        if a[group] is None:assert b[group] is None;continue
        assert set(a[group])==set(b[group]);max_delta=0.;exact=0;allclose=True
        for k,x in a[group].items():
            y=b[group][k];exact+=int(torch.equal(x,y))
            if x.dtype.is_floating_point:
                d=float((x-y).abs().max());max_delta=max(max_delta,d)
                ok=torch.allclose(x,y,rtol=1e-5,atol=1e-7)
            else:ok=torch.equal(x,y)
            if not ok:allclose=False
        result[group]={'passed':allclose,'max_absolute_delta':max_delta,'bitwise_equal_tensors':exact,'tensors':len(a[group]),'rtol':1e-5,'atol':1e-7}
        passed=passed and allclose
    result['RNG_equal']=snapshot_rng_equal(a['rng'],b['rng']);passed=passed and result['RNG_equal']
    result['AMP_and_scheduler_equal']=a['scaler']==b['scaler'] and a['scheduler']==b['scheduler'];passed=passed and result['AMP_and_scheduler_equal']
    # BatchNorm running statistics/counters are explicitly checked, not just output gradients.
    bnkeys=[k for k in a['student'] if any(s in k for s in ['running_mean','running_var','num_batches_tracked'])]
    result['BatchNorm_buffers_bitwise_equal']=all(torch.equal(a['student'][k],b['student'][k]) for k in bnkeys)
    passed=passed and result['BatchNorm_buffers_bitwise_equal'];result['status']='passed' if passed else 'failed'
    return result
def first_batch(ds,train,a):
    specs=next(iter(hist.SpeakerBalancedBatchSampler(train,1,16,2,a.seed)))
    return hist.collate_pairs([ds[x] for x in specs])
def supported_boundary(ds,records,train,a):
    extreme=load(RUN/'audit/CROP_EXTREMES.json')['extremes']
    wanted=[extreme['1.5']['recording']['speaker_id'],extreme['4.0']['recording']['speaker_id']]
    speakers=list(dict.fromkeys(wanted))
    speakers+= [s for s in train if s not in speakers][:16-len(speakers)]
    items=[ds[(s,hist.stable_int(f'HC0032:boundary:{s}:{i}'))] for s in speakers for i in range(2)]
    def replace(item,role,row,seconds,start):
        record=next(r for r in records[row['speaker_id']] if r.source_id==row['source_id'])
        waveform=hist.load_audio(record.path,a.sample_rate)
        class FixedOffset:
            def integers(self,low,high):
                assert low<=start<high;return start
        crop=hist.crop_by_voiced_duration(waveform,record.vad_intervals,seconds,FixedOffset(),a.sample_rate)
        item[role+'_audio']=torch.from_numpy(hist.peak_guard(crop));item[role+'_source_id']=record.source_id;item[role+'_seconds']=seconds
        if role=='student':item['augmentation']='clean'
        other='student' if role=='teacher' else 'teacher'
        if item[other+'_source_id']==record.source_id:
            choices=[r for r in records[row['speaker_id']] if r.source_id!=record.source_id and (other=='student' or r.voiced_seconds>=4.)]
            replacement=choices[0];duration=1.5 if other=='student' else 4.
            w=hist.load_audio(replacement.path,a.sample_rate)
            crop=hist.crop_by_voiced_duration(w,replacement.vad_intervals,duration,np.random.default_rng(20261107),a.sample_rate)
            item[other+'_audio']=torch.from_numpy(hist.peak_guard(crop));item[other+'_source_id']=replacement.source_id;item[other+'_seconds']=duration
            if other=='student':item['augmentation']='clean'
    for role,key in [('student','1.5'),('teacher','4.0')]:
        x=extreme[key];sp=x['recording']['speaker_id'];index=speakers.index(sp)*2+(1 if role=='teacher' else 0)
        replace(items[index],role,x['recording'],x['requested_voiced_seconds'],x['voiced_start'])
        assert len(items[index][role+'_audio'])==x['raw_samples']
    batch=hist.collate_pairs(items)
    assert len(set(x['speaker_id'] for x in items))==16 and len(items)==32
    assert all(x['teacher_source_id']!=x['student_source_id'] for x in items)
    return batch
def equivalence():
    cfg=load(BASE/'configs/B_seed2027.json');verify_inputs(cfg);a,records,train,ds,aug,assignments,used=make_data(cfg)
    batch=first_batch(ds,train,a);expected=load(BASE/'preparation/parity_checks.json')['checks'][0]['first_two_batch_sha256'][0]
    assert batch_hash(batch)==expected,'Prepared first batch changed'
    # Use a supported2s teacher reference so the original full32forward is measurable
    # in this slot. Preserve actual student inputs and different-source identities.
    for item in batch['metadata']:
        row=next(r for r in records[item['speaker_id']] if r.source_id==item['teacher_source_id'])
        lengths=np.asarray([end-start for start,end in row.vad_intervals],dtype=np.int64);cum=np.r_[0,np.cumsum(lengths)];req=32000
        starts=np.unique(np.clip(np.r_[0,int(cum[-1])-req,cum[1:-1]-1,cum[1:-1]-req+1],0,int(cum[-1])-req)).astype(np.int64)
        ends=starts+req;first=np.searchsorted(cum[1:],starts,side='right');last=np.searchsorted(cum[1:],ends-1,side='right')
        iv=np.asarray(row.vad_intervals);spans=iv[last,0]+ends-cum[last]-(iv[first,0]+starts-cum[first]);voiced_start=int(starts[int(np.argmin(spans))])
        class Offset:
            def integers(self,low,high):assert low<=voiced_start<high;return voiced_start
        wave=hist.load_audio(row.path,a.sample_rate)
        crop=hist.crop_by_voiced_duration(wave,row.vad_intervals,2.,Offset(),a.sample_rate)
        item['teacher_audio']=torch.from_numpy(hist.peak_guard(crop));item['teacher_seconds']=2.
    batch=hist.collate_pairs(batch['metadata'])
    save(OUT/'REFERENCE_BATCH.json',{'scope':'Semantic validation on supported2s teacher inputs; actual original student inputs and full16x2batch retained. Profile mode separately tests actual first5batches and global extrema.','batch_sha256':batch_hash(batch),'student_padded_seconds':batch['student_audio'].shape[1]/16000,'teacher_padded_seconds':batch['teacher_audio'].shape[1]/16000})
    for model in ['B','C','D']:
        out=OUT/'equivalence'/model;out.mkdir(parents=True);cfg=load(BASE/f'configs/{model}_seed2027.json');a=args_from(cfg)
        reference=build(cfg,a,train);metrics0,snapshot0=step(reference,batch,cfg,a,1,False,True)
        event(out,'REFERENCE_STEP',metrics=metrics0,batch_sha256=batch_hash(batch));cleanup(reference);del reference;gc.collect();torch.cuda.empty_cache()
        optimized=build(cfg,a,train);metrics1,snapshot1=step(optimized,batch,cfg,a,1,True,True)
        result=compare(snapshot0,snapshot1);result.update(loss_absolute_delta=abs(metrics0['loss']-metrics1['loss']),reference_metrics=metrics0,offload_metrics=metrics1,batch_sha256=batch_hash(batch),actual_batch_size=32,speakers_per_batch=16,
            saved_tensor_policy='LosslessCPU saved tensors; FULL32teacher/student/KL; originalfulltensorreductions; no teacher chunks or channel slicing; AMP temporaries and unusedfeaturemaps released; identitypadding bypassed')
        result['passed']=result['status']=='passed' and result['loss_absolute_delta']<=1e-6
        save(out/'result.json',result);event(out,'EQUIVALENCE',result=result)
        cleanup(optimized);del optimized,snapshot0,snapshot1;gc.collect();torch.cuda.empty_cache()
        assert result['passed'],f'{model} offload equivalence failed'
    save(OUT/'EQUIVALENCE_PASS.json',{'status':'passed','models':['B','C','D'],'seed':2027,'actual_batch_size':32,'gate':'loss, gradients, post-optimizer parameters, BatchNorm buffers, EMA, RNG and AMP/scheduler comparisons passed','tolerances':{'loss_abs':1e-6,'tensor_atol':1e-7,'tensor_rtol':1e-5,'BatchNorm_and_RNG':'bitwise equal'}})
def profile():
    assert load(OUT/'EQUIVALENCE_PASS.json')['status']=='passed'
    cfg=load(BASE/'configs/B_seed2027.json');a,records,train,ds,aug,assignments,used=make_data(cfg)
    boundary=supported_boundary(ds,records,train,a)
    save(OUT/'BOUNDARY_BATCH.json',{'status':'verified_real_waveforms','batch_sha256':batch_hash(boundary),'batch_size':32,'speakers':16,'student_padded_seconds':boundary['student_audio'].shape[1]/16000,'teacher_padded_seconds':boundary['teacher_audio'].shape[1]/16000,'scope':'Supported extreme stress test only, not a training-seed replicate; original different-source pairs and full32batch retained'})
    batches=list(hist.SpeakerBalancedBatchSampler(train,5,16,2,a.seed));expected_hashes=[]
    for model in ['B','C','D']:
        out=OUT/'profile'/model;out.mkdir(parents=True);cfg=load(BASE/f'configs/{model}_seed2027.json');a=args_from(cfg);state=build(cfg,a,train);metrics=[]
        for i,specs in enumerate(batches,1):
            batch=hist.collate_pairs([ds[x] for x in specs]);fingerprint=batch_hash(batch)
            if model=='B':expected_hashes.append(fingerprint)
            else:assert expected_hashes[i-1]==fingerprint
            event(out,'STEP_START',step=i,batch_sha256=fingerprint,student_seconds=batch['student_audio'].shape[1]/16000,teacher_seconds=batch['teacher_audio'].shape[1]/16000)
            row,_=step(state,batch,cfg,a,i,True);metrics.append(row);event(out,'STEP_COMPLETE',metrics=row)
        validations=[]
        for repetition in range(2):
            torch.manual_seed(hist.stable_int(f'HC0032:validation:{a.seed}'))
            torch.cuda.reset_peak_memory_stats();
            with memory_forward(state[0],None):v=hist.evaluate_verification(state[0],state[0],assignments,aug,a,torch.device('cuda'))
            validations.append(v)
            event(out,'VALIDATION_COMPLETE',repetition=repetition,metrics=v,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        assert validations[0]==validations[1],'Nondeterministic selector'
        save(out/'nominal_and_validation.json',{'status':'passed','successful_updates':5,'batch_hashes':expected_hashes,'steps':metrics,'validation':validations})
        state[0].train();state[1].train();event(out,'SUPPORTED_EXTREME_START',batch_sha256=batch_hash(boundary))
        stress,_=step(state,boundary,cfg,a,6,True);event(out,'SUPPORTED_EXTREME_COMPLETE',metrics=stress)
        save(out/'profile_pass.json',{'status':'passed','model':model,'successful_nominal_updates':5,'supported_extreme':stress,'deterministic_validation':True,'steps':metrics})
        cleanup(state);del state;gc.collect();torch.cuda.empty_cache()
    save(OUT/'PROFILE_PASS.json',{'status':'passed','models':['B','C','D'],'scope':'All unchanged full32nominal steps, both deterministic full validation passes and longest-supported-input forward/backward/update completed with CPU activation storage'})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['equivalence','profile']);args=p.parse_args();start=time.perf_counter()
    with (ROOT/'experiments/evaluation/watchdogs/.locks/arch_b_gpu_watchdog_20260921.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            OUT.mkdir(parents=True,exist_ok=True);gate=gpu_guard();event(OUT,'STARTED',mode=args.mode,command=[sys.executable,*sys.argv],resource_gate=gate)
            (equivalence if args.mode=='equivalence' else profile)()
            event(OUT,'COMPLETED',mode=args.mode,elapsed_seconds=time.perf_counter()-start)
        except BaseException as error:
            report={'status':'OOM' if isinstance(error,torch.cuda.OutOfMemoryError) else 'failed','mode':args.mode,'error':repr(error),'traceback':traceback.format_exc(),'elapsed_seconds':time.perf_counter()-start,
                'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'gpu':gpu()}
            save(OUT/f'failure_{args.mode}_{time.time_ns()}.json',report);event(OUT,'FAILED',**report);raise
