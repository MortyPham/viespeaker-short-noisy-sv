"""Prepare matched2027/2028 inputs/configs; CPU metadata and waveform checks only."""
import ast
import csv
import gc
import json
import os
import re
import statistics
import sys
from datetime import datetime,timezone
os.environ['HF_HUB_OFFLINE']='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from common import *
torch.set_num_threads(2)
old=ROOT/'artifacts/hc0032_revision/prompt1_20261106_2153'
reg=load(old/'final/model_registry.json')
for record_ in load(BASE/'preparation/source_snapshot.json'):verify(record_)
for m in ['B','C','D']:verify(reg['models'][m]['checkpoint'])
bundle=old/'pretrained_bundle_smoke'
for info in reg['models']['A']['pretrained_bundle']['files']:
    assert sha(bundle/Path(info['path']).name)==info['sha256']
training={'sample_rate':16000,'steps':10000,'speakers_per_batch':16,'pairs_per_speaker':2,'num_workers':2,
    'long_min_seconds':2.,'long_max_seconds':4.,'short_min_seconds':1.,'short_max_seconds':1.5,
    'encoder_lr':1e-5,'head_lr':1e-3,'weight_decay':1e-4,'warmup_steps':500,'gradient_clip':5.,
    'aam_margin':.2,'aam_scale':30.,'clean_probability':.2,'noise_min_db':0.,'noise_max_db':20.,
    'validation_enroll_seconds':3.,'validation_probe_seconds':1.25,'validation_enrollments':5,
    'validation_calibration':10,'validation_tests':10,'validation_noise_db':10.,'target_far':.05,
    'validation_enrollment_model':'student','eval_every':1000,'pooling_variant':'original','no_amp':False,
    'ema_alpha':.999,'kl_epsilon':1e-5,'allow_incomplete_speakers':True}
prototype={'training':training,'pretrained_bundle':str(bundle),'output_dir':str(BASE/'runs/unassigned')}
a=args_from(prototype);selected,train,val=data(a)
assert train==reg['models']['B']['resolved_config']['train_speaker_ids']
assignments=hist.create_validation_assignments(selected,val,a)
rows=[]
for speaker,roles in assignments.items():
    for role,records in roles.items():
        for r in records:rows.append({'speaker':speaker,'role':role,'source_id':r.source_id,'path':r.path,'voiced_seconds':r.voiced_seconds})
import io,hashlib
buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
destination=BASE/'preparation/validation_assignments.csv'
if destination.exists():assert destination.read_bytes()==buffer.getvalue().encode()
else:
    with destination.open('x',newline='') as f:f.write(buffer.getvalue())
population={'train_speaker_ids':train,'validation_speaker_ids':val,
    'training_speakers':len(train),'validation_speakers':len(val),
    'train_validation_overlap':sorted(set(train)&set(val))}
if (BASE/'preparation/populations.json').exists():assert load(BASE/'preparation/populations.json')==population
else:save(BASE/'preparation/populations.json',population)
augmenter=hist.MusanAugmenter(a.musan_dir,a)
noise=load(old/'current_musan_training_validation_files.json')
for row in noise:verify(row)
save(BASE/'preparation/noise_inputs.json',noise)
inputs=[record(a.speaker_splits),record(a.recording_manifest),record(old/'final/model_registry.json'),
    record(BASE/'preparation/validation_assignments.csv'),record(BASE/'preparation/populations.json'),
    record(BASE/'preparation/noise_inputs.json'),reg['models']['B']['trial_manifest'],reg['models']['B']['score_file']]
save(BASE/'preparation/inputs.json',inputs)
# Hash the selected cached WAVs. No dataset is copied, cleaned or rewritten.
with (BASE/'preparation/audio_inputs.csv').open('x') as f:
    writer=csv.DictWriter(f,fieldnames=['speaker_id','source_id','path','sha256']);writer.writeheader()
    for i,(speaker,records) in enumerate(sorted(selected.items())):
        for r in records:writer.writerow({'speaker_id':speaker,'source_id':r.source_id,'path':r.path,'sha256':sha(r.path)})
        if i%500==0:print('hashed speakers',i,flush=True)
inputs.append(record(BASE/'preparation/audio_inputs.csv'))
configs=[];parity=[]
for seed in [2027,2028]:
    hist.seed_everything(seed)
    net=hist.load_pretrained_encoder(a).cpu().eval()
    before=tensor_hash(net.state_dict())
    # No shape-probing forward: all pretrained buffers remain unchanged.
    head_seed=hist.stable_int(f'HC0032:head:{seed}')
    torch.manual_seed(head_seed)
    head=hist.AAMSoftmaxHead(192,len(train),.2,30.)
    initial=BASE/f'initialization/seed{seed}.pt'
    with initial.open('xb') as f:torch.save({'student':net.state_dict(),'aam_head':head.state_dict(),
        'seed':seed,'head_seed':head_seed,'pretrained_bundle':str(bundle),'pretrained_embedding_sha256':sha(bundle/'embedding_model.ckpt')},f)
    init_info={**record(initial),'student_tensor_sha256':before,'head_tensor_sha256':tensor_hash(head.state_dict())}
    save(BASE/f'initialization/seed{seed}.json',init_info)
    reference=[]
    for model,lv,lk in [('B',0.,0.),('C',1.,0.),('D',1.,.01)]:
        cfg={'cohort':'controlled_2027_2028_documented_MUSAN_recipe','paper_system':model,
            'historical_run_hint':Path(reg['models'][model]['checkpoint']['path']).parent.name,
            'training':{**training,'seed':seed,'lambda_vector':lv,'lambda_kl':lk},
            'initialization':init_info,'pretrained_bundle':str(bundle),'inputs':inputs,
            'output_dir':str(BASE/f'runs/seed{seed}/{model}'),
            'initialization_policy':'Exact current allowlisted A encoder bytes; shared head within seed; no claim these are proven historical initialization bytes',
            'augmentation_policy':'Documented current historical-family 20% clean /80% MUSAN; uniform0–20dB; no RIR. New controlled recipe, not exact historical replay.',
            'RNG_policy':'Same pair dataset, including unused teacher-side draws for B; item-local NumPy seeds and step-indexed sampler independent of model/global RNG. Separate student-forward torch RNG per update.',
            'selection_rule':'Minimum10dB noisy-dev EER, student/student, every1000updates, earliest step on exact ties',
            'inference_roles':'student/student','test_protocol':reg['models']['B']['trial_manifest'],
            'expected_scored_population':reg['models']['B']['score_file'],
            'amp_policy':'FP16 autocast with common initial loss scale1024; abort on overflow rather than silently use a different update/input schedule',
            'smoke_steps':5,'checkpoint_every_updates':1000,
            'optimizer':{'name':'AdamW','betas':[.9,.999],'eps':1e-8},
            'method_settings':{'ema_alpha':None if model=='B' else .999,'lambda_vector':lv,'lambda_kl':lk,'kl_epsilon':1e-5 if model=='D' else None},
            'training_launch_gate':'Priority replay queue inactive, idle GPU, input/init fingerprint checks, actual16x2CUDA smoke passed'}
        path=BASE/f'configs/{model}_seed{seed}.json';save(path,cfg);configs.append(record(path))
        args=args_from(cfg);dataset=hist.SpeakerPairDataset(selected,{s:i for i,s in enumerate(train)},augmenter,args)
        sampler=hist.SpeakerBalancedBatchSampler(train,2,16,2,seed)
        current=[]
        for index,specification in enumerate(sampler):
            batch=hist.collate_pairs([dataset[item] for item in specification])
            assert len(batch['labels'])==32 and len({x['speaker_id'] for x in batch['metadata']})==16
            assert all(x['teacher_source_id']!=x['student_source_id'] for x in batch['metadata'])
            current.append(batch_hash(batch))
        if not reference:reference=current
        assert current==reference,(seed,model)
        parity.append({'seed':seed,'model':model,'actual_batch_size':32,'speakers_per_batch':16,'first_two_batch_sha256':current,
            'student_initial_sha256':before,'head_initial_sha256':init_info['head_tensor_sha256'],'matched':True})
        print('matched input/init',seed,model,flush=True)
    del net,head;gc.collect()
save(BASE/'preparation/parity_checks.json',{'status':'passed','device':'cpu data preparation only; no training','checks':parity})
# CPU-only objective checks on synthetic embeddings and common mathematical functions.
torch.manual_seed(7)
t=torch.nn.functional.normalize(torch.randn(32,192),dim=-1)
s=torch.nn.functional.normalize(torch.randn(32,192),dim=-1).requires_grad_()
v=hist.vector_alignment_loss(t,s);kl=hist.symmetric_diagonal_gaussian_kl(t,s,1e-5)
assert torch.isfinite(v) and torch.isfinite(kl)
checks=[]
for model,lv,lk in [('B',0.,0.),('C',1.,0.),('D',1.,.01)]:
    h=hist.AAMSoftmaxHead(192,3371,.2,30.);aam,_=h(s,torch.arange(32));loss=aam+lv*v+lk*kl
    grad=torch.autograd.grad(loss,s,retain_graph=True)[0];assert torch.isfinite(grad).all()
    checks.append({'model':model,'lambda_vector':lv,'lambda_kl':lk,'finite_loss':True,'finite_embedding_gradient':True})
save(BASE/'preparation/objective_checks.json',checks)
evidence={'historical_2026_qualifies':False,'complete_matched_seed_blocks':0,
    'selected_steps_are_not_disqualifying':True,'historical_inference_difference_alone_is_not_disqualifying':True,
    'findings':{'initial_parent_hash':{'status':'UNVERIFIED','evidence':str(ROOT/'artifacts/hc0032_revision/overnight_20261106/initialization_audit/parent_candidates.json')},
        'same_V1_split_manifest':{'status':'VERIFIED','evidence':str(old/'checkpoint_inspection.json')},
        'sample_order_and_augmentation_fingerprints':{'status':'UNVERIFIED','evidence':str(ROOT/'artifacts/hc0032_revision/overnight_20261106/initialization_audit/decision.json')},
        'selection_roles_differ':{'status':'VERIFIED','B':'student/student','C':'teacher/student','D':'teacher/student','evidence':str(old/'final/model_registry.json')},
        'launch_commands':{'status':'VERIFIED saved commands, incomplete code-at-run proof','evidence':str(old/'final/historical_commands.json')},
        'historical_configs':{'status':'VERIFIED recorded settings; missing values not inferred as byte proof','evidence':str(ROOT/'artifacts/hc0032_revision/overnight_20261106/initialization_audit/historical_config_summary.json')}},
    'plan':'Six fresh runs: B/C/D seed2027, then B/C/D seed2028. Keep historical2026 separate.',
    'new_recipe_decision':'Latest user authorizes a documented controlled recipe when historical matching cannot be verified. Pin A now and use existing20/80MUSAN implementation, no invented RIR data.',
    'configs':configs,'status':'CPU preparation complete; GPU smoke/full training not launched',
    'GPU_time_estimate_status':'UNVERIFIED until actual batch CUDA smoke'}
save(BASE/'preparation/historical_qualification_and_plan.json',evidence)
save(BASE/'preparation/completion.json',{'status':'prepared','utc':datetime.now(timezone.utc).isoformat(),
    'command':[sys.executable,*sys.argv],'seed_blocks':[2027,2028],'training_started':False,'GPU_work':False,
    'inputs':inputs,'configs':configs,'software':{'python':sys.version,'torch':torch.__version__},
    'code_revision':'No usable Git; source snapshots hashed','research_question':'Paired B/C/D objective comparison conditional on two new matched seeds'})
print('PREPARATION_COMPLETE',flush=True)
