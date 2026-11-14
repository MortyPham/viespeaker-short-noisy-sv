from pathlib import Path
import json, hashlib, shutil, ast, datetime, os
R=Path('/media/ml4u/Samsung_T52/ptnk_team/mortyhuaphuoc')
N=Path('/media/ml4u/Samsung_T52/ptnk_team/mortyhuaphuoc/artifacts/hc0032_revision/completion_20261107_154453')
O=R/'artifacts/hc0032_revision/overnight_20261106'
M=R/'artifacts/hc0032_revision/matched_seeds_20261107_0858'
P1=R/'artifacts/hc0032_revision/prompt1_20261106_2153'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2)
def load(p):return json.loads(Path(p).read_text())
def record(p):return {'path':str(p),'sha256':sha(p)}
for folder in ['queue','prompt1_replay','prompt2','prompt5','cuda_smoke','source_snapshot','matched','reports']:(N/folder).mkdir()
for p in (O/'source_snapshot').glob('*.py'):shutil.copy2(p,N/'source_snapshot'/p.name)
save(N/'queue/source_snapshot.json',[record(p) for p in sorted((N/'source_snapshot').glob('*.py'))])
shutil.copy2(O/'queue/frozen_inputs.json',N/'queue/frozen_inputs.json')
for src in [O/'queue/status.json',M/'STATUS.json',O/'manuscript_handoff/LATEST.json',P1/'final/model_registry.json',P1/'final/protocol_registry.json']:
    shutil.copy2(src,N/'audit'/(src.parent.name+'_'+src.name))
code=(O/'queue/gpu_stages.py').read_text()
code=code.replace("[2.,3.,4.]","[3.,2.,4.]")
code=code.replace("historical_initialization_hash':'UNVERIFIED; does not substitute another model, training remains blocked'","historical_initialization_hash':'UNVERIFIED; new controlled initialization is separately pinned'")
code=code.replace("f['cohort']='historical2026_role_comparison'", "f['cohort']='historical2026_role_comparison'\n    f['realization_manifest_sha256']=digest(RUN/'queue/frozen_inputs.json')\n    f['batch_policy']='complete original sorted eligible BS8 batches; archived student probes fixed'")
(N/'queue/gpu_stages.py').write_text(code)

# Upgrade the representative CPU diagnostic to a distinct CUDA smoke. No tolerance change.
code=(R/'artifacts/hc0032_revision/cpu_replay_20261107_0834/cpu_smoke.py').read_text()
code=code.replace('Explicitly authorized CPU parity/timing test; no queue files are changed.','Representative CUDA parity/timing smoke; complete original batches; no full inference.')
code=code.replace("os.environ['CUDA_VISIBLE_DEVICES']=''", "os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')")
code=code.replace('ROOT=Path.cwd()',f'ROOT=Path({str(R)!r})')
code=code.replace("QUEUE=ROOT/'artifacts/hc0032_revision/overnight_20261106/queue'","QUEUE=OUT.parent/'queue'")
code=code.replace('torch.manual_seed(2026)',"torch.manual_seed(2026)\nassert torch.cuda.is_available(), 'CUDA required; no CPU fallback'\nm.assert_inputs()\ntorch.cuda.reset_peak_memory_stats()\nembedding_checks=[]")
code=code.replace("'device':'cpu'","'device':'cuda'").replace("device='cpu'","device='cuda'")
code=code.replace("'CPU_explicitly_authorized':True,",'')
code=code.replace(".cpu().eval();event", ".to('cuda').eval();event")
code=code.replace("with torch.inference_mode():values=torch.nn.functional.normalize(net(padded,lengths),dim=-1).numpy()", "with torch.inference_mode():values=torch.nn.functional.normalize(net(padded.to('cuda'),lengths.to('cuda')),dim=-1).cpu().numpy()\n        torch.cuda.synchronize()")
code=code.replace("vectors.update(zip(batchkeys,values.copy(),strict=True))", "assert np.allclose(np.linalg.norm(values,axis=1),1,atol=1e-5)\n        old=m.hist.load_embedding_map(Path(m.MODELS[model]['embedding_cache'][condition]['current_path']))\n        delta=max(float(np.max(np.abs(v-old[k]))) for k,v in zip(batchkeys,values,strict=True))\n        embedding_checks.append({'model':model,'role':role,'condition':condition,'batchid':batchid,'max_component_delta':delta,'passed':delta<=1e-5})\n        vectors.update(zip(batchkeys,values.copy(),strict=True))")
code += "\nsave('embedding_checks.json',embedding_checks)\npassed=all(r['passed'] for r in results) and all(r['passed'] for r in embedding_checks)\nsave('cuda_resource_report.json',{'status':'passed' if passed else 'parity_failed','peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'backend':m.metadata('representative_cuda_smoke')})\nif passed:\n    m.save(OUT.parent/'prompt1_replay/smoke_pass.json',{'status':'passed','representative_smoke':str(OUT/'smoke_result.json'),'embedding_checks':str(OUT/'embedding_checks.json'),'tolerance':1e-5})\nelse: raise RuntimeError('CUDA parity failed; dependent historical inference stopped; all score diagnostics preserved')\n"
(N/'cuda_smoke/smoke.py').write_text(code)

# Pinned imports and new explicit calibration output. Prepared inputs remain read only.
code=(O/'prompt5/calibration.py').read_text()
code=code.replace('sys.path.insert(0, str(ROOT))',"sys.path.insert(0, str(HERE.parent/'source_snapshot'))")
code=code.replace('def run(args):',"def run(args):\n    assert load(HERE.parent/'prompt1_replay/replay_pass.json')['status']=='passed', 'Historical B parity required'")
code=code.replace("default=HERE/'prepared'",f"default=Path({str(O/'prompt5/prepared')!r})")
code=code.replace('External dependency-aware queue checks fresh replay and Prompt2 before invocation','Versioned resource wrapper and fresh primary replay gate; calibration independent of full role grid')
(N/'prompt5/calibration.py').write_text(code)

# Version prepared cohort under a new root, keeping pinned initialization read only.
B=N/'matched'
for directory in ['source','preparation','configs','runs','smoke','evaluation','logs']:(B/directory).mkdir()
for folder in ['source','preparation']:
    for p in (M/folder).iterdir():
        if p.is_file():shutil.copy2(p,B/folder/p.name)
for name in ['common.py','train_matched.py','evaluate_matched.py','execute_plan.py','report_results.py','launch_when_idle.sh']:
    code=(M/name).read_text().replace(str(M),str(B))
    if name=='common.py':code=code.replace('ROOT=Path(__file__).resolve().parents[3]',f'ROOT=Path({str(R)!r})')
    if name=='evaluate_matched.py':
        code=code.replace('def run(config_path):',"def run(config_path, enrollment_role='student'):")
        code=code.replace("out=BASE/f'evaluation/seed{cfg[\"training\"][\"seed\"]}/{cfg[\"paper_system\"]}'", "ss_out=BASE/f'evaluation/seed{cfg[\"training\"][\"seed\"]}/{cfg[\"paper_system\"]}'\n    out=ss_out if enrollment_role=='student' else ss_out.parent/(ss_out.name+'_TS')\n    assert enrollment_role=='student' or cfg['paper_system'] in ['C','D']")
        code=code.replace("q=ROOT/'artifacts/hc0032_revision/overnight_20261106/queue'","q=BASE.parent/'queue'")
        code=code.replace("model.load_state_dict(ck['student'],strict=True);del ck", "model.load_state_dict(ck['student'],strict=True)\n    enrollment_model=model\n    if enrollment_role=='teacher':\n        enrollment_model=hist.load_pretrained_encoder(args_from(cfg));enrollment_model.load_state_dict(ck['teacher'],strict=True)\n    del ck\n    enrollment_model=enrollment_model.to('cuda').eval()")
        code=code.replace('encode_cached(model,smoke_keys','encode_cached(enrollment_model,smoke_keys')
        code=code.replace("input_identity=sha(q/'frozen_inputs.json');cache=out/'embeddings'/selected['sha256']/cfg['test_protocol']['sha256']/input_identity/'student'", "input_identity=sha(q/'frozen_inputs.json');cache=out/'embeddings'/selected['sha256']/cfg['test_protocol']['sha256']/input_identity/enrollment_role\n    probe_cache=ss_out/'embeddings'/selected['sha256']/cfg['test_protocol']['sha256']/input_identity/'student'\n    if enrollment_role=='teacher':\n        ss_completion=load(ss_out/'completion.json')\n        assert ss_completion['selected_checkpoint']==selected\n        for info in ss_completion['outputs']:verify(info)")
        code=code.replace('encode_cached(model,keys,lambda key,d=duration:m.hist.make_enrollment','encode_cached(enrollment_model,keys,lambda key,d=duration:m.hist.make_enrollment')
        old="condition=m.hist.condition_id(duration,snr);folder=cache/condition/'bs8'"
        code=code.replace(old,"condition=m.hist.condition_id(duration,snr);folder=probe_cache/condition/'bs8'")
        old="            errors=m.hist.encode_cached(model,keys,lambda key,d=duration,s=snr:m.probe_wave('B',key,d,s),folder,\n                f'{cfg[\"paper_system\"]} SS seed{cfg[\"training\"][\"seed\"]} {condition}',8,torch.device('cuda'))\n            assert not errors,errors"
        new="            if enrollment_role=='student':\n                errors=m.hist.encode_cached(model,keys,lambda key,d=duration,s=snr:m.probe_wave('B',key,d,s),folder,\n                    f'{cfg[\"paper_system\"]} SS seed{cfg[\"training\"][\"seed\"]} {condition}',8,torch.device('cuda'))\n                assert not errors,errors\n            else:\n                assert folder.is_dir(), 'TS requires completed same-checkpoint SS probe cache'"
        assert old in code;code=code.replace(old,new)
        code=code.replace("f['enrollment_role']='student';f['probe_role']='student'", "f['enrollment_role']=enrollment_role;f['probe_role']='student'\n        f['realization_manifest_sha256']=input_identity;f['batch_policy']='complete original sorted eligible BS8'\n        f['checkpoint_sha256']=selected['sha256']\n        f['probe_cache_root']=str(probe_cache)\n        f['probe_role']='student'")
        code=code.replace("'roles':'student/student'","'roles':enrollment_role+'/student'")
        code=code.replace("a=p.parse_args()", "p.add_argument('--enrollment-role',choices=['student','teacher'],default='student');a=p.parse_args()")
        code=code.replace('run(a.config)','run(a.config,a.enrollment_role)')
    (B/name).write_text(code)
for p in (M/'configs').glob('*.json'):
    cfg=load(p)
    # Only copied preparation/input references and output paths are rebound.
    cfg['output_dir']=cfg['output_dir'].replace(str(M),str(B))
    for info in cfg['inputs']:
        if '/preparation/' in info['path']:info['path']=info['path'].replace(str(M),str(B))
    save(B/'configs'/p.name,cfg)
# These are explicitly identified NEW release manifests, regenerated after source edits.
for filename,entries in [('source_snapshot.json',[record(p) for p in sorted((B/'source').glob('*.py'))]),
                         ('release.json',[record(p) for p in sorted(B.glob('*.py'))]+[record(p) for p in sorted((B/'configs').glob('*.json'))]+[record(B/'launch_when_idle.sh')])]:
    path=B/'preparation'/filename
    # Copied historical records are preserved before publishing the new release.
    preserved=path.with_name('original_'+filename)
    with preserved.open('x') as f:f.write(path.read_text())
    # New copied operational release is the sole designated mutable preparation pointer.
    temp=path.with_suffix('.new');temp.write_text(json.dumps(entries,indent=2));os.replace(temp,path)
save(N/'queue/code_release.json',[record(p) for p in sorted((N/'queue').glob('*.py'))]+[record(N/'cuda_smoke/smoke.py'),record(N/'prompt5/calibration.py')])
for p in N.rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
save(N/'audit/versioning.json',{'status':'VERIFIED source/config relocation and static syntax only; CUDA behavior UNVERIFIED',
    'originals_read_only':[str(O),str(M),str(P1)],'new_output_root':str(N),
    'changes':['Representative CUDA smoke covers all models, both labels, three noise levels and complete original BS8 batches; checks scores and embedding components without relaxing tolerances',
       'Calibration depends on primary raw replay, independent of complete role grid; frozen prepared inputs unchanged',
       'Historical roles extract 3s enrollment first; all27 cells still required for final role table',
       'Cohort source/config/output references relocated; pinned old initialization read only',
       'New cohort evaluator supports teacher enrollment at same selected checkpoint and reuses completed SS student probe cache; CUDA validation pending'],
    'new_recipe_decision':'INFERRED:20%clean80%MUSAN controlled cohort answers repeated-run stability under its documented recipe. It cannot establish historical exact augmentation reproduction; no RIR recipe invented.',
    'training_review_limits':'CPU-prepared objectives and input parity evidence verified by hashes; actual finite CUDA updates, memory, EMA changes, deterministic validation and role evaluation require smoke before full launch.'})
print(N)
