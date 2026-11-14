"""Reconstruct validation assignments using existing loaders, without audio inference."""
import csv
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path.cwd(); OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import train_viespeaker_d_full as legacy
args=legacy.parse_args([])
args.allow_incomplete_speakers=True
records=legacy.load_recording_manifest(ROOT/'experiments/shared/d_full_audio_cache/recordings.csv',ROOT/'experiments/shared/d_full_audio_cache')
partitions={'train':[],'val':[],'test':[]}
with (ROOT/'experiments/shared/speaker_splits.csv').open() as f:
    for row in csv.DictReader(f):
        split='val' if row['split']=='validation' else row['split']
        partitions[split].append(row['speaker_id'])
selected,train,val=legacy.select_eligible_records(records,partitions,args)
assignments=legacy.create_validation_assignments(selected,val,args)
rows=[]
for speaker,roles in assignments.items():
    for role,rs in roles.items():
        for r in rs:
            rows.append({'speaker':speaker,'role':role,'source_id':r.source_id,'path':r.path,'voiced_seconds':r.voiced_seconds})
with (OUT/'reconstructed_validation_assignments.csv').open('x',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
augment=legacy.MusanAugmenter(ROOT/'datasets/musan',args)
noise=[]
for partition,buckets in [('train',augment.training),('validation',augment.validation)]:
    for category,paths in buckets.items():
        for path in paths:
            with open(path,'rb') as f: h=hashlib.file_digest(f,'sha256').hexdigest()
            noise.append({'partition':partition,'category':category,'path':path,'sha256':h})
with (OUT/'current_musan_training_validation_files.json').open('x') as f: json.dump(noise,f,indent=2)
with (OUT/'validation_audit.json').open('x') as f:
    json.dump({'status':'INFERRED historical assignments; VERIFIED reconstruction from current hashed source and V1 manifest',
        'command':[sys.executable,*sys.argv], 'train_speakers':len(train),'validation_speakers':len(val),
        'excluded_train':sorted(set(partitions['train'])-set(train)),
        'excluded_validation':sorted(set(partitions['val'])-set(val)),
        'assignment_counts':{r:sum(len(x[r]) for x in assignments.values()) for r in ['enrollment','calibration','test']},
        'recordings_all_exist':True,'validation_manifest_historical_original':'UNVERIFIED: not persisted; reconstruction only',
        'musan_train_files':sum(map(len,augment.training.values())),
        'musan_validation_files':sum(map(len,augment.validation.values())),
        'musan_train_validation_overlap':sorted(set(p for x in augment.training.values() for p in x)&set(p for x in augment.validation.values() for p in x)),
        'noise_waveform_realizations':'UNVERIFIED: historical per-development-trial recipes not located'},f,indent=2)
print('training',len(train),'validation',len(val),'assignments',len(rows),flush=True)
