"""Small explicit CPU smoke test; no GPU or full evaluation is attempted."""
import copy
import importlib.metadata
import json
import os
import shutil
import sys
from pathlib import Path
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import torch
from evaluate_aam_only_full_viespeaker_h_1e1p import load_pretrained_encoder
torch.set_num_threads(2)
torch.manual_seed(2026)
source = Path('/home/ml4u/.cache/viespeaker_models/speechbrain_ecapa_voxceleb')
local = OUT/'pretrained_bundle_smoke'
local.mkdir(exist_ok=False)
for p in source.iterdir():
    if p.is_file(): shutil.copyfile(p,local/p.name)
base = load_pretrained_encoder(local).cpu().eval()
wave = torch.randn(1,16000) * .01
lengths = torch.ones(1)
models = json.loads((OUT/'model_registry.json').read_text())['models']
result = {'device':'cpu', 'purpose':'strict tensor loading and one synthetic 1s waveform per state; not benchmark replay',
    'seed':2026,'command':[sys.executable,*sys.argv], 'software':{}, 'models':{}}
for p in ['torch','torchaudio','speechbrain','numpy','pandas','scikit-learn']:
    result['software'][p]=importlib.metadata.version(p)
with torch.inference_mode():
    for name, cfg in models.items():
        ck = None if name=='A' else torch.load(cfg['checkpoint']['path'],map_location='cpu',weights_only=False)
        roles = ['pretrained'] if name=='A' else ['student'] if name=='B' else ['student','teacher']
        report = {}
        for role in roles:
            model = copy.deepcopy(base)
            if ck is not None:
                loaded = model.load_state_dict(ck[role],strict=True)
                assert not loaded.missing_keys and not loaded.unexpected_keys
                # Verify actual tensor values, not just key/shape compatibility.
                assert all(torch.equal(v,ck[role][k]) for k,v in model.state_dict().items())
            model.eval()
            assert all(not m.training for m in model.modules())
            vector = model(wave,lengths)
            assert tuple(vector.shape)==(1,192) and torch.isfinite(vector).all()
            report[role]={'strict_load':True,'tensor_values_equal_checkpoint':ck is not None,
                'all_modules_eval':True,'output_shape':list(vector.shape),'output_norm':float(vector.norm()),
                'finite':True,'step':None if ck is None else ck['global_step']}
            del model
        result['models'][name]=report
        print(name,report,flush=True)
        del ck
with (OUT/'strict_load_smoke.json').open('x') as f: json.dump(result,f,indent=2)
