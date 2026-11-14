"""CPU-only checkpoint metadata inspection; never performs model inference."""
import hashlib
import json
import sys
from pathlib import Path
import torch

torch.set_num_threads(2)
root = Path.cwd()
out = Path(__file__).resolve().parent
runs = dict(B='m0_aam_only_full_dataset', C='m2_legacy_vector_ema', D='m3_legacy_ema_gaussian_kl')
result = {'python': sys.version, 'torch': torch.__version__, 'device': 'cpu', 'models': {}}
for model, run in runs.items():
    path = root / 'experiments/mentor_stage' / run / 'best_validation_eer.pt'
    ck = torch.load(path, map_location='cpu', weights_only=False)
    states = {}
    for role in ('student', 'teacher'):
        if role not in ck:
            continue
        h = hashlib.sha256()
        tensors = ck[role]
        for name, tensor in sorted(tensors.items()):
            t = tensor.detach().cpu().contiguous()
            h.update(name.encode()); h.update(str(t.dtype).encode())
            h.update(str(tuple(t.shape)).encode()); h.update(t.numpy().tobytes())
        states[role] = {'sha256_name_dtype_shape_bytes': h.hexdigest(),
                        'tensor_count': len(tensors),
                        'parameters_and_buffers': sum(t.numel() for t in tensors.values()),
                        'keys_shapes': {k: list(v.shape) for k, v in tensors.items()}}
    result['models'][model] = {'path': str(path), 'keys': list(ck),
        'system': ck.get('system', ck.get('system_name')), 'step': ck.get('global_step'),
        'best_noise_eer': ck.get('best_noise_eer'), 'signature': ck.get('signature'),
        'run_id': ck.get('wandb_run_id'), 'states': states,
        'strict_model_load': 'UNVERIFIED: metadata/tensor inspection only; no inference',
        'pretrained_parent': 'UNVERIFIED: checkpoint signature does not pin parent tensor hash'}
    print(model, result['models'][model]['system'], ck.get('global_step'), flush=True)
    del ck
with (out / 'checkpoint_inspection.json').open('x') as f:
    json.dump(result, f, indent=2)
