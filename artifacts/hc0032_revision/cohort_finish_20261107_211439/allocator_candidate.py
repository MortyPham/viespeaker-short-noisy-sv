"""One targeted expandable-segment allocator investigation; unchanged recipe/cap."""
import json,os,runpy,sys,time,traceback
from pathlib import Path
N=Path(__file__).resolve().parent
OUT=N/'allocator_expandable_seed2027'
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
sys.path.insert(0,str(N/'matched'))
sys.argv=[str(N/'matched/qualify_capacity.py'),'all','--seed','2027','--out',str(OUT)]
started=time.perf_counter()
try:runpy.run_path(str(N/'matched/qualify_capacity.py'),run_name='__main__')
finally:
    import torch
    OUT.mkdir(exist_ok=True)
    with (OUT/'allocator_summary.txt').open('x') as f:f.write(torch.cuda.memory_summary())
    with (OUT/'allocator_environment.json').open('x') as f:json.dump({'configuration':os.environ['PYTORCH_CUDA_ALLOC_CONF'],'backend':torch.cuda.get_allocator_backend(),
        'elapsed_seconds':time.perf_counter()-started,'question':'Can nativeallocatorfragmentation be reduced without any scientificrecipe/batch/precision/cap change?',
        'full_training_started':False,'scope':'Targeted investigation only; allsemantic/profilegates stillrequired'},f,indent=2)
