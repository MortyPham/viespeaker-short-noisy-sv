import sys,argparse,fcntl,traceback,hashlib
sys.path.insert(0,'/media/ml4u/Samsung_T52/ptnk_team/mortyhuaphuoc/artifacts/hc0032_revision/cohort_finish_20261107_211439/matched')
import qualify_capacity as q
import engine_recovery_v2 as e
from engine_recovery_v2 import *
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,default=2027);args=p.parse_args()
q.OUT=args.out;q.QUAL_SEED=args.seed;q.step=e.step;q.memory_forward=e.memory_forward
q.gpu_guard=e.gpu_guard
t=time.perf_counter()
with (ROOT/'experiments/evaluation/watchdogs/.locks/arch_b_gpu_watchdog_20260921.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        args.out.mkdir(parents=True,exist_ok=False);gate=gpu_guard()
        event(args.out,'STARTED',command=[sys.executable,*sys.argv],resource_gate=gate,source=record(Path(e.__file__)),allocator=os.getenv('PYTORCH_CUDA_ALLOC_CONF'))
        q.equivalence();q.profile()
        event(args.out,'COMPLETED',elapsed_seconds=time.perf_counter()-t)
    except BaseException as error:
        report={'status':'OOM' if isinstance(error,torch.cuda.OutOfMemoryError) else 'failed','error':repr(error),'traceback':traceback.format_exc(),'elapsed_seconds':time.perf_counter()-t,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'gpu':gpu()}
        save(args.out/f'failure_{time.time_ns()}.json',report);event(args.out,'FAILED',**report);raise
