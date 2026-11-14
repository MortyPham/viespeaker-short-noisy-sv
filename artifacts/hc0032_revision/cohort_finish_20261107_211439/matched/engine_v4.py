"""Full32teacher/student, original reductions; only lossless storage/lifetime changes."""
import types
import engine as reference_engine
from engine import *
from engine_v2 import amp_context_forward
from engine_v3 import gpu_guard,low_memory_ecapa
@contextlib.contextmanager
def memory_forward(student,teacher):
    patches=[]
    for encoder in [student,teacher]:
        if encoder is None:continue
        net=encoder.embedding_model
        for module,method in [(net,low_memory_ecapa),(net.asp,amp_context_forward)]:
            patches.append((module,module.forward));module.forward=types.MethodType(method,module)
    try:yield
    finally:
        for module,old in reversed(patches):module.forward=old
def step(state,batch,cfg,a,index,offload=True,capture=False):
    if not offload:return reference_engine.step(state,batch,cfg,a,index,False,capture)
    with memory_forward(state[0],state[2]):return reference_engine.step(state,batch,cfg,a,index,True,capture)
