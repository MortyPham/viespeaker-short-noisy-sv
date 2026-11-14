"""Lossless activation storage plus validated eval-teacher/AMP temporary reductions.

Student forward and batch-Gaussian KL always operate on all32 examples.
"""
import types
import engine as reference_engine
from engine import *
def amp_context_forward(self,x,lengths=None):
    # Same original attentive-pooling formula and state layout. Cast individual
    # concatenands where the immediately following AMP convolution casts them.
    # FP32 statistics and FP32 repeat-backward reductions remain unchanged.
    from speechbrain.dataio.dataio import length_to_mask
    L=x.shape[-1]
    def stats(value,weight):
        mean=(weight*value).sum(2)
        std=torch.sqrt((weight*(value-mean.unsqueeze(2)).pow(2)).sum(2).clamp(self.eps))
        return mean,std
    if lengths is None:lengths=torch.ones(x.shape[0],device=x.device)
    mask=length_to_mask(lengths*L,max_len=L,device=x.device).unsqueeze(1)
    if self.global_context:
        total=mask.sum(dim=2,keepdim=True).float();mean,std=stats(x,mask/total)
        dtype=torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else x.dtype
        if torch.is_autocast_enabled('cuda'):
            mean=mean.unsqueeze(2).repeat(1,1,L).to(dtype)
            std=std.unsqueeze(2).repeat(1,1,L).to(dtype)
            attn=torch.cat([x.to(dtype),mean,std],dim=1)
        else:
            attn=torch.cat([x,mean.unsqueeze(2).repeat(1,1,L),std.unsqueeze(2).repeat(1,1,L)],dim=1)
    else:attn=x
    attn=self.conv(self.tanh(self.tdnn(attn)))
    attn=attn.masked_fill(mask==0,float('-inf'));attn=torch.nn.functional.softmax(attn,dim=2)
    mean,std=stats(x,attn)
    return torch.cat((mean,std),dim=1).unsqueeze(2)
@contextlib.contextmanager
def memory_forward(student,teacher):
    patches=[]
    for encoder in [student,teacher]:
        if encoder is None:continue
        asp=encoder.embedding_model.asp
        old=asp.forward;patches.append((asp,old));asp.forward=types.MethodType(amp_context_forward,asp)
    if teacher is not None:
        assert all(not m.training for m in teacher.modules()),'Only eval teacher may be chunked'
        old=teacher.forward;patches.append((teacher,old))
        def teacher_chunks(self,waveform,lengths):
            # Slice the already-padded FULL batch; never recollate or renormalize lengths.
            return torch.cat([old(waveform[i:i+8],lengths[i:i+8]) for i in range(0,len(waveform),8)],dim=0)
        teacher.forward=types.MethodType(teacher_chunks,teacher)
    try:yield
    finally:
        for module,old_forward in reversed(patches):module.forward=old_forward
def step(state,batch,cfg,a,index,offload=True,capture=False):
    if not offload:return reference_engine.step(state,batch,cfg,a,index,False,capture)
    with memory_forward(state[0],state[2]):
        return reference_engine.step(state,batch,cfg,a,index,True,capture)
