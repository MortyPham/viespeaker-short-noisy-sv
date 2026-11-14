"""Original full32forward math; lossless CPU saved tensors and shorter lifetimes."""
import types
import engine as reference_engine
from engine import *
def gpu_guard():
    before=gpu();assert torch.cuda.is_available()
    budget=before['free_mib']-1024-512;assert budget>=512,'Insufficient bounded CUDA capacity'
    torch.cuda.set_per_process_memory_fraction(budget/(torch.cuda.get_device_properties(0).total_memory/2**20),0)
    x=torch.empty(1,device='cuda');del x;torch.cuda.empty_cache()
    return {'before':before,'after_context':gpu(),'allocator_budget_mib':budget,'physical_headroom_mib':1024,'nonallocator_allowance_mib':512,
        'policy':'Explicitv3oneGiBphysicalheadroom plus512MiBnonallocator; one existing vLLMowner unchanged; no utilization gate'}
def low_memory_ecapa(self,x,lengths=None):
    from speechbrain.lobes.models.ECAPA_TDNN import TDNNBlock
    x=x.transpose(1,2);xl=[]
    for layer in self.blocks:
        x=layer(x) if isinstance(layer,TDNNBlock) else layer(x,lengths=lengths)
        xl.append(x)
    x=torch.cat(xl[1:],dim=1);x=self.mfa(x)
    del xl  # Original retains four unused feature maps through attention pooling.
    x=self.asp(x,lengths=lengths);x=self.asp_bn(x);x=self.fc(x)
    return x.transpose(1,2)
def low_memory_pool(self,x,lengths=None):
    from speechbrain.dataio.dataio import length_to_mask
    L=x.shape[-1]
    def stats(value,weight):
        means=[];stds=[]
        # Statistics are independent per channel; time reductions and dtypes stay unchanged.
        for start in range(0,value.shape[1],512):
            v=value[:,start:start+512];w=weight if weight.shape[1]==1 else weight[:,start:start+512]
            mean=(w*v).sum(2);std=torch.sqrt((w*(v-mean.unsqueeze(2)).pow(2)).sum(2).clamp(self.eps))
            means.append(mean);stds.append(std)
        return torch.cat(means,1),torch.cat(stds,1)
    if lengths is None:lengths=torch.ones(x.shape[0],device=x.device)
    mask=length_to_mask(lengths*L,max_len=L,device=x.device).unsqueeze(1)
    if self.global_context:
        total=mask.sum(dim=2,keepdim=True).float();mean,std=stats(x,mask/total)
        if torch.is_autocast_enabled('cuda'):
            dtype=torch.get_autocast_dtype('cuda')
            mean=mean.unsqueeze(2).repeat(1,1,L).to(dtype);std=std.unsqueeze(2).repeat(1,1,L).to(dtype)
            attn=torch.cat([x.to(dtype),mean,std],dim=1)
        else:attn=torch.cat([x,mean.unsqueeze(2).repeat(1,1,L),std.unsqueeze(2).repeat(1,1,L)],1)
        del mean,std
    else:attn=x
    attn=self.conv(self.tanh(self.tdnn(attn)))
    attn=attn.masked_fill(mask==0,float('-inf'));attn=torch.nn.functional.softmax(attn,dim=2)
    mean,std=stats(x,attn)
    return torch.cat((mean,std),1).unsqueeze(2)
@contextlib.contextmanager
def memory_forward(student,teacher):
    patched=[]
    for encoder in [student,teacher]:
        if encoder is None:continue
        net=encoder.embedding_model
        for module,method in [(net,low_memory_ecapa),(net.asp,low_memory_pool)]:
            patched.append((module,module.forward));module.forward=types.MethodType(method,module)
    try:yield
    finally:
        for module,old in reversed(patched):module.forward=old
def step(state,batch,cfg,a,index,offload=True,capture=False):
    if not offload:return reference_engine.step(state,batch,cfg,a,index,False,capture)
    with memory_forward(state[0],state[2]):return reference_engine.step(state,batch,cfg,a,index,True,capture)
