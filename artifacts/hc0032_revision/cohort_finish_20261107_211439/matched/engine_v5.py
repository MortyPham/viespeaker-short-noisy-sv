"""Final full-batch execution: skip identity padding and release unused tensors."""
import types
import engine as reference_engine
from engine import *
from engine_v3 import gpu_guard,low_memory_ecapa
def pooled_forward(self,x,lengths=None):
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
        del mean,std
    else:attn=x
    attn=self.conv(self.tanh(self.tdnn(attn)))
    attn=attn.masked_fill(mask==0,float('-inf'));attn=torch.nn.functional.softmax(attn,dim=2)
    mean,std=stats(x,attn)
    return torch.cat((mean,std),dim=1).unsqueeze(2)
@contextlib.contextmanager
def memory_forward(student,teacher):
    from speechbrain.nnet.CNN import Conv1d,get_padding_elem
    patches=[]
    def skip_identity(original):
        def impl(self,x,kernel_size,dilation,stride):
            padding=get_padding_elem(self.in_channels,stride,kernel_size,dilation)
            return x if not any(padding) else original(x,kernel_size,dilation,stride)
        return impl
    for encoder in [student,teacher]:
        if encoder is None:continue
        net=encoder.embedding_model
        for module,method in [(net,low_memory_ecapa),(net.asp,pooled_forward)]:
            patches.append((module,'forward',module.forward));module.forward=types.MethodType(method,module)
        for module in net.modules():
            if isinstance(module,Conv1d):
                original=module._manage_padding;patches.append((module,'_manage_padding',original))
                module._manage_padding=types.MethodType(skip_identity(original),module)
    try:yield
    finally:
        for module,name,original in reversed(patches):setattr(module,name,original)
def step(state,batch,cfg,a,index,offload=True,capture=False):
    if not offload:return reference_engine.step(state,batch,cfg,a,index,False,capture)
    with memory_forward(state[0],state[2]):return reference_engine.step(state,batch,cfg,a,index,True,capture)
