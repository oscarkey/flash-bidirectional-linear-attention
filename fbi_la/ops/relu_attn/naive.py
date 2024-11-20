import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/public/liguoqi/ssl/wds/flash-non-causal-linear-attention')))
from fla_nc.ops.relu_attn.relu_attn import relu_attention


def naive_relu_attention(
    q,
    k,
    v,
    scale = None
):
    if scale is None:
        scale = k.shape[-2] ** -1.0
        
    z = q @ k.sum(dim=-2, keepdim=True).transpose(-2, -1)
    s = k.transpose(-2, -1) @ (v * scale)
    o = q @ s / (z + 1e-6)
    
    return o


def check_close(A, B):
    b = torch.allclose(A, B, rtol=1e-2, atol=1e-2)
    # print(b, (A - B).abs().max().item())
    return b


if __name__ == "__main__":
    B, H, L, D = 2, 4, 256, 32
    dtype = torch.float32

    q = torch.randn((B, H, L, D), dtype=dtype, device="cuda", requires_grad=True)
    k = torch.randn((B, H, L, D), dtype=dtype, device="cuda", requires_grad=True)
    v = torch.randn((B, H, L, D), dtype=dtype, device="cuda", requires_grad=True)
    
    q = F.relu(q)
    k = F.relu(k)
    
    do1 = torch.rand_like(v).cuda()
    do2 = torch.rand_like(v).cuda()
    
    # naive
    ref = naive_relu_attention(q, k, v)
    q.retain_grad(), k.retain_grad(), v.retain_grad()
    ref.backward(do1, retain_graph=True)
    
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    
    # triton
    tri = relu_attention(q, k, v)
    q.retain_grad(), k.retain_grad(), v.retain_grad()
    tri.backward(do2, retain_graph=True)
    
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    
    assert check_close(ref, tri)
    assert check_close(ref_dq, tri_dq)
    assert check_close(ref_dk, tri_dk)
    assert check_close(ref_dv, tri_dv)
    
    print("Pass")