import torch
import torch.nn as nn
import torch.nn.functional as F

from fbi_la.ops.linear_attn.attention import linear_attention


def naive_attn_mean(
    q,
    k,
    v,
    scale = None
):
    if scale is None:
        scale = k.shape[-2] ** -1.0
        
    z = q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1)
    s = k.transpose(-2, -1) @ (v * scale)
    o = q @ s / (z + 1e-6)
    
    return o


def naive_attn_sum(
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
    b = torch.allclose(A, B, rtol=0, atol=1e-3)
    return b


if __name__ == "__main__":
    B, H, L, D = 2, 4, 256, 32
    dtype = torch.float32

    q = torch.randn((B, H, L, D), dtype=dtype, device="cuda", requires_grad=True)
    k = torch.randn((B, H, L, D), dtype=dtype, device="cuda", requires_grad=True)
    v = torch.randn((B, H, L, D), dtype=dtype, device="cuda", requires_grad=True)
    
    q = F.elu(q) + 1.0
    k = F.elu(k) + 1.0
    
    do = torch.randn_like(v).cuda()
    
    assert check_close(naive_attn_mean(q, k, v), naive_attn_sum(q, k, v, 1.0))
    print("Pass")
    
    # naive sum
    ref = naive_attn_sum(q, k, v)
    
    q.retain_grad(), k.retain_grad(), v.retain_grad()
    ref.backward(do, retain_graph=True)
    
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    
    # triton sum
    tri = linear_attention(q, k, v)
    
    q.retain_grad(), k.retain_grad(), v.retain_grad()
    tri.backward(do, retain_graph=True)
    
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    
    assert check_close(ref, tri)
    assert check_close(ref_dq, tri_dq)
    assert check_close(ref_dk, tri_dk)
    assert check_close(ref_dv, tri_dv)
    print("Pass")
    
    # naive mean
    ref = naive_attn_mean(q, k, v)
    
    q.retain_grad(), k.retain_grad(), v.retain_grad()
    ref.backward(do, retain_graph=True)
    
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    
    # triton mean
    tri = linear_attention(q, k, v, 1.0)
    
    q.retain_grad(), k.retain_grad(), v.retain_grad()
    tri.backward(do, retain_graph=True)
    
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    
    assert check_close(ref, tri)
    assert check_close(ref_dq, tri_dq)
    assert check_close(ref_dk, tri_dk)
    assert check_close(ref_dv, tri_dv)
    print("Pass")