from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl

from fbi_la.utils import contiguous


# @triton.autotune(
#     configs=[
#         triton.Config({"BL": 128, "BK": 128, "BV": 128}, num_warps=8),
#         triton.Config({"BL": 128, "BK": 64, "BV": 64}, num_warps=4),
#         triton.Config({"BL": 64, "BK": 64, "BV": 64}, num_warps=2),
#     ],
#     key=["L", "DK", "DV"],
# )
@triton.jit
def _fwd_kv_kernel(
    K, V, S, Z,
    stride_qk_bh, stride_qk_l, stride_qk_d,
    stride_vo_bh, stride_vo_l, stride_vo_d,
    stride_s_bh, stride_s_dk, stride_s_dv,
    stride_z_bh,
    scale,
    B: tl.constexpr,
    H: tl.constexpr,
    L: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BL: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    start_kv, start_m, off_bs_head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    
    NV = tl.cdiv(DV, BV)
    i_k = start_kv // (NV)
    i_v = start_kv % (NV)
    
    K_block_ptr = tl.make_block_ptr(
        base=K + off_bs_head * stride_qk_bh,
        shape=(DK, L),
        strides=(stride_qk_d, stride_qk_l),
        offsets=(i_k * BK, start_m * BL),
        block_shape=(BK, BL),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + off_bs_head * stride_vo_bh,
        shape=(L, DV),
        strides=(stride_vo_l, stride_vo_d),
        offsets=(start_m * BL, i_v * BV),
        block_shape=(BL, BV),
        order=(1, 0),
    )
    
    s = tl.zeros([BK, BV], dtype=tl.float32)
    z = tl.zeros([BK], dtype=tl.float32)

    k = tl.load(K_block_ptr, boundary_check=(0, 1))
    v = tl.load(V_block_ptr, boundary_check=(0, 1))
    
    v = (v * scale).to(v.dtype)
    s += tl.dot(k, v, allow_tf32=False)
    z += tl.sum(k, axis=1) / L

    S_block_ptr = tl.make_block_ptr(
        base=S + (off_bs_head + B * H * start_m) * stride_s_bh,
        shape=(DK, DV),
        strides=(stride_s_dk, stride_s_dv),
        offsets=(i_k * BK, i_v * BV),
        block_shape=(BK, BV),
        order=(1, 0),
    )
    tl.store(S_block_ptr, s.to(S.dtype.element_ty), boundary_check=(0, 1))
    
    Z_block_ptr = Z + (off_bs_head + B * H * start_m) * stride_z_bh + i_k * BK + tl.arange(0, BK)
    tl.store(Z_block_ptr, z.to(Z.dtype.element_ty), mask=((i_k * BK + tl.arange(0, BK)) < DK))
 

# @triton.autotune(
#     configs=[
#         triton.Config({"BL": 128, "BK": 128, "BV": 128}, num_warps=8),
#         triton.Config({"BL": 128, "BK": 64, "BV": 64}, num_warps=4),
#         triton.Config({"BL": 64, "BK": 64, "BV": 64}, num_warps=2),
#     ],
#     key=["L", "DK", "DV"],
# )
@triton.jit
def _fwd_qs_kernel(
    Q, S, O, Z,
    stride_qk_bh, stride_qk_l, stride_qk_d,
    stride_vo_bh, stride_vo_l, stride_vo_d,
    stride_s_bh, stride_s_dk, stride_s_dv,
    stride_z_bh,
    B: tl.constexpr,
    H: tl.constexpr,
    L: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BL: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    start_v, start_m, off_bs_head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    qkv_base_offset = off_bs_head * stride_qk_bh
    
    NV = tl.cdiv(DV, BV)
    
    Q_block_ptr = tl.make_block_ptr(
        base=Q + off_bs_head * stride_qk_bh,
        shape=(L, DK),
        strides=(stride_qk_l, stride_qk_d),
        offsets=(start_m * BL, 0),
        block_shape=(BL, BK),
        order=(1, 0),
    )
    S_block_ptr = tl.make_block_ptr(
        base=S + off_bs_head * stride_s_bh,
        shape=(DK, DV),
        strides=(stride_s_dk, 1),
        offsets=(0, start_v * BV),
        block_shape=(BK, BV),
        order=(1, 0),
    )
    Z_block_ptr = Z + off_bs_head * stride_z_bh + tl.arange(0, BK)
    
    o = tl.zeros([BL, BV], dtype=tl.float32)
    z_buffer = tl.zeros([BL], dtype=tl.float32)
    
    for i_k in range(0, DK, BK):
        q = tl.load(Q_block_ptr, boundary_check=(0, 1))
        s = tl.load(S_block_ptr, boundary_check=(0, 1))
        z = tl.load(Z_block_ptr, mask=((i_k + tl.arange(0, BK)) < DK))
    
        z_buffer += tl.sum(q * z[None, :], axis=1, keep_dims=False)
        o += tl.dot(q, s, allow_tf32=False)
        
        Q_block_ptr = tl.advance(Q_block_ptr, (0, BK))
        S_block_ptr = tl.advance(S_block_ptr, (BK, 0))
        Z_block_ptr = Z_block_ptr + tl.arange(0, BK)
        
    o = o / (z_buffer[:, None] + 1e-6)

    O_block_ptr = tl.make_block_ptr(
        base=O + off_bs_head * stride_vo_bh,
        shape=(L, DV),
        strides=(stride_vo_l, stride_vo_d),
        offsets=(start_m * BL, start_v * BV),
        block_shape=(BL, BV),
        order=(1, 0),
    )
    tl.store(O_block_ptr, o.to(O.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def _bwd_ds_kernel(
    O, Q, S, Z,
    DO, DQ, DS, DZ, 
    stride_qk_bh, stride_qk_l, stride_qk_d,
    stride_vo_bh, stride_vo_l, stride_vo_d,
    stride_s_bh, stride_s_dk, stride_s_dv,
    stride_z_bh,
    stride_dz_bh,
    B: tl.constexpr,
    H: tl.constexpr,
    L: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BL: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    start_kv, start_m, off_bs_head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    
    NV = tl.cdiv(DV, BV)
    i_k = start_kv // (NV)
    i_v = start_kv % (NV)
    
    O_block_ptr = tl.make_block_ptr(
        base=O + off_bs_head * stride_vo_bh,
        shape=(L, DV),
        strides=(stride_vo_l, stride_vo_d),
        offsets=(start_m * BL, i_v * BV),
        block_shape=(BL, BV),
        order=(1, 0),
    )
    Q_block_ptr = tl.make_block_ptr(
        base=Q + off_bs_head * stride_qk_bh,
        shape=(L, DK),
        strides=(stride_qk_l, stride_qk_d),
        offsets=(start_m * BL, i_k * BK),
        block_shape=(BL, BK),
        order=(1, 0),
    )
    DO_block_ptr = tl.make_block_ptr(
        base=DO + off_bs_head * stride_vo_bh,
        shape=(L, DV),
        strides=(stride_vo_l, stride_vo_d),
        offsets=(start_m * BL, i_v * BV),
        block_shape=(BL, BV),
        order=(1, 0),
    )
    S_block_ptr = tl.make_block_ptr(
        base=S + off_bs_head * stride_s_bh,
        shape=(DK, DV),
        strides=(stride_s_dk, stride_s_dv),
        offsets=(i_k * BK, i_v * BV),
        block_shape=(BK, BV),
        order=(1, 0),
    )
    Z_block_ptr = Z + off_bs_head * stride_z_bh + i_k * BK + tl.arange(0, BK)
    
    ds = tl.zeros([BK, BV], dtype=tl.float32)
    dz = tl.zeros([BL], dtype=tl.float32)
    dq = tl.zeros([BL, BK], dtype=tl.float32)
    
    do = tl.load(DO_block_ptr, boundary_check=(0, 1))
    o = tl.load(O_block_ptr, boundary_check=(0, 1))
    q = tl.load(Q_block_ptr, boundary_check=(0, 1))#, padding_option=0)
    s = tl.load(S_block_ptr, boundary_check=(0, 1))
    z = tl.load(Z_block_ptr, mask=((i_k * BK + tl.arange(0, BK)) < DK))
    
    z = tl.sum(q * z[None, :], axis=1, keep_dims=True).to(q.dtype) + 1e-6
    
    ds += tl.dot(tl.trans(q / z).to(do.dtype), do, allow_tf32=False)
    dz -= tl.sum(o * do / z, axis=1)
    dq += tl.dot(do, tl.trans(s), allow_tf32=False) / z
    
    DS_block_ptr = tl.make_block_ptr(
        base=DS + (off_bs_head + B * H * start_m) * stride_s_bh,
        shape=(BK, BV),
        strides=(stride_s_dk, stride_s_dv),
        offsets=(i_k * BK, i_v * BV),
        block_shape=(BK, BV),
        order=(1, 0),
    )
    tl.store(DS_block_ptr, ds.to(DS.dtype.element_ty), boundary_check=(0, 1))
    
    DQ_block_ptr = tl.make_block_ptr(
        base=DQ + off_bs_head * stride_vo_bh,
        shape=(L, BK),
        strides=(stride_vo_l, stride_vo_d),
        offsets=(start_m * BL, i_k * BK),
        block_shape=(BL, BK),
        order=(1, 0),
    )
    tl.store(DQ_block_ptr, dq.to(DQ.dtype.element_ty), boundary_check=(0, 1))
    
    DZ_block_ptr = DZ + off_bs_head * stride_dz_bh + start_m * BL + tl.arange(0, BL)
    tl.store(DZ_block_ptr, dz.to(DZ.dtype.element_ty), mask=((start_m * BL + tl.arange(0, BL)) < L))

    
"""
version 3.0
"""
@triton.jit
def _bwd_dkv_kernel(
    K, V,
    dK, dV, dS,
    stride_qk_bh, stride_qk_l, stride_qk_d,
    stride_s_bh, stride_s_dk, stride_s_dv,
    scale,
    B: tl.constexpr,
    H: tl.constexpr,
    L: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BL: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    start_kv, start_m, off_bs_head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    
    NV = tl.cdiv(DV, BV)
    i_k = start_kv // (NV)
    i_v = start_kv % (NV)
    
    qkv_base_offset = off_bs_head * stride_qk_bh

    K_block_ptr = tl.make_block_ptr(
        base=K + qkv_base_offset,
        shape=(L, DK),
        strides=(stride_qk_l, stride_qk_d),
        offsets=(start_m * BL, 0),
        block_shape=(BL, BK),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qkv_base_offset,
        shape=(L, DV),
        strides=(stride_qk_l, stride_qk_d),
        offsets=(start_m * BL, 0),
        block_shape=(BL, BV),
        order=(1, 0),
    )
    dS_block_ptr = tl.make_block_ptr(
        base=dS + off_bs_head * stride_s_bh,
        shape=(DK, DV),
        strides=(stride_s_dk, stride_s_dv),
        offsets=(0, 0),
        block_shape=(BK, BV),
        order=(1, 0),
    )

    dk = tl.zeros([BL, BK], dtype=tl.float32)
    dv = tl.zeros([BL, BV], dtype=tl.float32)
    
    
    ds = tl.load(dS_block_ptr, boundary_check=(0, 1))
    k = tl.load(K_block_ptr, boundary_check=(0, 1))
    v = tl.load(V_block_ptr, boundary_check=(0, 1))
    v = (v * scale).to(v.dtype)
    
    dk += tl.dot(v, tl.trans(ds).to(v.dtype), allow_tf32=False)
    dv += tl.dot(k, ds.to(k.dtype), allow_tf32=False) * scale
    
    dK_block_ptr = tl.make_block_ptr(
        base=dK + qkv_base_offset,
        shape=(L, DK),
        strides=(stride_qk_l, stride_qk_d),
        offsets=(start_m * BL, 0),
        block_shape=(BL, BK),
        order=(1, 0),
    )
    dV_block_ptr = tl.make_block_ptr(
        base=dV + qkv_base_offset,
        shape=(L, DV),
        strides=(stride_qk_l, stride_qk_d),
        offsets=(start_m * BL, 0),
        block_shape=(BL, BV),
        order=(1, 0),
    )
    tl.store(dK_block_ptr, dk.to(dK.dtype.element_ty), boundary_check=(0, 1))
    tl.store(dV_block_ptr, dv.to(dV.dtype.element_ty), boundary_check=(0, 1))
   
    
class LinearAttnFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, q, k, v, scale):
        B, H, L, K, V = *q.shape, v.shape[-1]
        
        BL= 128
        BK = min(128, triton.next_power_of_2(k.shape[-1]))
        BV = min(BK, triton.next_power_of_2(v.shape[-1]))
        BK, BV = max(BK, 16), max(BV, 16)
        
        NK = triton.cdiv(K, BK)
        NV = triton.cdiv(V, BV)
        NL = triton.cdiv(L, BL)

        num_warps = 4 if K <= 64 else 8
        num_stages = 2
        
        grid = (NK * NV, NL, B * H)
        s = torch.empty(NL, B, H, K, V, device=k.device)
        z = torch.empty(NL, B, H, K, device=k.device)
        _fwd_kv_kernel[grid](
            k, v, s, z,
            k.stride(1), k.stride(2), k.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            s.stride(2), s.stride(3), s.stride(4),
            z.stride(2),
            scale,
            B, H, L, K, V,
            BL=BL, BV=BV, BK=BK,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        s = s.sum(0).to(k.dtype)
        z = z.sum(0).to(k.dtype)
        
        grid = (NV, NL, B * H)
        o = torch.empty_like(v)
        _fwd_qs_kernel[grid](
            q, s, o, z,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            s.stride(1), s.stride(2), s.stride(3),
            z.stride(1),
            B, H, L, K, V,
            BL=BL, BV=BV, BK=BK,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        
        ctx.save_for_backward(q, k, v, o, s, z)
        ctx.scale = scale
        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        q, k, v, o, s, z = ctx.saved_tensors
        B, H, L, K, V = *q.shape, v.shape[-1]
        
        BL = 128
        BK = min(128, triton.next_power_of_2(k.shape[-1]))
        BV = min(128, triton.next_power_of_2(v.shape[-1]))
        BK, BV = max(BK, 16), max(BV, 16)
        
        NK = triton.cdiv(K, BK)
        NV = triton.cdiv(V, BV)
        NL = triton.cdiv(L, BL)
        
        assert NK == 1 and NV == 1
        
        num_warps = 4 if K <= 64 else 8
        num_stages = 2

        grid = (NK * NV, NL, B * H)
        
        do = do.contiguous()
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dz = torch.empty(B, H, L, device=k.device)
        ds = torch.empty(NL, B, H, K, V, device=k.device)

        _bwd_ds_kernel[grid](
            o, q, s, z,
            do, dq, ds, dz,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            ds.stride(2), ds.stride(3), ds.stride(4),
            z.stride(1),
            dz.stride(1),
            B, H, L, K, V,
            BL=BL, BK=BK, BV=BV,
            num_warps=num_warps,
            num_stages=num_stages
        )
        ds = ds.sum(0)#.to(k.dtype)
        
        grid = (NK * NV, NL, B * H)
        _bwd_dkv_kernel[grid](
            k, v,
            dk, dv, ds,
            k.stride(1), k.stride(2), k.stride(3),
            ds.stride(1), ds.stride(2), ds.stride(3),
            ctx.scale,
            B, H, L, K, V,
            BL=BL, BK=BK, BV=BV,
            num_warps=num_warps,
            num_stages=num_stages
        )
        
        dqk_k = dz.unsqueeze(-1) * k.mean(dim=-2, keepdim=True)
        dqk_q = (dz.unsqueeze(-1) * q).mean(dim=-2, keepdim=True)

        dq = dq + dqk_k
        dk = dk + dqk_q
        
        return dq, dk, dv, None, None, None
        # return dq, dk, dv, ds, dz, None, None, None, None, None


def linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[int] = None
) -> torch.Tensor:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `(B, H, L, K)`
        k (torch.Tensor):
            keys of shape `(B, H, L, K)`
        v (torch.Tensor):
            values of shape `(B, H, L, V)`
        scale (Optional[int]):
            Scale factor for the attention scores.
            If not provided, it will default to `1 / L`. Default: `None`.
    """
    if scale is None:
        scale = k.shape[-2] ** -1.0
    o = LinearAttnFunction.apply(q, k, v, scale)
    return o