from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl

from fbi_la.utils import contiguous


@triton.jit
def _fwd_kv_(
    K, V, S, Z,
    stride_qk_head, stride_qk_seqlen, stride_qk_dim,
    stride_s_bh, stride_s_dimk, stride_s_dimv,
    stride_z_bh,
    scale,
    B: tl.constexpr,
    H: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    start_m, off_bs_head = tl.program_id(0), tl.program_id(1)
    qkv_base_offset = off_bs_head * stride_qk_head
    
    K_block_ptr = tl.make_block_ptr(
        base=K + qkv_base_offset,
        shape=(D, L),
        strides=(stride_qk_dim, stride_qk_seqlen),
        offsets=(0, start_m * BLOCK_M),
        block_shape=(D, BLOCK_M),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qkv_base_offset,
        shape=(L, D),
        strides=(stride_qk_seqlen, stride_qk_dim),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, D),
        order=(1, 0),
    )
    
    s = tl.zeros([D, D], dtype=tl.float32)
    z = tl.zeros([D], dtype=tl.float32)

    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)
    
    v = (v * scale).to(v.dtype)
    s += tl.dot(k, v, allow_tf32=False)
    z += tl.sum(k, axis=1)

    S_block_ptr = tl.make_block_ptr(
        base=S + (off_bs_head + B * H * start_m) * stride_s_bh,
        shape=(D, D),
        strides=(stride_s_dimk, stride_s_dimv),
        offsets=(0, 0),
        block_shape=(D, D),
        order=(1, 0),
    )
    tl.store(S_block_ptr, s.to(S.dtype.element_ty))
    
    Z_block_ptr = Z + (off_bs_head + B * H * start_m) * stride_z_bh + tl.arange(0, D)
    tl.store(Z_block_ptr, z.to(Z.dtype.element_ty))
    
    
@triton.jit
def _fwd_qs_(
    Q, S, O, Z,
    stride_qk_head, stride_qk_seqlen, stride_qk_dim,
    stride_s_bh, stride_s_dimk, stride_s_dimv,
    stride_z_bh,
    B: tl.constexpr,
    H: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    start_m, off_bs_head = tl.program_id(0), tl.program_id(1)
    qkv_base_offset = off_bs_head * stride_qk_head
    
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qkv_base_offset,
        shape=(L, D),
        strides=(stride_qk_seqlen, stride_qk_dim),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, D),
        order=(1, 0),
    )
    S_block_ptr = tl.make_block_ptr(
        base=S + off_bs_head * stride_s_bh,
        shape=(D, D),
        strides=(stride_s_dimk, 1),
        offsets=(0, 0),
        block_shape=(D, D),
        order=(1, 0),
    )
    Z_block_ptr = Z + off_bs_head * stride_z_bh + tl.arange(0, D)
    
    o = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    q = tl.load(Q_block_ptr)
    s = tl.load(S_block_ptr)
    z = tl.load(Z_block_ptr)
    
    z = tl.sum(q * z[None, :], axis=1, keep_dims=True)
    
    o += tl.dot(q, s, allow_tf32=False)
    o = o / (z + 1e-6)

    O_block_ptr = tl.make_block_ptr(
        base=O + qkv_base_offset,
        shape=(L, D),
        strides=(stride_qk_seqlen, stride_qk_dim),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, D),
        order=(1, 0),
    )
    tl.store(O_block_ptr, o.to(O.dtype.element_ty))
    

@triton.jit
def _bwd_ds_kernel(
    O, Q, S, Z,
    DO, DQ, DS, DZ, 
    stride_qk_head, stride_qk_seqlen, stride_qk_dim,
    stride_s_bh, stride_s_dimk, stride_s_dimv,
    stride_z_bh, stride_dz_bh,
    B: tl.constexpr,
    H: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    start_m, off_bs_head = tl.program_id(0), tl.program_id(1)
    qkv_base_offset = off_bs_head * stride_qk_head
    
    O_block_ptr = tl.make_block_ptr(
        base=O + qkv_base_offset,
        shape=(L, D),
        strides=(stride_qk_seqlen, stride_qk_dim),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, D),
        order=(1, 0),
    )
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qkv_base_offset,
        shape=(L, D),
        strides=(stride_qk_seqlen, stride_qk_dim),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, D),
        order=(1, 0),
    )
    DO_block_ptr = tl.make_block_ptr(
        base=DO + qkv_base_offset,
        shape=(L, D),
        strides=(stride_qk_seqlen, stride_qk_dim),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, D),
        order=(1, 0),
    )
    S_block_ptr = tl.make_block_ptr(
        base=S + off_bs_head * stride_s_bh,
        shape=(D, D),
        strides=(stride_s_dimk, 1),
        offsets=(0, 0),
        block_shape=(D, D),
        order=(1, 0),
    )
    Z_block_ptr = Z + off_bs_head * stride_z_bh + tl.arange(0, D)
    
    ds = tl.zeros([D, D], dtype=tl.float32)
    dz = tl.zeros([BLOCK_M], dtype=tl.float32)
    dq = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    
    do = tl.load(DO_block_ptr)
    o = tl.load(O_block_ptr)
    q = tl.load(Q_block_ptr)
    s = tl.load(S_block_ptr)
    z = tl.load(Z_block_ptr)
    
    z = tl.sum(q * z[None, :], axis=1, keep_dims=True).to(q.dtype)
    
    ds += tl.dot(tl.trans(q / z).to(do.dtype), do, allow_tf32=False)
    dz -= tl.sum(o * do / z, axis=1)
    dq += tl.dot(do, tl.trans(s)) / z
    
    # tl.static_print(dz)
    
    DS_block_ptr = tl.make_block_ptr(
        base=DS + (off_bs_head + B * H * start_m) * stride_s_bh,
        shape=(D, D),
        strides=(stride_s_dimk, stride_s_dimv),
        offsets=(0, 0),
        block_shape=(D, D),
        order=(1, 0),
    )
    tl.store(DS_block_ptr, ds.to(DS.dtype.element_ty))
    
    DQ_block_ptr = tl.make_block_ptr(
        base=DQ + qkv_base_offset,
        shape=(L, D),
        strides=(stride_qk_seqlen, stride_qk_dim),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, D),
        order=(1, 0),
    )
    tl.store(DQ_block_ptr, dq.to(DQ.dtype.element_ty))
    
    DZ_block_ptr = DZ + off_bs_head * stride_dz_bh + start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(DZ_block_ptr, dz.to(DZ.dtype.element_ty))
    
    
@triton.jit
def _bwd_dkv_kernel(
    K, V,
    DK, DV, DS,
    stride_qk_head, stride_qk_seqlen, stride_qk_dim,
    stride_s_bh, stride_s_dimk, stride_s_dimv,
    scale,
    B: tl.constexpr,
    H: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    start_m, off_bs_head = tl.program_id(0), tl.program_id(1)
    qkv_base_offset = off_bs_head * stride_qk_head

    K_block_ptr = tl.make_block_ptr(
        base=K + qkv_base_offset,
        shape=(L, D),
        strides=(stride_qk_seqlen, stride_qk_dim),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qkv_base_offset,
        shape=(L, D),
        strides=(stride_qk_seqlen, stride_qk_dim),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, D),
        order=(1, 0),
    )
    DS_block_ptr = tl.make_block_ptr(
        base=DS + off_bs_head * stride_s_bh,
        shape=(D, D),
        strides=(stride_s_dimk, stride_s_dimv),
        offsets=(0, 0),
        block_shape=(D, D),
        order=(1, 0),
    )

    dk = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    
    
    ds = tl.load(DS_block_ptr)
    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)
    v = (v * scale).to(v.dtype)
    
    dk += tl.dot(v, tl.trans(ds).to(v.dtype), allow_tf32=False)
    dv += tl.dot(k, ds.to(k.dtype), allow_tf32=False) * scale
    
    
    DK_block_ptr = tl.make_block_ptr(
        base=DK + qkv_base_offset,
        shape=(L, D),
        strides=(stride_qk_seqlen, stride_qk_dim),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, D),
        order=(1, 0),
    )
    DV_block_ptr = tl.make_block_ptr(
        base=DV + qkv_base_offset,
        shape=(L, D),
        strides=(stride_qk_seqlen, stride_qk_dim),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, D),
        order=(1, 0),
    )
    tl.store(DK_block_ptr, dk.to(DK.dtype.element_ty))
    tl.store(DV_block_ptr, dv.to(DV.dtype.element_ty))
    
    
class LinearAttnFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, q, k, v, scale=None):
        B, H, L, K, V = *q.shape, v.shape[-1]
        BLOCK_M = 128
        
        assert K in {16, 32, 64, 128} and V in {16, 32, 64, 128}
        assert L % BLOCK_M == 0

        num_warps = 4 if K <= 64 else 8
        
        if scale is None:
            scale = k.shape[-2] ** -1.0

        NL = triton.cdiv(L, BLOCK_M)

        grid = (NL, B * H, 1)

        s = torch.empty(NL, B, H, K, V, device=k.device)
        z = torch.empty(NL, B, H, K, device=k.device)
        _fwd_kv_[grid](
            k, v, s, z,
            k.stride(1), k.stride(2), k.stride(3),
            s.stride(2), s.stride(3), s.stride(4),
            z.stride(2),
            scale,
            B, H, L, K,
            BLOCK_M=BLOCK_M,
        )
        s = s.sum(0).to(k.dtype)
        z = z.sum(0).to(k.dtype)
        
        o = torch.empty_like(q)
        _fwd_qs_[grid](
            q, s, o, z,
            q.stride(1), q.stride(2), q.stride(3),
            s.stride(1), s.stride(2), s.stride(3),
            z.stride(1),
            B, H, L, K,
            BLOCK_M=BLOCK_M,
        )
        
        ctx.save_for_backward(q, k, v, o, s, z)
        ctx.grid = grid
        ctx.scale = scale
        ctx.BLOCK_DMODEL = K
        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        q, k, v, o, s, z = ctx.saved_tensors
        B, H, L, K, V = *q.shape, v.shape[-1]
        BLOCK_M = 128
        
        assert K in {16, 32, 64, 128} and V in {16, 32, 64, 128}
        assert L % BLOCK_M == 0
        
        NL = triton.cdiv(L, BLOCK_M)

        grid = (NL, B * H, 1)
        
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
            ds.stride(2), ds.stride(3), ds.stride(4),
            z.stride(1), dz.stride(1),
            B, H, L, K,
            BLOCK_M=BLOCK_M,
            # num_warps=num_warps,
            # num_stages=4
        )
        ds = ds.sum(0)#.to(k.dtype)
        # print(ds.shape)
        grid = (NL, B * H, 1)
        _bwd_dkv_kernel[grid](
            k, v,
            dk, dv, ds,
            k.stride(1), k.stride(2), k.stride(3),
            ds.stride(1), ds.stride(2), ds.stride(3),
            ctx.scale,
            B, H, L, K,
            BLOCK_M=BLOCK_M,
            # num_warps=num_warps,
            # num_stages=4
        )
        
        
        dz_ = dz.unsqueeze(-1)
        dqk_k = dz_ * k.sum(dim=-2, keepdim=True)
        dqk_q = (dz_ * q).sum(dim=-2, keepdim=True)

        dq = dq + dqk_k
        dk = dk + dqk_q
        
        return dq, dk, dv, None, None


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