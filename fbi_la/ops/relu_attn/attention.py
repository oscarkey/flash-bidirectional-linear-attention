from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Q, K, V, O, Z,
    stride_q_bs, stride_q_head, stride_q_seqlen, stride_q_dim,
    stride_k_bs, stride_k_head, stride_k_seqlen, stride_k_dim,
    stride_v_bs, stride_v_head, stride_v_seqlen, stride_v_dim,
    stride_o_bs, stride_o_head, stride_o_seqlen, stride_o_dim,
    BS, HEAD, SEQLEN,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bs_head = tl.program_id(1)

    qkv_base_offset = off_bs_head * stride_q_head
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qkv_base_offset,
        shape=(SEQLEN, DIM),
        strides=(stride_q_seqlen, stride_q_dim),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qkv_base_offset,
        shape=(DIM, SEQLEN),
        strides=(stride_k_dim, stride_k_seqlen),
        offsets=(0, 0),
        block_shape=(DIM, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qkv_base_offset,
        shape=(SEQLEN, DIM),
        strides=(stride_k_seqlen, stride_v_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_N, DIM),
        order=(1, 0),
    )
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    
    out_buffer = tl.zeros([BLOCK_M, DIM], dtype=tl.float32)
    z_buffer = tl.zeros([BLOCK_M], dtype=tl.float32)

    q = tl.load(Q_block_ptr)

    for start_n in range(0, SEQLEN, BLOCK_N):
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        
        z = tl.dot(q, k, allow_tf32=False)
        z_buffer += tl.sum(z, axis=1)

        # k = (k * sm_scale).to(k.dtype)
        v = (v * sm_scale).to(k.dtype)
        
        kv = tl.dot(k, v)#, allow_tf32=False)
        
        out_buffer += tl.dot(q, kv.to(q.dtype))#, allow_tf32=False)

        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    out_buffer = out_buffer / (z_buffer[:, None] + 1e-6)

    O_block_ptr = tl.make_block_ptr(
        base=O + qkv_base_offset,
        shape=(SEQLEN, DIM),
        strides=(stride_o_seqlen, stride_o_dim),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, DIM),
        order=(1, 0),
    )
    tl.store(O_block_ptr, out_buffer.to(O.dtype.element_ty))
    
    Z_block_ptr = Z + off_bs_head * SEQLEN + offs_m
    tl.store(Z_block_ptr, z_buffer.to(Z.dtype.element_ty))


@triton.jit
def _bwd_preprocess(
    Out, DO,
    Delta,
    BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr,
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)
    # load
    o = tl.load(Out + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    # compute
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_m, delta)


@triton.jit
def _bwd_kernel(
    Q, K, V, sm_scale, Out, DO,
    DQ, DK, DV,
    L,
    D,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    Z, H, N_CTX,
    num_block,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H
    # qk_scale = sm_scale * 1.44269504
    qk_scale = sm_scale
    # offset pointers for batch/head
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_qz + off_h * stride_qh
    V += off_z * stride_qz + off_h * stride_qh
    DO += off_z * stride_qz + off_h * stride_qh
    DQ += off_z * stride_qz + off_h * stride_qh
    DK += off_z * stride_qz + off_h * stride_qh
    DV += off_z * stride_qz + off_h * stride_qh
    for start_n in range(0, num_block):
        lo = 0
        # initialize row/col offsets
        offs_qm = lo + tl.arange(0, BLOCK_M)
        offs_n = start_n * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_DMODEL)
        # initialize pointers to value-like data
        q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        k_ptrs = K + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        v_ptrs = V + (offs_n[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        do_ptrs = DO + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        dq_ptrs = DQ + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        # pointer to row-wise quantities in value-like data
        D_ptrs = D + off_hz * N_CTX
        l_ptrs = L + off_hz * N_CTX
        # initialize dv amd dk
        dv = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        dk = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        # k and v stay in SRAM throughout
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        v = (v * sm_scale).to(v.dtype)
        # loop over rows
        for start_m in range(lo, num_block * BLOCK_M, BLOCK_M):
            offs_m_curr = start_m + offs_m
            # load q, k, v, do on-chip
            q = tl.load(q_ptrs)
            # recompute p = softmax(qk, dim=-1).T
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, tl.trans(k))
            l_i = tl.load(l_ptrs + offs_m_curr)
            p = qk / (l_i[:, None] + 1e-6)
            # compute dv
            do = tl.load(do_ptrs)
            dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do)
            # compute dp = dot(v, do)
            Di = tl.load(D_ptrs + offs_m_curr)
            dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
            dp += tl.dot(do, tl.trans(v))
            # compute ds = p * (dp - delta[:, None])
            ds = dp / (l_i[:, None] + 1e-6)
            # compute dk = dot(ds.T, q)
            dk += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q)
            # compute dq
            dq = tl.load(dq_ptrs)
            dq += tl.dot(ds.to(Q.dtype.element_ty), k)
            tl.store(dq_ptrs, dq)
            # increment pointers
            dq_ptrs += BLOCK_M * stride_qm
            q_ptrs += BLOCK_M * stride_qm
            do_ptrs += BLOCK_M * stride_qm
        # write-back
        dv *= sm_scale
        dv_ptrs = DV + (offs_n[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        dk_ptrs = DK + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        tl.store(dv_ptrs, dv)
        tl.store(dk_ptrs, dk)
        
        
class ReluAttnFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, scale):
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}

        BLOCK_M = 128
        BLOCK_N = 64
        
        num_warps = 4 if Lk <= 64 else 8

        grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)

        o = torch.empty_like(q)
        z = torch.empty(q.shape[0], q.shape[1], q.shape[2], device=q.device)
        _fwd_kernel[grid](
            q, k, v, o, z,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[0], q.shape[1], q.shape[2],
            scale,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, DIM=Lk,
            num_warps=num_warps,
            num_stages=4
        )
        ctx.save_for_backward(q, k, v, o, z)
        ctx.grid = grid
        ctx.scale = scale
        ctx.BLOCK_DMODEL = Lk
        return o

    @staticmethod
    def backward(ctx, do):
        BLOCK = 128
        q, k, v, o, z = ctx.saved_tensors
        do = do.contiguous()
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        delta = torch.empty_like(z)
        _bwd_preprocess[(ctx.grid[0] * ctx.grid[1], )](
            o, do,
            delta,
            BLOCK_M=BLOCK, D_HEAD=ctx.BLOCK_DMODEL,
        )
        _bwd_kernel[(ctx.grid[1],)](
            q, k, v, ctx.scale,
            o, do,
            dq, dk, dv,
            z, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            q.shape[0], q.shape[1], q.shape[2],
            ctx.grid[0],
            BLOCK_M=BLOCK, BLOCK_N=BLOCK,
            BLOCK_DMODEL=ctx.BLOCK_DMODEL, num_warps=8,
            num_stages=1,
        )
        return dq, dk, dv, None, None


def relu_attention(
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
    o = ReluAttnFunction.apply(q, k, v, scale)
    return o