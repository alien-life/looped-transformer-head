"""Looped Transformer Head (`fn_looped_attn`).

A small weight-shared transformer block that is run R times on top of a frozen
backbone's final hidden states, with First-Pass KV Reuse:

* pass 1 computes keys/values for every layer of the block and keeps them;
* passes 2..R compute fresh queries from the updated hidden state and attend
  to the pass-1 keys/values, then run the full FFN.

Because pass-1 keys/values at position t depend only on positions <= t, the
same code path serves parallel (teacher-forced) training and incremental
decoding with a 1x KV cache. `tests/test_head.py` checks that equivalence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class HeadConfig:
    hidden_size: int  # width of the frozen backbone's final hidden state
    num_heads: int = 16
    num_layers: int = 1  # layers inside the shared block (1-4)
    max_loops: int = 50  # size of the step-embedding table
    head_dim: Optional[int] = None  # bottleneck width of the block; None = hidden_size
    ffn_mult: float = 8 / 3  # SwiGLU hidden = ffn_mult * width
    gate: str = "scalar"  # zero-init residual gate: "scalar", "vector" or "matrix"
    rope_base: float = 10000.0
    tap_layer: Optional[int] = (
        None  # backbone hidden_states index the block READS; None = final state
    )
    standardize: bool = False  # per-dimension standardisation of the block's input
    norm_out: bool = (
        False  # RMSNorm on (x - z0) before proj_out for depth-invariant scale
    )
    qk_norm: bool = (
        False  # per-head RMSNorm on Q and K before RoPE to prevent logit saturation
    )
    use_rope: bool = (
        True  # apply RoPE to Q/K; False = pure content-addressable attention (NoPE)
    )

    @property
    def remap(self) -> bool:
        """The block works in coordinates that differ from the final hidden state's."""
        return self.tap_layer is not None or self.standardize

    @property
    def width(self) -> int:
        return self.head_dim or self.hidden_size

    def validate(self) -> None:
        if self.width % self.num_heads:
            raise ValueError(
                f"width {self.width} is not divisible by num_heads {self.num_heads}"
            )
        if (self.width // self.num_heads) % 2:
            raise ValueError("per-head dimension must be even (rotary embeddings)")
        if self.gate not in ("scalar", "vector", "matrix"):
            raise ValueError("gate must be 'scalar', 'vector' or 'matrix'")
        if self.num_layers < 1 or self.max_loops < 1:
            raise ValueError("num_layers and max_loops must be >= 1")


@dataclass
class HeadCache:
    """Pass-1 keys/values of the looped block. Grows by one entry per decoded token."""

    k: list[torch.Tensor]  # per layer: (B, heads, T, head_dim), rotary already applied
    v: list[torch.Tensor]  # per layer: (B, heads, T, head_dim)
    key_mask: torch.Tensor  # (B, T) bool, False = padding


@dataclass
class HeadOutput:
    h_out: torch.Tensor  # refined hidden state after the last loop
    exits: dict[int, torch.Tensor]  # loop count -> refined hidden state at that loop
    cache: Optional[HeadCache] = None


def _apply_rope(
    x: torch.Tensor, rot: Optional[tuple[torch.Tensor, torch.Tensor]]
) -> torch.Tensor:
    if rot is None:
        return x
    cos, sin = rot[0].to(x.dtype), rot[1].to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, mult: float):
        super().__init__()
        hidden = max(8, int(round(dim * mult / 8)) * 8)
        self.gate_up = nn.Linear(dim, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class LoopedLayer(nn.Module):
    """Pre-norm self-attention + SwiGLU layer with explicit Q / K / V paths."""

    def __init__(
        self, dim: int, num_heads: int, ffn_mult: float, qk_norm: bool = False
    ):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        head_dim = dim // num_heads
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6) if qk_norm else nn.Identity()
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.ffn = SwiGLU(dim, ffn_mult)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        return x.view(b, t, self.num_heads, d // self.num_heads).transpose(1, 2)

    def attn_input(self, x: torch.Tensor, step_vec: torch.Tensor) -> torch.Tensor:
        # The step embedding is injected into the attention input only, not the residual.
        return self.norm1(x + step_vec)

    def keys_values(self, a_in: torch.Tensor, rot) -> tuple[torch.Tensor, torch.Tensor]:
        k = _apply_rope(self.k_norm(self._split(self.k_proj(a_in))), rot)
        return k, self._split(self.v_proj(a_in))

    def attend(self, x, a_in, k, v, rot, mask) -> torch.Tensor:
        q = _apply_rope(self.q_norm(self._split(self.q_proj(a_in))), rot)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        b, _, t, _ = a.shape
        x = x + self.o_proj(a.transpose(1, 2).reshape(b, t, -1))
        return x + self.ffn(self.norm2(x))


class LoopedHead(nn.Module):
    def __init__(self, config: HeadConfig):
        super().__init__()
        config.validate()
        self.config = config
        d, w = config.hidden_size, config.width
        self.proj_in = nn.Identity() if w == d else nn.Linear(d, w, bias=False)
        self.norm_out = nn.RMSNorm(w, eps=1e-6) if config.norm_out else None
        matrix_gate = config.gate == "matrix"
        self.proj_out = (
            nn.Linear(w, d, bias=False)
            if (w != d or config.remap or matrix_gate)
            else nn.Identity()
        )
        if config.standardize:
            self.register_buffer("in_mean", torch.zeros(d))
            self.register_buffer("in_std", torch.ones(d))
        self.layers = nn.ModuleList(
            LoopedLayer(w, config.num_heads, config.ffn_mult, qk_norm=config.qk_norm)
            for _ in range(config.num_layers)
        )
        self.step_embed = nn.Embedding(config.max_loops, w)
        # Zero-init gate: h_out == h_base exactly until the gate moves.
        #   scalar / vector: h_out = h_base + alpha * out(x_R - x_0), alpha starts at 0.
        #   matrix: the output projection itself starts at 0 and alpha is fixed at 1, so every
        #           output direction gets its own scale instead of sharing one global factor.
        if matrix_gate:
            self.register_buffer("alpha", torch.ones(1))
        else:
            self.alpha = nn.Parameter(torch.zeros(1 if config.gate == "scalar" else d))
        half = (w // config.num_heads) // 2
        inv_freq = 1.0 / (
            config.rope_base ** (torch.arange(half, dtype=torch.float32) / half)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.apply(self._init_weights)
        if matrix_gate:
            nn.init.zeros_(self.proj_out.weight)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def set_input_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Per-dimension statistics of the states the block reads (saved with the head)."""
        if not self.config.standardize:
            raise ValueError("head was built with standardize=False")
        self.in_mean.copy_(mean.to(self.in_mean))
        self.in_std.copy_(std.to(self.in_std).clamp_min(1e-6))

    # -- one pass over the block -------------------------------------------------

    def _first_pass(self, x, rot, mask, cache: Optional[HeadCache]):
        """Pass 1: compute (and keep) keys/values for every layer."""
        step = self.step_embed.weight[0]
        ks, vs = [], []
        for i, layer in enumerate(self.layers):
            a_in = layer.attn_input(x, step)
            k, v = layer.keys_values(a_in, rot)
            if cache is not None:
                k = torch.cat([cache.k[i].to(k.dtype), k], dim=2)
                v = torch.cat([cache.v[i].to(v.dtype), v], dim=2)
            ks.append(k)
            vs.append(v)
            x = layer.attend(x, a_in, k, v, rot, mask)
        return x, ks, vs

    def _later_pass(self, x, r: int, ks, vs, rot, mask):
        """Passes 2..R: fresh queries against the pass-1 keys/values."""
        step = self.step_embed.weight[r]
        for layer, k, v in zip(self.layers, ks, vs):
            x = layer.attend(x, layer.attn_input(x, step), k, v, rot, mask)
        return x

    def _emit(self, h0, z0, x) -> torch.Tensor:
        delta = x - z0
        if self.norm_out is not None:
            delta = self.norm_out(delta)
        return h0 + self.alpha * self.proj_out(delta)

    # -- forward -----------------------------------------------------------------

    def forward(
        self,
        h_base: torch.Tensor,
        num_loops: int,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        exits: Optional[Sequence[int]] = None,
        bptt_window: Optional[int] = None,
        use_checkpoint: bool = False,
        cache: Optional[HeadCache] = None,
        return_cache: bool = False,
        last_only: bool = False,
        h_in: Optional[torch.Tensor] = None,
    ) -> HeadOutput:
        """Refine `h_base` with `num_loops` passes of the shared block.

        h_base:         (B, T_new, D) final backbone hidden states for the new positions.
                        The block's output is added to these.
        h_in:           (B, T_new, D) states the block reads, when the head taps an earlier
                        backbone layer (`config.tap_layer`). Default: `h_base`.
        attention_mask: (B, T_cached + T_new), 1 = token, 0 = padding. None = no padding.
        exits:          loop counts at which to emit an output (deep supervision /
                        early exit). Must include `num_loops`. Default: only `num_loops`.
        bptt_window:    truncated BPTT. A pass keeps its graph only if it lies within
                        `bptt_window` passes of some exit. Pass-1 keys/values always
                        keep their graph, so K/V projections still receive gradient.
        use_checkpoint: recompute pass activations during backward instead of storing them.
        cache:          pass-1 K/V of already-processed positions (incremental decoding).
        last_only:      after pass 1, keep iterating only the last position. Valid when
                        only the next-token prediction is needed (the prompt at inference).
        """
        cfg = self.config
        if not 0 <= num_loops <= cfg.max_loops:
            raise ValueError(
                f"num_loops must be in [0, {cfg.max_loops}], got {num_loops}"
            )
        if num_loops == 0:
            h = h_base[:, -1:] if last_only else h_base
            return HeadOutput(h, {0: h}, cache)
        exit_list = sorted(set(exits)) if exits else [num_loops]
        if exit_list[0] < 1 or exit_list[-1] != num_loops:
            raise ValueError("exits must lie in [1, num_loops] and include num_loops")
        if bptt_window is not None and bptt_window < 1:
            raise ValueError("bptt_window must be >= 1")

        bsz, t_new, _ = h_base.shape
        device = h_base.device
        if cfg.tap_layer is not None and h_in is None:
            raise ValueError(
                f"this head reads backbone layer {cfg.tap_layer}: pass those states as h_in"
            )
        h0 = h_base.to(self.alpha.dtype)
        source = h0 if h_in is None else h_in.to(self.alpha.dtype)
        if cfg.standardize:
            source = (source - self.in_mean) / self.in_std
        z0 = self.proj_in(source)

        t_past = 0 if cache is None else cache.key_mask.shape[1]
        if attention_mask is None:
            key_mask = torch.ones(bsz, t_past + t_new, dtype=torch.bool, device=device)
        else:
            key_mask = attention_mask.to(device=device, dtype=torch.bool)
            if key_mask.shape != (bsz, t_past + t_new):
                raise ValueError(
                    f"attention_mask must cover cached + new positions: expected "
                    f"{(bsz, t_past + t_new)}, got {tuple(key_mask.shape)}"
                )

        # Rotary positions count real tokens, so padding side does not matter.
        if cfg.use_rope:
            pos = (key_mask.long().cumsum(-1) - 1).clamp_min(0)[:, t_past:]
            ang = pos[..., None].to(self.inv_freq.dtype) * self.inv_freq
            rot: Optional[tuple[torch.Tensor, torch.Tensor]] = (
                ang.cos()[:, None],
                ang.sin()[:, None],
            )
        else:
            rot = None

        # Causal + padding mask, True = may attend. The diagonal stays on so padded
        # query rows never become all-masked (their outputs are discarded anyway).
        q_idx = torch.arange(t_past, t_past + t_new, device=device)[:, None]
        k_idx = torch.arange(t_past + t_new, device=device)[None, :]
        mask = ((k_idx <= q_idx)[None] & key_mask[:, None, :]) | (k_idx == q_idx)[None]
        mask = mask[:, None]

        grad_on = torch.is_grad_enabled()

        def in_window(p: int) -> bool:
            return bptt_window is None or any(
                e - bptt_window < p <= e for e in exit_list
            )

        x, ks, vs = self._first_pass(z0, rot, mask, cache)
        if grad_on and not in_window(1):
            x = x.detach()  # K/V keep their graph; the hidden state does not.
        if last_only:
            x, h0, z0 = x[:, -1:], h0[:, -1:], z0[:, -1:]
            if rot is not None:
                rot = (rot[0][:, :, -1:], rot[1][:, :, -1:])
            mask = mask[:, :, -1:]

        outs: dict[int, torch.Tensor] = {}
        if 1 in exit_list:
            outs[1] = self._emit(h0, z0, x)
        for p in range(2, num_loops + 1):
            if grad_on and not in_window(p):
                with torch.no_grad():
                    x = self._later_pass(x, p - 1, ks, vs, rot, mask)
            elif grad_on and use_checkpoint:
                x = checkpoint(
                    self._later_pass, x, p - 1, ks, vs, rot, mask, use_reentrant=False
                )
            else:
                x = self._later_pass(x, p - 1, ks, vs, rot, mask)
            if p in exit_list:
                outs[p] = self._emit(h0, z0, x)

        new_cache = HeadCache(ks, vs, key_mask) if return_cache else None
        return HeadOutput(outs[num_loops], outs, new_cache)
