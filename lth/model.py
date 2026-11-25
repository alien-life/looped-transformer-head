"""Frozen backbone + trainable looped head."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .head import HeadConfig, LoopedHead

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def load_base(model_id: str, dtype: str = "bf16", quant: Optional[str] = None, device_map="auto"):
    """Load a causal LM and its tokenizer. `tiny-random` builds an offline smoke-test model."""
    if model_id == "tiny-random":
        from .tiny import build_tiny

        return build_tiny(device="cuda" if torch.cuda.is_available() else "cpu")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs = {"torch_dtype": DTYPES[dtype], "device_map": device_map}
    if quant in ("4bit", "8bit"):
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=quant == "4bit",
            load_in_8bit=quant == "8bit",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=DTYPES[dtype],
        )
    base = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return base, tokenizer


def eos_token_ids(base, tokenizer) -> list[int]:
    """Every id that should end generation (model eos plus chat end-of-turn tokens)."""
    ids: set[int] = set()
    for source in (getattr(base, "generation_config", None), tokenizer):
        value = getattr(source, "eos_token_id", None)
        if isinstance(value, int):
            ids.add(value)
        elif value:
            ids.update(int(v) for v in value)
    return sorted(ids)


@dataclass
class LossOutput:
    loss: torch.Tensor
    exit_losses: dict[int, torch.Tensor]


@dataclass
class GenerationOutput:
    tokens: torch.Tensor        # (B, n_new)
    ttft: float                 # seconds until the first new token
    itl: list[float]            # seconds between subsequent tokens


class LoopedHeadModel(nn.Module):
    def __init__(self, base: nn.Module, head_config: Optional[HeadConfig] = None, **head_kwargs):
        super().__init__()
        self.base = base.eval().requires_grad_(False)
        lm_head = base.get_output_embeddings()
        if lm_head is None:
            raise ValueError("base model exposes no output embeddings (lm_head)")
        hidden = lm_head.weight.shape[1]
        config = head_config or HeadConfig(hidden_size=hidden, **head_kwargs)
        if config.hidden_size != hidden:
            raise ValueError(f"head hidden_size {config.hidden_size} != backbone hidden size {hidden}")
        self.head = LoopedHead(config).to(lm_head.weight.device)
        text_config = getattr(base.config, "text_config", None)
        self.softcap = getattr(base.config, "final_logit_softcapping", None) or getattr(
            text_config, "final_logit_softcapping", None
        )
        self.head_meta: dict = {}

    @classmethod
    def bare(cls, base: nn.Module) -> "LoopedHeadModel":
        """Wrap `base` with a throwaway head: baseline (R=0) use, or before `load_head`."""
        hidden = base.get_output_embeddings().weight.shape[1]
        return cls(base, HeadConfig(hidden_size=hidden, num_heads=1, head_dim=8, max_loops=1))

    # -- plumbing ----------------------------------------------------------------

    @property
    def backbone(self) -> nn.Module:
        return self.base.base_model

    @property
    def lm_head(self) -> nn.Module:
        return self.base.get_output_embeddings()

    @property
    def head_device(self) -> torch.device:
        return self.head.alpha.device

    @property
    def input_device(self) -> torch.device:
        return self.base.get_input_embeddings().weight.device

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()  # the backbone never leaves eval mode
        return self

    def trainable_parameters(self):
        return self.head.parameters()

    def hidden_to_logits(self, h: torch.Tensor) -> torch.Tensor:
        weight = self.lm_head.weight
        dtype = weight.dtype if weight.is_floating_point() else torch.bfloat16
        logits = self.lm_head(h.to(dtype))
        if self.softcap:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return logits

    @torch.no_grad()
    def encode_states(self, input_ids, attention_mask=None, past_key_values=None, use_cache: bool = False):
        """Frozen backbone forward.

        Returns (final hidden states, states the head reads or None, kv cache). The second
        item is the tapped layer's output when the head has `tap_layer`, otherwise None
        (the head then reads the final states).
        """
        dev = self.input_device
        tap = self.head.config.tap_layer
        out = self.backbone(
            input_ids=input_ids.to(dev),
            attention_mask=None if attention_mask is None else attention_mask.to(dev),
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=tap is not None,
        )
        h_in = None if tap is None else out.hidden_states[tap].to(self.head_device)
        return out.last_hidden_state.to(self.head_device), h_in, (out.past_key_values if use_cache else None)

    def encode(self, input_ids, attention_mask=None, past_key_values=None, use_cache: bool = False):
        """(final hidden states on the head device, kv cache)."""
        h_base, _, past = self.encode_states(input_ids, attention_mask, past_key_values, use_cache)
        return h_base, past

    @torch.no_grad()
    def verify_parity(self, input_ids: Optional[torch.Tensor] = None) -> float:
        """Check that backbone -> lm_head reproduces the base model's own logits.

        Guards against architectures that put something between the final hidden
        state and the logits that this wrapper does not replicate.
        """
        if input_ids is None:
            vocab = self.lm_head.weight.shape[0]
            lo = 1000 if vocab > 3000 else 0
            gen = torch.Generator().manual_seed(0)
            input_ids = torch.randint(lo, min(vocab, lo + 1000), (1, 8), generator=gen)
        ref = self.base(input_ids=input_ids.to(self.input_device)).logits.float()
        h, _ = self.encode(input_ids)
        got = self.hidden_to_logits(h).float().to(ref.device)
        err = (ref - got).abs().max().item()
        scale = max(1.0, ref.abs().max().item())
        if not err <= 1e-3 * scale:
            raise RuntimeError(
                f"backbone+lm_head does not reproduce base logits (max abs err {err:.4g}); "
                "this architecture needs a custom hidden_to_logits"
            )
        return err

    # -- training / scoring forward ----------------------------------------------

    def forward(
        self,
        input_ids,
        attention_mask=None,
        num_loops: int = 5,
        labels=None,
        exits: Optional[Sequence[int]] = None,
        bptt_window: Optional[int] = None,
        use_checkpoint: bool = False,
        h_base: Optional[torch.Tensor] = None,
        h_in: Optional[torch.Tensor] = None,
    ):
        """Without labels: logits (or {exit: logits} when `exits` is given).

        With labels (-100 = ignore): `LossOutput`, cross-entropy averaged over exits.
        `labels` may be {exit: labels} to give every exit its own targets (chain supervision).
        Logits are computed only at supervised positions, which matters for large
        vocabularies (a full B x T x V logits tensor is never materialised).
        `h_base` (and `h_in` for a head that taps an earlier layer) supply precomputed
        backbone states and skip the backbone forward. With a tap, `h_base` only needs to
        be valid at the supervised positions: the block reads `h_in` everywhere else.
        """
        if h_base is None:
            h_base, h_in, _ = self.encode_states(input_ids, attention_mask)
        else:
            h_base = h_base.to(self.head_device)
            h_in = None if h_in is None else h_in.to(self.head_device)
        mask = None if attention_mask is None else attention_mask.to(h_base.device)
        out = self.head(
            h_base, num_loops, mask, h_in=h_in, exits=exits, bptt_window=bptt_window,
            use_checkpoint=use_checkpoint,
        )
        if labels is None:
            if exits:
                return {e: self.hidden_to_logits(h) for e, h in out.exits.items()}
            return self.hidden_to_logits(out.h_out)

        losses = {}
        for e, h in out.exits.items():
            exit_labels = (labels[e] if isinstance(labels, dict) else labels).to(h_base.device)
            select = exit_labels[:, 1:] != -100
            if select.any():
                logits = self.hidden_to_logits(h[:, :-1][select]).float()
                losses[e] = F.cross_entropy(logits, exit_labels[:, 1:][select])
        return LossOutput(torch.stack(list(losses.values())).mean(), losses)

    # -- cached greedy decoding --------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        attention_mask=None,
        num_loops: int = 5,
        max_new_tokens: int = 32,
        eos_ids: Sequence[int] = (),
        pad_id: int = 0,
        ignore_eos: bool = False,
        timed: bool = False,
    ) -> GenerationOutput:
        """Greedy decoding with a backbone KV cache and a 1x looped-head KV cache.

        Batches must be LEFT-padded. For the prompt, pass 1 runs over every position
        (to build keys/values) and passes 2..R run on the last position only.
        """
        def clock() -> float:
            if timed and torch.cuda.is_available():
                torch.cuda.synchronize()
            return time.perf_counter()

        bsz = input_ids.shape[0]
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        mask = attention_mask.to(self.head_device)
        eos = torch.tensor(list(eos_ids), device=self.head_device, dtype=torch.long)
        finished = torch.zeros(bsz, dtype=torch.bool, device=self.head_device)
        tokens, stamps = [], []
        start = clock()

        h, h_in, past = self.encode_states(input_ids, attention_mask, use_cache=True)
        cache = None
        if num_loops > 0:
            out = self.head(h, num_loops, mask, h_in=h_in, return_cache=True, last_only=True)
            h, cache = out.h_out, out.cache
        for i in range(max_new_tokens):
            nxt = self.hidden_to_logits(h[:, -1]).argmax(-1)
            nxt = torch.where(finished, torch.full_like(nxt, pad_id), nxt)
            tokens.append(nxt)
            stamps.append(clock())
            if not ignore_eos and eos.numel():
                finished |= torch.isin(nxt, eos)
            if i == max_new_tokens - 1 or bool(finished.all()):
                break
            mask = torch.cat([mask, mask.new_ones(bsz, 1)], dim=1)
            h, h_in, past = self.encode_states(nxt[:, None], mask, past_key_values=past, use_cache=True)
            if num_loops > 0:
                out = self.head(h, num_loops, mask, h_in=h_in, cache=cache, return_cache=True)
                h, cache = out.h_out, out.cache

        itl = [b - a for a, b in zip(stamps, stamps[1:])]
        return GenerationOutput(torch.stack(tokens, dim=1), stamps[0] - start, itl)

    # -- head checkpoints (the backbone is never saved) --------------------------

    def save_head(self, path, meta: Optional[dict] = None) -> None:
        torch.save(
            {
                "config": asdict(self.head.config),
                "state_dict": {k: v.detach().cpu() for k, v in self.head.state_dict().items()},
                "meta": meta or {},
            },
            path,
        )

    def load_head(self, path) -> dict:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        config = HeadConfig(**ckpt["config"])
        if config.hidden_size != self.lm_head.weight.shape[1]:
            raise ValueError("checkpoint was trained for a backbone with a different hidden size")
        head = LoopedHead(config)
        head.load_state_dict(ckpt["state_dict"])
        self.head = head.to(self.head_device)
        self.head_meta = ckpt.get("meta", {})
        return self.head_meta
