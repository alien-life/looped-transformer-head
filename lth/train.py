"""Train the looped head on an A -> B -> C dataset. The backbone stays frozen.

    python -m lth.train --model <hf id> --data data/abc --out runs/5x --tier 5x
    python -m lth.train --model <hf id> --data data/abc --out runs/10x --tier 10x --init-from runs/5x/head.pt
    python -m lth.train --model <hf id> --data data/abc --out runs/50x --tier 50x --init-from runs/10x/head.pt

`--tier 1x` trains the same head with a single pass. It is the control that
separates "a trained adapter helps" from "looping helps".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .data import Encoder, read_jsonl, score
from .head import HeadConfig
from .model import DTYPES, LoopedHeadModel, eos_token_ids, load_base


@dataclass(frozen=True)
class Tier:
    loops: int  # loop depth R (upper bound when sample_loops)
    bptt_window: Optional[int]  # None = full BPTT
    sample_loops: bool  # draw R ~ Uniform(1, loops) every step
    exits: tuple[int, ...]  # deep-supervision exits (the sampled R is always added)
    checkpoint: bool  # gradient checkpointing across passes


TIERS = {
    "1x": Tier(1, None, False, (), False),
    "2x": Tier(2, None, False, (), False),
    "3x": Tier(3, None, False, (), False),
    "5x": Tier(5, None, False, (), True),
    "10x": Tier(10, None, False, (), True),
    "50x": Tier(50, 8, True, (5, 10, 25, 50), True),
}


def step_plan(tier: Tier, rng: random.Random) -> tuple[int, list[int]]:
    """Loop depth and supervised exits for one optimisation step."""
    loops = rng.randint(1, tier.loops) if tier.sample_loops else tier.loops
    return loops, sorted({e for e in tier.exits if e < loops} | {loops})


def autocast(amp: str, device: torch.device):
    enabled = amp != "none" and device.type == "cuda"
    return torch.autocast("cuda", dtype=DTYPES.get(amp, torch.float16), enabled=enabled)


@torch.no_grad()
def evaluate_val(
    model,
    encoder,
    examples,
    loops: int,
    batch_size: int,
    amp: str,
    eos_ids,
    n_generate: int,
) -> dict:
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(examples), batch_size):
        batch = encoder.train_batch(examples[i : i + batch_size])
        n = int((batch["labels"][:, 1:] != -100).sum())
        with autocast(amp, model.head_device):
            loss = model(
                batch["input_ids"],
                batch["attention_mask"],
                num_loops=loops,
                labels=batch["labels"],
            ).loss
        total, count = total + float(loss) * n, count + n
    correct = 0
    subset = examples[:n_generate]
    for i in range(0, len(subset), batch_size):
        chunk = subset[i : i + batch_size]
        batch = encoder.prompt_batch(chunk)
        with autocast(amp, model.head_device):
            out = model.generate(
                batch["input_ids"],
                batch["attention_mask"],
                num_loops=loops,
                max_new_tokens=32,
                eos_ids=eos_ids,
                pad_id=encoder.tok.pad_token_id,
            )
        for ex, toks in zip(chunk, out.tokens.tolist()):
            correct += score(
                encoder.tok.decode(toks, skip_special_tokens=True), ex["answer"]
            )["correct"]
    model.train()
    return {"val_loss": total / max(count, 1), "val_acc": correct / max(len(subset), 1)}


class HiddenCache:
    """Frozen backbone states for every training example, memory-mapped from disk as fp16.

    The backbone never changes, so computing its states once removes it from the
    training loop. `h` holds the states the head reads, for every token. When the head
    taps an earlier layer, the final-layer states are also needed, but only where the
    loss is taken, so `final` holds them for each example's supervised positions.

    `path` is a directory (h.npy, final.npy, key.json). The arrays stay on disk and are
    paged in on demand, so a training process does not hold gigabytes of them in RAM.
    Reused when it matches this model, tap layer and data; otherwise built.
    """

    def __init__(
        self,
        model,
        encoder,
        examples,
        path,
        model_id: str,
        amp: str,
        batch_size: int = 32,
    ):
        path = Path(path)
        tap = model.head.config.tap_layer
        rows = [encoder.encode(ex) for ex in examples]
        self.lengths = [len(p) + len(t) for p, t in rows]
        # positions whose output predicts a target token: last prompt token .. second-to-last target token
        self.spans = [(len(p) - 1, len(t)) for p, t in rows]
        digest = hashlib.sha1(
            (model_id if tap is None else f"{model_id}|tap={tap}").encode()
        )
        for p, t in rows:
            digest.update(struct.pack(f"{len(p) + len(t)}i", *p, *t))
        key = digest.hexdigest()
        self.offsets = [0]
        for n in self.lengths:
            self.offsets.append(self.offsets[-1] + n)
        self.final_offsets = [0]
        for _, n in self.spans:
            self.final_offsets.append(self.final_offsets[-1] + n)

        if path.is_file():
            raise SystemExit(
                f"{path} is a single-file cache from an older version; delete it"
            )
        key_file, h_file, final_file = (
            path / "key.json",
            path / "h.npy",
            path / "final.npy",
        )
        if key_file.exists():
            if json.loads(key_file.read_text())["key"] != key:
                raise SystemExit(
                    f"{path} was built for a different model, tap layer or dataset; "
                    "delete it or use another path"
                )
        else:
            self._build(
                model,
                encoder,
                examples,
                amp,
                batch_size,
                h_file,
                final_file if tap is not None else None,
            )
            key_file.write_text(
                json.dumps({"key": key, "model": model_id, "tap_layer": tap}) + "\n"
            )
        self.h = np.load(h_file, mmap_mode="r")
        self.final = np.load(final_file, mmap_mode="r") if tap is not None else None

    def _build(
        self, model, encoder, examples, amp, batch_size, h_file, final_file
    ) -> None:
        """Write the arrays straight to disk. key.json is written last, by the caller, so an
        interrupted build is never mistaken for a finished one."""
        h_file.parent.mkdir(parents=True, exist_ok=True)
        width = model.lm_head.weight.shape[1]
        h = np.lib.format.open_memmap(
            h_file, mode="w+", dtype=np.float16, shape=(self.offsets[-1], width)
        )
        final = None
        if final_file is not None:
            final = np.lib.format.open_memmap(
                final_file,
                mode="w+",
                dtype=np.float16,
                shape=(self.final_offsets[-1], width),
            )
        was_training = model.training
        model.eval()
        order = sorted(range(len(examples)), key=self.lengths.__getitem__)
        with torch.no_grad():
            for start in range(0, len(order), batch_size):
                idx = order[start : start + batch_size]
                batch = encoder.train_batch([examples[i] for i in idx])
                with autocast(amp, model.head_device):
                    h_base, h_in, _ = model.encode_states(
                        batch["input_ids"], batch["attention_mask"]
                    )
                read = (
                    (h_base if h_in is None else h_in).to("cpu", torch.float16).numpy()
                )
                last = (
                    h_base.to("cpu", torch.float16).numpy()
                    if final is not None
                    else None
                )
                for row, i in enumerate(idx):
                    h[self.offsets[i] : self.offsets[i + 1]] = read[
                        row, : self.lengths[i]
                    ]
                    if final is not None:
                        a, n = self.spans[i]
                        final[self.final_offsets[i] : self.final_offsets[i + 1]] = last[
                            row, a : a + n
                        ]
        h.flush()
        if final is not None:
            final.flush()
        del h, final
        model.train(was_training)

    def gigabytes(self) -> float:
        return (
            (self.h.size + (0 if self.final is None else self.final.size)) * 2 / 2**30
        )

    def batch(
        self, picks: list[int], width: int
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """(h_base, h_in) padded to `width`, as `LoopedHeadModel.forward` expects them."""
        read = torch.zeros(len(picks), width, self.h.shape[1], dtype=torch.float16)
        for row, i in enumerate(picks):
            read[row, : self.lengths[i]] = torch.from_numpy(
                np.array(self.h[self.offsets[i] : self.offsets[i + 1]])
            )
        if self.final is None:
            return read, None
        final = torch.zeros_like(read)  # only the supervised positions are ever used
        for row, i in enumerate(picks):
            a, n = self.spans[i]
            final[row, a : a + n] = torch.from_numpy(
                np.array(self.final[self.final_offsets[i] : self.final_offsets[i + 1]])
            )
        return final, read

    def stats(self, chunk: int = 50_000) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-dimension mean and standard deviation of the states the head reads."""
        total = torch.zeros(self.h.shape[1], dtype=torch.float64)
        square = torch.zeros_like(total)
        for start in range(0, self.h.shape[0], chunk):
            x = torch.from_numpy(np.array(self.h[start : start + chunk])).double()
            total += x.sum(0)
            square += x.pow(2).sum(0)
        mean = total / self.h.shape[0]
        return mean.float(), (square / self.h.shape[0] - mean.pow(2)).clamp_min(
            0
        ).sqrt().float()


@torch.no_grad()
def live_input_stats(model, encoder, examples, amp: str, batch_size: int = 16):
    """Same statistics from a sample of examples, when there is no cache."""
    chunks = []
    for start in range(0, len(examples), batch_size):
        batch = encoder.train_batch(examples[start : start + batch_size])
        with autocast(amp, model.head_device):
            h_base, h_in, _ = model.encode_states(
                batch["input_ids"], batch["attention_mask"]
            )
        read = h_base if h_in is None else h_in
        chunks.append(
            read[batch["attention_mask"].bool().to(read.device)].float().cpu()
        )
    x = torch.cat(chunks)
    return x.mean(0), x.std(0)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--model",
        required=True,
        help="HF model id or path, or 'tiny-random' for a smoke test",
    )
    ap.add_argument(
        "--data", required=True, help="directory with train.jsonl and val.jsonl"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--tier", choices=list(TIERS), default="5x")
    ap.add_argument(
        "--init-from",
        help="head checkpoint to continue from (curriculum 5x -> 10x -> 50x)",
    )
    # head (ignored with --init-from)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--num-heads", type=int, default=16)
    ap.add_argument(
        "--head-dim",
        type=int,
        default=None,
        help="bottleneck width; default = backbone width",
    )
    ap.add_argument("--max-loops", type=int, default=50)
    ap.add_argument(
        "--gate",
        choices=["scalar", "vector", "matrix"],
        default="matrix",
        help="zero-init residual gate. 'matrix' zero-initialises the output projection instead of "
        "using one learned factor for the whole output",
    )
    ap.add_argument(
        "--norm-out",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply RMSNorm to (x - z0) before proj_out for depth-invariant output scale",
    )
    ap.add_argument(
        "--qk-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply per-head RMSNorm to Q and K before RoPE to prevent attention logit saturation",
    )
    ap.add_argument(
        "--rope",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply rotary position embeddings in the head; --no-rope uses content-only attention (NoPE)",
    )
    ap.add_argument(
        "--tap-layer",
        type=int,
        default=None,
        help="backbone hidden_states index the head reads (0 = embeddings); its output is still "
        "added to the final state. Implies --standardize. Default: read the final state.",
    )
    ap.add_argument(
        "--standardize",
        action="store_true",
        help="standardise the head's input per dimension, using training-set statistics",
    )
    # optimisation
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument(
        "--alpha-lr-mult",
        type=float,
        default=1.0,
        help="LR multiplier for the scalar/vector gate",
    )
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument(
        "--schedule",
        choices=["wsd", "cosine"],
        default="wsd",
        help="LR schedule after warmup: 'wsd' holds constant LR for 75%% of post-warmup steps then cosine decays "
        "to 0.1x (avoids decaying before induction-head phase transitions); 'cosine' decays immediately",
    )
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument(
        "--bptt-window",
        type=int,
        default=None,
        help="override the tier's truncated-BPTT window",
    )
    ap.add_argument(
        "--no-checkpoint", action="store_true", help="disable gradient checkpointing"
    )
    ap.add_argument(
        "--chain-supervision",
        action="store_true",
        help="train loop r to hold the answer after hop r (needs `chain_answers` in the data); "
        "the last loop is still trained on the final answer. Fixed-depth tiers only.",
    )
    ap.add_argument("--amp", choices=["none", "bf16", "fp16"], default="none")
    # backbone
    ap.add_argument("--dtype", choices=list(DTYPES), default="bf16")
    ap.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument(
        "--cache-hidden",
        metavar="DIR",
        help="directory holding the backbone's hidden states for the training set (built if "
        "missing, memory-mapped when used). Training then skips the backbone forward. "
        "Disk: about 2 bytes x tokens x width.",
    )
    # bookkeeping
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument(
        "--eval-examples",
        type=int,
        default=64,
        help="val examples decoded for accuracy",
    )
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    tier = TIERS[args.tier]
    window = args.bptt_window if args.bptt_window is not None else tier.bptt_window
    use_ckpt = tier.checkpoint and not args.no_checkpoint
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    base, tok = load_base(
        args.model, args.dtype, None if args.quant == "none" else args.quant
    )
    if args.init_from:
        model = LoopedHeadModel.bare(base)
        model.load_head(args.init_from)
    else:
        hidden = base.get_output_embeddings().weight.shape[1]
        model = LoopedHeadModel(
            base,
            HeadConfig(
                hidden_size=hidden,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                max_loops=args.max_loops,
                head_dim=args.head_dim,
                gate=args.gate,
                tap_layer=args.tap_layer,
                standardize=args.standardize or args.tap_layer is not None,
                norm_out=args.norm_out,
                qk_norm=args.qk_norm,
                use_rope=args.rope,
            ),
        )
    if tier.loops > model.head.config.max_loops:
        raise SystemExit(f"tier {args.tier} needs max_loops >= {tier.loops}")
    model.verify_parity()
    model.train()
    eos_ids = eos_token_ids(base, tok)

    encoder = Encoder(tok, args.max_length)
    data_dir = Path(args.data)
    train = [ex for ex in read_jsonl(data_dir / "train.jsonl") if encoder.fits(ex)]
    val = [ex for ex in read_jsonl(data_dir / "val.jsonl") if encoder.fits(ex)]
    if len(train) < args.batch_size:
        raise SystemExit(
            f"only {len(train)} training examples fit in --max-length {args.max_length}"
        )
    print(
        f"head: {model.head.num_parameters() / 1e6:.2f}M trainable params | tier {args.tier} "
        f"(R={tier.loops}, bptt={window}, sample={tier.sample_loops}, exits={tier.exits}) | "
        f"{len(train)} train / {len(val)} val examples"
    )

    if args.chain_supervision:
        if tier.sample_loops:
            raise SystemExit(
                "--chain-supervision needs a fixed-depth tier (1x, 2x, 3x, 5x, 10x)"
            )
        with_chain = [ex for ex in train if ex.get("chain_answers")]
        if not with_chain:
            raise SystemExit(
                "--chain-supervision: no training example has `chain_answers`"
            )
        sharp = sum(encoder.chain_sharp(ex) for ex in with_chain) / len(with_chain)
        print(
            f"chain supervision: {len(with_chain)}/{len(train)} examples carry a chain; {sharp:.0%} of them "
            f"put each hop's answer on a single token"
            + (
                ""
                if sharp == 1
                else " (the rest are supervised on the first differing token only)"
            )
        )

    cached = None
    if args.cache_hidden:
        tic = time.time()
        cached = HiddenCache(
            model, encoder, train, args.cache_hidden, args.model, args.amp
        )
        print(
            f"hidden-state cache: {cached.gigabytes():.2f} GB ready in {time.time() - tic:.0f}s"
        )
    if model.head.config.standardize and not args.init_from:
        mean, std = (
            cached.stats()
            if cached
            else live_input_stats(model, encoder, train[:256], args.amp)
        )
        model.head.set_input_stats(mean, std)
        model.train()

    params = dict(model.head.named_parameters())
    decay_params = [p for n, p in params.items() if n != "alpha" and p.ndim >= 2]
    no_decay_params = [p for n, p in params.items() if n != "alpha" and p.ndim < 2]
    block_params = decay_params + no_decay_params
    alpha_params = [params["alpha"]] if "alpha" in params else []
    groups = [{"params": decay_params, "weight_decay": args.weight_decay}]
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0})
    if (
        alpha_params
    ):  # scalar / vector gate; the matrix gate has no separate gate parameter
        groups.append(
            {
                "params": alpha_params,
                "weight_decay": 0.0,
                "lr": args.lr * args.alpha_lr_mult,
            }
        )
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95))

    def gate_size() -> float:
        gate = (
            model.head.proj_out.weight
            if model.head.config.gate == "matrix"
            else model.head.alpha
        )
        return gate.detach().abs().mean().item()

    def lr_scale(step: int) -> float:
        if step < args.warmup:
            return (step + 1) / args.warmup
        progress = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
        if args.schedule == "wsd":
            if progress < 0.75:
                return 1.0
            progress = (progress - 0.75) / 0.25
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=args.amp == "fp16" and model.head_device.type == "cuda"
    )

    meta = {
        "base_model": args.model,
        "tier": args.tier,
        "train_loops": tier.loops,
        "sample_loops": tier.sample_loops,
        "bptt_window": window,
        "exits": list(tier.exits),
        "init_from": args.init_from,
        "steps": 0,
        "chain_supervision": args.chain_supervision,
    }
    config_out = {**vars(args), **{k: v for k, v in meta.items() if k != "steps"}}
    (out_dir / "config.json").write_text(json.dumps(config_out, indent=2) + "\n")
    log = open(out_dir / "train_log.jsonl", "a")

    def record(row: dict) -> None:
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(
            "  ".join(
                f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in row.items()
            ),
            flush=True,
        )

    order: list[int] = []
    best = math.inf
    skipped = 0
    tic, running = time.time(), 0.0
    for step in range(1, args.steps + 1):
        loops, exits = step_plan(tier, rng)
        for _ in range(args.grad_accum):
            if len(order) < args.batch_size:
                order = rng.sample(range(len(train)), len(train))
            picks = [order.pop() for _ in range(args.batch_size)]
            batch = encoder.train_batch([train[i] for i in picks])
            labels = batch["labels"]
            if args.chain_supervision:
                exits = list(range(1, loops + 1))
                labels = encoder.chain_batch([train[i] for i in picks], loops)
            states, read = (
                cached.batch(picks, batch["input_ids"].shape[1])
                if cached
                else (None, None)
            )
            with autocast(args.amp, model.head_device):
                out = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    num_loops=loops,
                    labels=labels,
                    exits=exits,
                    bptt_window=window,
                    use_checkpoint=use_ckpt,
                    h_base=states,
                    h_in=read,
                )
            scaler.scale(out.loss / args.grad_accum).backward()
            running += out.loss.item() / args.grad_accum
        scaler.unscale_(opt)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(block_params, args.clip))
        alpha_norm = (
            float(torch.nn.utils.clip_grad_norm_(alpha_params, args.clip))
            if alpha_params
            else 0.0
        )
        stepped = True
        if scaler.is_enabled():
            scale_before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            stepped = scaler.get_scale() >= scale_before
        elif math.isfinite(grad_norm) and math.isfinite(alpha_norm):
            opt.step()
        else:
            stepped = False
        opt.zero_grad(set_to_none=True)
        if stepped:
            sched.step()
        else:
            skipped += 1

        if step % args.log_every == 0:
            row = {
                "step": step,
                "loss": running / args.log_every,
                "loops": loops,
                "gate": gate_size(),
                "grad_norm": grad_norm,
                "lr": sched.get_last_lr()[0],
                "it_per_s": args.log_every / (time.time() - tic),
            }
            if skipped:
                row["skipped"] = skipped
            if model.head_device.type == "cuda":
                row["gpu_gb"] = torch.cuda.max_memory_allocated() / 2**30
            record(row)
            tic, running = time.time(), 0.0

        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate_val(
                model,
                encoder,
                val,
                tier.loops,
                args.batch_size,
                args.amp,
                eos_ids,
                args.eval_examples,
            )
            record({"step": step, "eval_loops": tier.loops, **metrics})
            meta["steps"] = step
            model.save_head(out_dir / "head.pt", {**meta, **metrics})
            if metrics["val_loss"] < best:
                best = metrics["val_loss"]
                model.save_head(out_dir / "head_best.pt", {**meta, **metrics})
            tic = time.time()
    log.close()
    print(f"saved {out_dir / 'head.pt'} (best val loss {best:.4f} in head_best.pt)")


if __name__ == "__main__":
    main()
