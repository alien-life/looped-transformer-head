"""Per-loop decoding probe: which hop's answer does the head hold after each loop?

    python -m lth.probe --model <hf id> --data data/names/test.jsonl \\
        --head plain=runs/names/3x-plain/head.pt --head chain=runs/names/3x-chain/head.pt --out reports/probe

The prompt and the gold final answer are fed in (teacher forcing) and every loop's output
is decoded through the frozen lm_head. For each hop count and loop it reports

* on target: the loop reproduces what chain supervision would ask of it
  (hop r's answer token for an intermediate loop, the whole final answer otherwise);
* holds hop j: at the token where the entity is decided, the loop's top prediction is
  hop j's entity. Shown for chains of 2+ hops, where that token can be located.

Run it on a head trained without chain supervision to see whether intermediate
entities appear on their own. Needs `chain_answers` in the data.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from .data import Encoder, read_jsonl
from .evaluate import _named
from .model import DTYPES, LoopedHeadModel, load_base
from .train import autocast


@torch.no_grad()
def probe(model, encoder, examples, loops: int, batch_size: int = 16, amp: str = "none") -> list[dict]:
    """One row per (hops, loop). `loops == 0` probes the frozen backbone as a single row."""
    dev = model.head_device
    exits = list(range(1, loops + 1)) if loops else [0]
    stats = defaultdict(lambda: {"n": 0, "on_target": 0, "located": 0, "holds": defaultdict(int)})
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        batch = encoder.train_batch(chunk)
        labels = encoder.chain_batch(chunk, max(loops, 1))
        located = [(i, encoder.chain_tokens(ex)) for i, ex in enumerate(chunk)]
        located = [(i, info) for i, info in located if info]
        rows = torch.tensor([i for i, _ in located], device=dev, dtype=torch.long)
        # sequence position whose output predicts the entity token
        cols = torch.tensor([len(encoder.encode(chunk[i])[0]) + d - 1 for i, (d, _) in located],
                            device=dev, dtype=torch.long)
        with autocast(amp, dev):
            h_base, h_in, _ = model.encode_states(batch["input_ids"], batch["attention_mask"])
            if loops:
                states = model.head(h_base, loops, batch["attention_mask"].to(dev), h_in=h_in, exits=exits).exits
            else:
                states = {0: h_base}
            for r in exits:
                h = states[r]
                lab = labels[max(r, 1)].to(dev)[:, 1:]
                select = lab != -100
                pred = model.hidden_to_logits(h[:, :-1][select]).argmax(-1)
                misses = torch.zeros(len(chunk), dtype=torch.long, device=dev)
                misses.index_add_(0, select.nonzero()[:, 0], (pred != lab[select]).long())
                held = model.hidden_to_logits(h[rows, cols]).argmax(-1).tolist() if located else []
                for i, ex in enumerate(chunk):
                    cell = stats[ex["hops"], r]
                    cell["n"] += 1
                    cell["on_target"] += int(misses[i] == 0)
                for (i, (_, tokens)), token in zip(located, held):
                    cell = stats[chunk[i]["hops"], r]
                    cell["located"] += 1
                    cell["holds"][tokens.index(token) + 1 if token in tokens else 0] += 1
    out = []
    for (hops, r), cell in sorted(stats.items()):
        target = "answer" if (r == 0 or r == loops or r >= hops) else f"hop {r}"
        holds = {str(j): c / cell["located"] for j, c in sorted(cell["holds"].items())} if cell["located"] else {}
        out.append({"hops": hops, "loop": r, "n": cell["n"], "target": target,
                    "on_target": cell["on_target"] / cell["n"], "holds": holds})
    return out


def render_markdown(results: dict[str, list[dict]]) -> str:
    max_hops = max((row["hops"] for rows in results.values() for row in rows), default=1)
    lines = ["# Per-loop decoding probe\n"]
    for name, rows in results.items():
        lines.append(f"## {name}\n")
        lines.append("| hops | n | loop | trained target | on target | "
                     + " | ".join(f"holds hop {j}" for j in range(1, max_hops + 1)) + " | other |")
        lines.append("|---|---|---|---|---|" + "---|" * (max_hops + 1))
        for row in rows:
            cells = [str(row["hops"]), str(row["n"]), str(row["loop"]), row["target"], f"{row['on_target']:.3f}"]
            for j in list(range(1, max_hops + 1)) + [0]:
                show = row["holds"] and (j == 0 or j <= row["hops"])
                cells.append(f"{row['holds'].get(str(j), 0.0):.3f}" if show else "")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    lines.append("`holds hop j`: share of examples whose top prediction at the entity token is hop j's "
                 "entity (`other`: anything else). Loop 0 is the frozen backbone.\n")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="JSONL with `chain_answers`")
    ap.add_argument("--head", action="append", required=True, help="[NAME=]head.pt; repeat to compare heads")
    ap.add_argument("--loops", type=int, default=None, help="default: each head's trained depth")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--amp", choices=["none", "bf16", "fp16"], default="none")
    ap.add_argument("--dtype", choices=list(DTYPES), default="bf16")
    ap.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none")
    args = ap.parse_args(argv)

    base, tok = load_base(args.model, args.dtype, None if args.quant == "none" else args.quant)
    model = LoopedHeadModel.bare(base).eval()
    encoder = Encoder(tok)
    examples = [ex for ex in read_jsonl(args.data) if ex.get("chain_answers")][: args.limit]
    if not examples:
        raise SystemExit(f"{args.data}: no example has `chain_answers`")

    results = {"frozen backbone": probe(model, encoder, examples, 0, args.batch_size, args.amp)}
    for name, path in _named(args.head, "head").items():
        meta = model.load_head(path)
        model.eval()
        loops = args.loops or meta.get("train_loops") or model.head.config.max_loops
        tag = f"{name} (R={loops}, {'chain-supervised' if meta.get('chain_supervision') else 'final answer only'})"
        results[tag] = probe(model, encoder, examples, loops, args.batch_size, args.amp)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe.json").write_text(json.dumps(results, indent=2) + "\n")
    markdown = render_markdown(results)
    (out / "probe.md").write_text(markdown)
    print(markdown)
    print(f"wrote {out / 'probe.md'} and {out / 'probe.json'}")


if __name__ == "__main__":
    main()
