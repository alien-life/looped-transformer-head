"""Depth-sweep evaluation harness: does looping beat the frozen baseline?

    python -m lth.evaluate --model <hf id> --data test=data/abc/test.jsonl \\
        --head 1x=runs/1x/head.pt --head 5x=runs/5x/head.pt --head 10x=runs/10x/head.pt \\
        --head 50x=runs/50x/head.pt --control 1x --out reports/sweep --strict

For every head it measures, on the same held-out examples:

1. exact-match accuracy at each loop depth, against the R=0 frozen baseline
   (paired exact McNemar test), broken down by hop count and task;
2. time-to-first-token and inter-token latency per depth;
3. drift on general-purpose text (perplexity ratio, KL, top-1 agreement vs R=0).

It writes report.json, report.md and per-example predictions, and with --strict
exits non-zero when a required acceptance criterion fails.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch

from .data import Encoder, read_jsonl, score
from .general_prompts import GENERAL_TEXTS
from .model import DTYPES, LoopedHeadModel, eos_token_ids, load_base
from .train import autocast


# -- statistics ---------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar(a: list[bool], b: list[bool]) -> float:
    """Two-sided exact McNemar p-value for paired correctness vectors."""
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1))
    return min(1.0, 2 * tail / 2**n)


def summarise(rows: list[dict]) -> dict:
    def block(items):
        k, n = sum(r["correct"] for r in items), len(items)
        lo, hi = wilson(k, n)
        return {"n": n, "acc": k / n if n else 0.0, "ci": [lo, hi]}

    groups = {"hops": defaultdict(list), "task": defaultdict(list)}
    for r in rows:
        for key in groups:
            if r.get(key) is not None:
                groups[key][r[key]].append(r)
    out = block(rows)
    out["valid_json"] = sum(r["valid_json"] for r in rows) / max(len(rows), 1)
    out["by_hops"] = {str(h): block(v) for h, v in sorted(groups["hops"].items())}
    out["by_task"] = {t: block(v) for t, v in sorted(groups["task"].items())}
    return out


# -- measurements -------------------------------------------------------------


@torch.no_grad()
def run_accuracy(
    model, encoder, examples, loops, batch_size, max_new_tokens, eos_ids, amp
) -> list[dict]:
    order = sorted(
        range(len(examples)), key=lambda i: len(encoder.encode(examples[i])[0])
    )
    rows: list = [None] * len(examples)
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        chunk = [examples[i] for i in idx]
        batch = encoder.prompt_batch(chunk)
        with autocast(amp, model.head_device):
            out = model.generate(
                batch["input_ids"],
                batch["attention_mask"],
                num_loops=loops,
                max_new_tokens=max_new_tokens,
                eos_ids=eos_ids,
                pad_id=encoder.tok.pad_token_id,
            )
        for i, ex, toks in zip(idx, chunk, out.tokens.tolist()):
            text = encoder.tok.decode(toks, skip_special_tokens=True)
            rows[i] = {
                "id": ex["id"],
                "task": ex.get("task"),
                "hops": ex.get("hops"),
                "prediction": text,
                **score(text, ex["answer"]),
            }
    return rows


@torch.no_grad()
def measure_latency(model, encoder, examples, loops, new_tokens, amp) -> dict:
    """Median TTFT / ITL at batch size 1 with a fixed number of new tokens."""

    def once(ex):
        batch = encoder.prompt_batch([ex])
        with autocast(amp, model.head_device):
            return model.generate(
                batch["input_ids"],
                batch["attention_mask"],
                num_loops=loops,
                max_new_tokens=new_tokens,
                ignore_eos=True,
                timed=True,
            )

    once(examples[0])  # warm-up
    ttft, itl = [], []
    for ex in examples:
        out = once(ex)
        ttft.append(out.ttft * 1e3)
        itl.extend(t * 1e3 for t in out.itl)
    return {
        "ttft_ms": statistics.median(ttft),
        "itl_ms": statistics.median(itl) if itl else 0.0,
    }


@torch.no_grad()
def measure_regression(model, tok, texts, depths, max_tokens, amp) -> dict:
    """Next-token drift on general text for each depth, relative to the frozen model."""
    nll0, n_tok = 0.0, 0
    acc = {r: {"nll": 0.0, "kl": 0.0, "agree": 0} for r in depths}
    bos = getattr(tok, "bos_token_id", None)
    for text in texts:
        ids = ([bos] if bos is not None else []) + list(
            tok(text, add_special_tokens=False)["input_ids"]
        )
        ids = torch.tensor(ids[:max_tokens])[None]
        if ids.shape[1] < 2:
            continue
        with autocast(amp, model.head_device):
            h_base, h_in, _ = model.encode_states(ids)
            target = ids[0, 1:].to(h_base.device)
            logp0 = model.hidden_to_logits(h_base).float().log_softmax(-1)[0, :-1]
            nll0 -= float(logp0.gather(1, target[:, None]).sum())
            n_tok += target.numel()
            for r in depths:
                logp = (
                    model.hidden_to_logits(model.head(h_base, r, h_in=h_in).h_out)
                    .float()
                    .log_softmax(-1)[0, :-1]
                )
                acc[r]["nll"] -= float(logp.gather(1, target[:, None]).sum())
                acc[r]["kl"] += float((logp0.exp() * (logp0 - logp)).sum())
                acc[r]["agree"] += int((logp.argmax(-1) == logp0.argmax(-1)).sum())
    n_tok = max(n_tok, 1)
    ppl0 = math.exp(nll0 / n_tok)
    out = {"tokens": n_tok, "baseline_ppl": ppl0, "depths": {}}
    for r, m in acc.items():
        ppl = math.exp(m["nll"] / n_tok)
        out["depths"][str(r)] = {
            "ppl": ppl,
            "ppl_ratio": ppl / ppl0,
            "kl": m["kl"] / n_tok,
            "top1_agreement": m["agree"] / n_tok,
        }
    return out


# -- acceptance ---------------------------------------------------------------


def acceptance(
    name, meta, sweep, correct, control, latency, regression, args
) -> list[dict]:
    """Acceptance criteria for one head on the primary dataset.

    sweep:   {depth: summary}, including depth 0 (the frozen baseline)
    correct: {depth: per-example correctness}
    control: (name, per-example correctness) of the 1x-trained head at R=1, or None
    """
    depths = sorted(sweep)
    trained = meta.get("train_loops") or max(depths)
    sampled = bool(meta.get("sample_loops"))
    in_range = [r for r in depths if 1 <= r <= trained]
    star = max(in_range)
    out = []

    def add(criterion, passed, detail, required=True):
        out.append(
            {
                "criterion": criterion,
                "passed": passed,
                "required": required,
                "detail": detail,
            }
        )

    gain = sweep[star]["acc"] - sweep[0]["acc"]
    p = mcnemar(correct[0], correct[star])
    add(
        "beats frozen baseline",
        gain > 0 and p < args.alpha,
        f"R={star}: {sweep[star]['acc']:.3f} vs R=0: {sweep[0]['acc']:.3f} (gain {gain:+.3f}, p={p:.2g})",
    )

    path = [0] + in_range
    drops = [
        (a, b)
        for a, b in zip(path, path[1:])
        if sweep[b]["acc"] < sweep[a]["acc"] - args.mono_tol
    ]
    detail = " -> ".join(f"R{r}:{sweep[r]['acc']:.3f}" for r in path)
    if drops:
        detail += " | drops at " + ", ".join(f"R{a}->R{b}" for a, b in drops)
    if not sampled and len(in_range) > 1:
        detail += " | head was trained at a single depth; shallower depths are off-distribution"
    add(
        "monotonic up to trained depth",
        not drops,
        detail,
        required=sampled or len(in_range) == 1,
    )

    if star > 1:
        if control is not None:
            c_name, c_correct = control
            c_acc = sum(c_correct) / len(c_correct)
            p = mcnemar(c_correct, correct[star])
            add(
                "looping beats single-pass control",
                sweep[star]["acc"] > c_acc and p < args.alpha,
                f"R={star}: {sweep[star]['acc']:.3f} vs '{c_name}' at R=1: {c_acc:.3f} (p={p:.2g})",
            )
        if 1 in sweep:
            p = mcnemar(correct[1], correct[star])
            add(
                "deeper beats own first pass",
                sweep[star]["acc"] > sweep[1]["acc"] and p < args.alpha,
                f"R={star}: {sweep[star]['acc']:.3f} vs R=1: {sweep[1]['acc']:.3f} (p={p:.2g})",
                required=sampled and control is None,
            )

    beyond = [r for r in depths if r > trained]
    if beyond:
        worst = min(beyond, key=lambda r: sweep[r]["acc"])
        add(
            "stable beyond trained depth",
            sweep[worst]["acc"] >= sweep[star]["acc"] - args.mono_tol,
            f"worst at R={worst}: {sweep[worst]['acc']:.3f} vs R={star}: {sweep[star]['acc']:.3f}",
            required=False,
        )

    if latency and (args.sla_ttft_ms or args.sla_itl_ms):
        lat = latency[str(star)]
        ok = (not args.sla_ttft_ms or lat["ttft_ms"] <= args.sla_ttft_ms) and (
            not args.sla_itl_ms or lat["itl_ms"] <= args.sla_itl_ms
        )
        add(
            "latency within SLA",
            ok,
            f"R={star}: TTFT {lat['ttft_ms']:.1f} ms, ITL {lat['itl_ms']:.1f} ms",
        )

    if regression:
        reg = regression["depths"][str(star)]
        add(
            "no regression on general text",
            reg["ppl_ratio"] <= 1 + args.regression_tol,
            f"R={star}: perplexity x{reg['ppl_ratio']:.4f} (limit x{1 + args.regression_tol:.2f}), "
            f"top-1 agreement {reg['top1_agreement']:.3f}",
        )
    return out


# -- report -------------------------------------------------------------------


def render_markdown(report: dict) -> str:
    lines = [f"# Looped-head depth sweep\n\nBase model: `{report['model']}`\n"]
    for ds_name, ds in report["accuracy"].items():
        base = ds["baseline"]
        hops = sorted(base["by_hops"], key=int)
        lines.append(f"## Accuracy: {ds_name} (n={base['n']})\n")
        lines.append(
            "| head | R | accuracy | 95% CI | vs R=0 | p | valid JSON | "
            + " | ".join(f"{h} hops" for h in hops)
            + " |"
        )
        lines.append("|---|---|---|---|---|---|---|" + "---|" * len(hops))

        def row(head, r, s, mark="", p=None):
            delta = "" if r == 0 else f"{s['acc'] - base['acc']:+.3f}"
            cells = [
                head,
                f"{r}{mark}",
                f"{s['acc']:.3f}",
                f"{s['ci'][0]:.3f}-{s['ci'][1]:.3f}",
                delta,
                "" if p is None else f"{p:.2g}",
                f"{s['valid_json']:.3f}",
            ]
            cells += [
                f"{s['by_hops'][h]['acc']:.3f}" if h in s["by_hops"] else ""
                for h in hops
            ]
            lines.append("| " + " | ".join(cells) + " |")

        row("frozen baseline", 0, base)
        for head, info in ds["heads"].items():
            trained = report["heads"][head]["meta"].get("train_loops")
            for r, s in info.items():
                mark = (
                    ""
                    if trained is None
                    else (
                        "*" if int(r) == trained else ("†" if int(r) > trained else "")
                    )
                )
                row(head, int(r), s, mark, s["p_vs_baseline"])
        lines.append(
            "\n`*` trained depth, `†` deeper than the head was trained. p: exact McNemar vs R=0.\n"
        )

    if report.get("latency"):
        lat = report["latency"]
        lines.append(
            f"## Latency (batch 1, {lat['prompts']} prompts, {lat['new_tokens']} new tokens)\n"
        )
        lines.append(
            "| head | R | TTFT ms | ITL ms | ITL vs R=0 |\n|---|---|---|---|---|"
        )
        b = lat["baseline"]
        lines.append(
            f"| frozen baseline | 0 | {b['ttft_ms']:.1f} | {b['itl_ms']:.1f} | |"
        )
        for head, rows in lat["heads"].items():
            for r, m in rows["depths"].items():
                lines.append(
                    f"| {head} | {r} | {m['ttft_ms']:.1f} | {m['itl_ms']:.1f} | "
                    f"x{m['itl_ms'] / max(b['itl_ms'], 1e-9):.2f} |"
                )
            lines.append(f"| {head} | per loop | | +{rows['itl_ms_per_loop']:.3f} | |")
        lines.append("")

    if report.get("regression"):
        lines.append("## General-text regression\n")
        lines.append(
            "| head | R | perplexity | ratio vs R=0 | KL (nats/token) | top-1 agreement |\n|---|---|---|---|---|---|"
        )
        first = next(iter(report["regression"].values()))
        lines.append(f"| frozen baseline | 0 | {first['baseline_ppl']:.4g} | | | |")
        for head, reg in report["regression"].items():
            for r, m in reg["depths"].items():
                lines.append(
                    f"| {head} | {r} | {m['ppl']:.4g} | x{m['ppl_ratio']:.4g} | {m['kl']:.3g} | "
                    f"{m['top1_agreement']:.3f} |"
                )
        lines.append("")

    lines.append("## Acceptance\n")
    lines.append("| head | criterion | result | detail |\n|---|---|---|---|")
    for head, info in report["heads"].items():
        for c in info["acceptance"]:
            result = ("PASS" if c["passed"] else "FAIL") + (
                "" if c["required"] else " (info)"
            )
            lines.append(f"| {head} | {c['criterion']} | {result} | {c['detail']} |")
        lines.append(
            f"| {head} | **verdict** | **{'PASS' if info['accepted'] else 'FAIL'}** | required criteria only |"
        )
    return "\n".join(lines) + "\n"


# -- CLI ----------------------------------------------------------------------


def _named(items, default_prefix: str) -> dict[str, str]:
    out = {}
    for i, item in enumerate(items or []):
        name, sep, path = item.partition("=")
        if not sep:
            name, path = (
                f"{default_prefix}{i}" if len(items) > 1 else default_prefix,
                item,
            )
        if name in out:
            raise SystemExit(f"duplicate name '{name}'")
        out[name] = path
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--data",
        action="append",
        required=True,
        help="[NAME=]test.jsonl; first one is primary",
    )
    ap.add_argument(
        "--head", action="append", help="[NAME=]head.pt; repeat to compare heads"
    )
    ap.add_argument("--control", help="name of the single-pass (1x) control head")
    ap.add_argument("--depths", default="0,1,5,10,25,50")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="evaluate only the first N examples per dataset",
    )
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument(
        "--latency-prompts", type=int, default=8, help="0 disables the latency sweep"
    )
    ap.add_argument("--latency-tokens", type=int, default=16)
    ap.add_argument("--sla-ttft-ms", type=float, default=None)
    ap.add_argument("--sla-itl-ms", type=float, default=None)
    ap.add_argument(
        "--general",
        help="JSONL of {'text': ...} for the regression check; default: built-in set",
    )
    ap.add_argument("--no-regression", action="store_true")
    ap.add_argument(
        "--regression-tol",
        type=float,
        default=0.01,
        help="allowed relative perplexity increase",
    )
    ap.add_argument(
        "--mono-tol",
        type=float,
        default=0.01,
        help="accuracy drop tolerated between depths",
    )
    ap.add_argument(
        "--alpha", type=float, default=0.05, help="significance level for paired tests"
    )
    ap.add_argument("--amp", choices=["none", "bf16", "fp16"], default="none")
    ap.add_argument("--dtype", choices=list(DTYPES), default="bf16")
    ap.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 if any head fails a required criterion",
    )
    args = ap.parse_args(argv)

    datasets = _named(args.data, "test")
    heads = _named(args.head, "head")
    if args.control and args.control not in heads:
        raise SystemExit(
            f"--control '{args.control}' is not one of the --head names {list(heads)}"
        )
    sweep_depths = sorted({int(d) for d in args.depths.split(",")} - {0})
    out_dir = Path(args.out)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)

    base, tok = load_base(
        args.model, args.dtype, None if args.quant == "none" else args.quant
    )
    model = LoopedHeadModel.bare(base).eval()
    model.verify_parity()
    eos_ids = eos_token_ids(base, tok)
    encoder = Encoder(tok, args.max_length)
    examples = {name: read_jsonl(path)[: args.limit] for name, path in datasets.items()}
    primary = next(iter(datasets))
    lat_examples = examples[primary][: args.latency_prompts]
    texts = (
        [r["text"] for r in read_jsonl(args.general)] if args.general else GENERAL_TEXTS
    )

    def accuracy(ds: str, tag: str, loops: int) -> list[dict]:
        rows = run_accuracy(
            model,
            encoder,
            examples[ds],
            loops,
            args.batch_size,
            args.max_new_tokens,
            eos_ids,
            args.amp,
        )
        with open(out_dir / "predictions" / f"{ds}__{tag}__R{loops}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        acc = sum(r["correct"] for r in rows) / max(len(rows), 1)
        print(f"[{ds}] {tag} R={loops}: accuracy {acc:.3f}", flush=True)
        return rows

    report: dict = {
        "model": args.model,
        "depths": [0] + sweep_depths,
        "accuracy": {},
        "heads": {},
    }
    correct: dict = {}  # (dataset, head, depth) -> per-example correctness
    for ds in datasets:
        rows = accuracy(ds, "baseline", 0)
        correct[ds, None, 0] = [r["correct"] for r in rows]
        report["accuracy"][ds] = {"baseline": summarise(rows), "heads": {}}
    if lat_examples:
        report["latency"] = {
            "prompts": len(lat_examples),
            "new_tokens": args.latency_tokens,
            "baseline": measure_latency(
                model, encoder, lat_examples, 0, args.latency_tokens, args.amp
            ),
            "heads": {},
        }
    if not args.no_regression and heads:
        report["regression"] = {}

    head_depths: dict[str, list[int]] = {}
    for name, path in heads.items():
        meta = model.load_head(path)
        model.eval()
        max_loops = model.head.config.max_loops
        trained = meta.get("train_loops")
        depths = sorted(
            {d for d in sweep_depths if d <= max_loops}
            | ({trained} if trained else set())
        )
        head_depths[name] = depths
        gate_tensor = (
            model.head.proj_out.weight
            if model.head.config.gate == "matrix"
            else model.head.alpha
        )
        report["heads"][name] = {
            "path": path,
            "meta": meta,
            "params": model.head.num_parameters(),
            "alpha": model.head.alpha.detach().abs().mean().item(),
            "gate": gate_tensor.detach().abs().mean().item(),
        }
        for ds in datasets:
            per_depth = {}
            for r in depths:
                rows = accuracy(ds, name, r)
                correct[ds, name, r] = [x["correct"] for x in rows]
                per_depth[str(r)] = {
                    **summarise(rows),
                    "p_vs_baseline": mcnemar(
                        correct[ds, None, 0], correct[ds, name, r]
                    ),
                }
            report["accuracy"][ds]["heads"][name] = per_depth
        if lat_examples:
            per = {
                str(r): measure_latency(
                    model, encoder, lat_examples, r, args.latency_tokens, args.amp
                )
                for r in depths
            }
            xs = [0] + depths
            ys = [report["latency"]["baseline"]["itl_ms"]] + [
                per[str(r)]["itl_ms"] for r in depths
            ]
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / max(
                sum((x - mx) ** 2 for x in xs), 1e-9
            )
            report["latency"]["heads"][name] = {"depths": per, "itl_ms_per_loop": slope}
        if "regression" in report:
            report["regression"][name] = measure_regression(
                model, tok, texts, depths, 256, args.amp
            )

    for name in heads:
        control = None
        if (
            args.control
            and name != args.control
            and (primary, args.control, 1) in correct
        ):
            control = (args.control, correct[primary, args.control, 1])
        sweep = {0: report["accuracy"][primary]["baseline"]}
        sweep.update(
            {int(r): s for r, s in report["accuracy"][primary]["heads"][name].items()}
        )
        per_correct = {
            0: correct[primary, None, 0],
            **{r: correct[primary, name, r] for r in head_depths[name]},
        }
        checks = acceptance(
            name,
            report["heads"][name]["meta"],
            sweep,
            per_correct,
            control,
            report.get("latency", {}).get("heads", {}).get(name, {}).get("depths"),
            report.get("regression", {}).get(name),
            args,
        )
        report["heads"][name]["acceptance"] = checks
        report["heads"][name]["accepted"] = all(
            c["passed"] for c in checks if c["required"]
        )

    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    markdown = render_markdown(report)
    (out_dir / "report.md").write_text(markdown)
    print("\n" + markdown)
    print(f"wrote {out_dir / 'report.md'} and {out_dir / 'report.json'}")
    failed = [n for n, h in report["heads"].items() if not h["accepted"]]
    return 1 if args.strict and failed else 0


if __name__ == "__main__":
    sys.exit(main())
