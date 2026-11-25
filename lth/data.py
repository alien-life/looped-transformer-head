"""A -> B -> C dataset: schema, synthetic generators, scoring, tokenisation.

One JSON object per line:

    id        unique string
    task      generator family ("lookup_chain" | "rule_eval" | "ledger" | your own)
    hops      number of latent reasoning steps needed
    context   A: the records the model reads
    question  A: what is asked
    format    the JSON shape the answer must take (shown to the model)
    chain     B: gold intermediate states. Never shown to the model; kept for diagnostics.
    answer    C: the gold JSON object
    chance    accuracy of guessing among the candidates present in the context
    chain_answers  optional: the answer after each hop, in the same JSON shape as
              `answer` (the last one equals `answer`). Enables per-loop supervision
              (`lth.train --chain-supervision`): loop r is trained to hold hop r's answer.

Bring your own data by writing the same fields (`chain`, `chance`, `chain_answers` are optional).

    python -m lth.data --out data/abc --train 8000 --val 300 --test 600 --hops 2-4 --ood-hops 5-6
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Callable, Optional

import torch

INSTRUCTION = "You are a business operations assistant. Answer using only the records below."


def _ids(rng: random.Random, prefix: str, n: int) -> list[str]:
    return [f"{prefix}-{v}" for v in rng.sample(range(100, 1000), n)]


def _money(v: int) -> str:
    return f"${v:,}"


# -- task 1: relational composition -------------------------------------------

LOOKUP_LAYERS = [
    ("account", "ACC"), ("sales rep", "REP"), ("team", "TEAM"), ("region", "RGN"),
    ("price book", "PB"), ("approval policy", "POL"), ("cost center", "CC"), ("ledger", "LED"),
]
LOOKUP_FACTS = [
    "Account {a} is owned by sales rep {b}.",
    "Sales rep {a} reports to team {b}.",
    "Team {a} operates in region {b}.",
    "Region {a} is governed by price book {b}.",
    "Price book {a} falls under approval policy {b}.",
    "Approval policy {a} is audited by cost center {b}.",
    "Cost center {a} is funded by ledger {b}.",
]


def gen_lookup_chain(rng: random.Random, hops: int) -> dict:
    """Follow `hops` links through shuffled records, with parallel distractor chains."""
    n_chains = rng.randint(3, 5)
    ents = [_ids(rng, LOOKUP_LAYERS[layer][1], n_chains) for layer in range(hops + 1)]
    facts = [
        LOOKUP_FACTS[layer].format(a=ents[layer][c], b=ents[layer + 1][c])
        for layer in range(hops)
        for c in range(n_chains)
    ]
    rng.shuffle(facts)
    t = rng.randrange(n_chains)
    name = LOOKUP_LAYERS[hops][0]
    return {
        "context": "\n".join(facts),
        "question": f"Which {name} is account {ents[0][t]} ultimately linked to?",
        "format": '{"answer": "<%s id>"}' % name,
        "chain": [{"hop": i + 1, "from": ents[i][t], "to": ents[i + 1][t]} for i in range(hops)],
        "answer": {"answer": ents[hops][t]},
        "chance": 1 / n_chains,
    }


# -- task 2: rule application -------------------------------------------------

RULE_LAYERS = [("customer", "CUS"), ("segment", "SEG"), ("contract", "CON"), ("credit policy", "CRP")]
RULE_LINKS = [
    "Customer {a} belongs to segment {b}.",
    "Segment {a} is served under contract {b}.",
    "Contract {a} follows credit policy {b}.",
]
RULE = (
    "Rule: an order is approved if its amount is at most the order limit that applies to it; "
    "otherwise it is escalated."
)


def gen_rule_eval(rng: random.Random, hops: int) -> dict:
    """Resolve an order's limit through `hops - 2` links, read it, then compare."""
    depth = hops - 2  # entity layers between the order and its limit
    n_chains = rng.randint(3, 4)
    orders = _ids(rng, "ORD", n_chains)
    ents = [_ids(rng, RULE_LAYERS[layer][1], n_chains) for layer in range(depth)]
    limits = [500 * v for v in rng.sample(range(4, 41), n_chains)]
    amounts = []
    for limit in limits:
        delta = 100 * rng.randint(1, 15)
        amounts.append(limit + delta if rng.random() < 0.5 else max(100, limit - delta))
    facts = []
    for c in range(n_chains):
        facts.append(f"Order {orders[c]} for {_money(amounts[c])} was placed by customer {ents[0][c]}.")
        for layer in range(depth - 1):
            facts.append(RULE_LINKS[layer].format(a=ents[layer][c], b=ents[layer + 1][c]))
        holder = RULE_LAYERS[depth - 1][0].capitalize()
        facts.append(f"{holder} {ents[depth - 1][c]} carries an order limit of {_money(limits[c])}.")
    rng.shuffle(facts)
    t = rng.randrange(n_chains)
    decision = "approved" if amounts[t] <= limits[t] else "escalated"
    chain = [{"hop": 1, "from": orders[t], "to": ents[0][t]}]
    chain += [{"hop": i + 2, "from": ents[i][t], "to": ents[i + 1][t]} for i in range(depth - 1)]
    chain.append({"hop": depth + 1, "from": ents[depth - 1][t], "to": limits[t]})
    chain.append({"hop": depth + 2, "compare": [amounts[t], limits[t]], "to": decision})
    return {
        "context": RULE + "\n" + "\n".join(facts),
        "question": f"What order limit applies to order {orders[t]}, and is the order approved or escalated?",
        "format": '{"limit": <number>, "decision": "approved" or "escalated"}',
        "chain": chain,
        "answer": {"limit": limits[t], "decision": decision},
        "chance": 1 / n_chains,
    }


# -- task 3: sequential state tracking ----------------------------------------

def gen_ledger(rng: random.Random, hops: int) -> dict:
    """Track one SKU's stock through `hops` ordered events, with events on other SKUs mixed in."""
    skus = _ids(rng, "SKU", rng.randint(3, 4))
    stock = {s: rng.randint(10, 60) for s in skus}
    opening = "; ".join(f"{s}: {q} units" for s, q in stock.items())
    target = rng.choice(skus)
    others = [s for s in skus if s != target]
    order = [target] * hops + [rng.choice(others) for _ in range(rng.randint(1, hops + 1))]
    rng.shuffle(order)
    lines, chain = [], []
    for i, sku in enumerate(order, 1):
        roll = rng.random()
        if roll < 0.15:
            stock[sku] = rng.randint(0, 60)
            text = f"Cycle count set {sku} to {stock[sku]} units."
        elif roll < 0.55 and stock[sku] > 0:
            n = rng.randint(1, min(20, stock[sku]))
            stock[sku] -= n
            text = f"Shipped {n} units of {sku}."
        else:
            n = rng.randint(1, 20)
            stock[sku] += n
            text = f"Received {n} units of {sku}."
        lines.append(f"{i}. {text}")
        if sku == target:
            chain.append({"hop": len(chain) + 1, "event": i, "to": stock[sku]})
    return {
        "context": f"Opening stock: {opening}.\nEvents, in order:\n" + "\n".join(lines),
        "question": f"How many units of {target} are in stock after all events?",
        "format": '{"stock": <number>}',
        "chain": chain,
        "answer": {"stock": stock[target]},
        "chance": 0.0,
    }


# -- task 4: lookup chains over single-token names ----------------------------
# Same structure as lookup_chain, but every entity is one common word, chosen so that it
# is a single token both mid-sentence (" Paris") and at the start of a JSON string ("Paris")
# for GPT-2-style BPE tokenizers (checked against SmolLM2). That puts the whole identity of
# each hop's answer on one target token, which is what per-loop supervision needs.
# `Encoder.chain_sharp` reports whether this holds for the tokenizer in use.

NAME_LAYERS = [
    ("employee", "Alice Bob David Frank Henry Jack Mary Nick Paul Rose Sam Adam Ben Chris Dan Fred George "
                 "James Mark Peter Tom Anna Edward Sarah Alex Andrew Anne Bill Carl Daniel Emily Jane Jeff "
                 "Jim John Kim Luke Maria Martin Matt Max Michael Pat Rob Scott Simon Tim Will"),
    ("team", "River Forest Ocean Lake Park Market Hill Field Hall Farm Camp Port Bay Road Gate Wall Mill "
             "Fort Church School College Library Bank Factory Store"),
    ("site", "Paris London Rome Boston Chicago Oxford Cambridge Lincoln Washington Jackson Madison Texas "
             "Florida Virginia"),
    ("unit", "Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March April May "
             "June July August September October December Spring Summer Winter"),
    ("zone", "North South East West Central Red Blue Green Yellow Brown Black White"),
    ("fund", "Apple Sugar Honey Water Fire Wind Rain Snow Ice Sun Moon Star Sky Cloud Light Hope Peace"),
]
NAME_POOLS = [words.split() for _, words in NAME_LAYERS]
NAME_FACTS = [
    "Employee {a} works in team {b}.",
    "Team {a} is based at site {b}.",
    "Site {a} reports to unit {b}.",
    "Unit {a} falls under zone {b}.",
    "Zone {a} is financed by fund {b}.",
]


def gen_name_chain(rng: random.Random, hops: int) -> dict:
    """Follow `hops` links through shuffled records whose entities are single words."""
    n_chains = rng.randint(3, 5)
    ents = [rng.sample(NAME_POOLS[layer], n_chains) for layer in range(hops + 1)]
    facts = [
        NAME_FACTS[layer].format(a=ents[layer][c], b=ents[layer + 1][c])
        for layer in range(hops)
        for c in range(n_chains)
    ]
    rng.shuffle(facts)
    t = rng.randrange(n_chains)
    name = NAME_LAYERS[hops][0]
    return {
        "context": "\n".join(facts),
        "question": f"Which {name} is employee {ents[0][t]} ultimately linked to?",
        "format": '{"answer": "<%s name>"}' % name,
        "chain": [{"hop": i + 1, "from": ents[i][t], "to": ents[i + 1][t]} for i in range(hops)],
        "chain_answers": [{"answer": ents[i + 1][t]} for i in range(hops)],
        "answer": {"answer": ents[hops][t]},
        "chance": 1 / n_chains,
    }


# name -> (generator, min hops, max hops)
TASKS: dict[str, tuple[Callable[[random.Random, int], dict], int, int]] = {
    "lookup_chain": (gen_lookup_chain, 1, 7),
    "rule_eval": (gen_rule_eval, 3, 6),
    "ledger": (gen_ledger, 1, 12),
    "name_chain": (gen_name_chain, 1, 5),
}
DEFAULT_TASKS = ("lookup_chain", "rule_eval", "ledger")


def generate(n: int, hops: tuple[int, int], seed: int, split: str, tasks=None, seen: Optional[set] = None) -> list[dict]:
    """`n` examples with hop counts uniform in `hops`. `seen` keeps splits disjoint."""
    tasks = list(tasks or DEFAULT_TASKS)
    rng = random.Random(f"{seed}/{split}")
    seen = set() if seen is None else seen
    out: list[dict] = []
    while len(out) < n:
        h = rng.randint(*hops)
        eligible = [t for t in tasks if TASKS[t][1] <= h <= TASKS[t][2]]
        if not eligible:
            raise ValueError(f"no task supports {h} hops")
        task = rng.choice(eligible)
        ex = TASKS[task][0](rng, h)
        key = (ex["context"], ex["question"])
        if key in seen:
            continue
        seen.add(key)
        out.append({"id": f"{split}-{len(out):06d}", "task": task, "hops": h, **ex})
    return out


def write_jsonl(path, rows) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# -- prompt, target, scoring --------------------------------------------------

def build_prompt(ex: dict) -> str:
    prompt = f"{INSTRUCTION}\n\n{ex['context']}\n\nQuestion: {ex['question']}"
    if ex.get("format"):
        prompt += f"\nRespond with only a JSON object of the form {ex['format']}."
    return prompt


def answer_text(answer: dict) -> str:
    return json.dumps(answer)


def extract_json(text: str) -> Optional[dict]:
    """First JSON object found in `text`, or None."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(text, start)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        start = text.find("{", start + 1)
    return None


def _norm(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            return float(text.replace(",", "").replace("$", ""))
        except ValueError:
            return text.casefold()
    return value


def score(text: str, answer: dict) -> dict:
    """Exact match on the parsed JSON object (same keys, normalised values)."""
    pred = extract_json(text)
    correct = pred is not None and {k: _norm(v) for k, v in pred.items()} == {
        k: _norm(v) for k, v in answer.items()
    }
    return {"correct": bool(correct), "valid_json": pred is not None}


class Encoder:
    """Turns examples into token ids. Uses the tokenizer's chat template when it has one."""

    def __init__(self, tokenizer, max_length: int = 2048):
        self.tok = tokenizer
        self.max_length = max_length
        self._cache: dict[str, tuple[list[int], list[int]]] = {}
        self._chain: dict[str, list[list[int]]] = {}

    def encode(self, ex: dict) -> tuple[list[int], list[int]]:
        """(prompt ids, target ids). The target ends with the end-of-answer token."""
        if ex["id"] not in self._cache:
            self._cache[ex["id"]] = self._render(ex, ex["answer"])
        return self._cache[ex["id"]]

    def _render(self, ex: dict, answer_obj: dict) -> tuple[list[int], list[int]]:
        tok, prompt, answer = self.tok, build_prompt(ex), answer_text(answer_obj)
        if getattr(tok, "chat_template", None):
            messages = [{"role": "user", "content": prompt}]
            prefix = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            full = tok.apply_chat_template(
                messages + [{"role": "assistant", "content": answer}], tokenize=False
            )
            suffix = full[len(prefix):].rstrip() if full.startswith(prefix) else answer + (tok.eos_token or "")
            p_ids = tok(prefix, add_special_tokens=False)["input_ids"]
            t_ids = tok(suffix, add_special_tokens=False)["input_ids"]
        else:
            bos = [tok.bos_token_id] if getattr(tok, "bos_token_id", None) is not None else []
            p_ids = bos + tok(prompt + "\nAnswer:", add_special_tokens=False)["input_ids"]
            t_ids = tok(" " + answer, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        return list(p_ids), list(t_ids)

    # -- per-loop (chain) supervision -----------------------------------------

    def chain_targets(self, ex: dict) -> list[list[int]]:
        """Target ids for the answer after each hop; the last entry is the final target."""
        if ex["id"] not in self._chain:
            final = self.encode(ex)[1]
            steps = ex.get("chain_answers") or [ex["answer"]]
            self._chain[ex["id"]] = [self._render(ex, a)[1] for a in steps[:-1]] + [final]
        return self._chain[ex["id"]]

    @staticmethod
    def _first_diff(a: list[int], b: list[int]) -> Optional[int]:
        return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)

    def chain_labels(self, ex: dict, num_exits: int) -> list[list[int]]:
        """Labels for exits 1..num_exits, aligned with the final target tokens (-100 = ignore).

        The input sequence is always the final answer (teacher forcing), so an intermediate
        hop can only be supervised where its answer first departs from the final one: at that
        token the inputs so far are identical. Exit r < num_exits gets that single token of
        hop r's answer. The last exit, and any exit at or past the end of the chain, gets the
        full final answer: once the chain is resolved, further loops must leave it alone.
        """
        targets = self.chain_targets(ex)
        final = targets[-1]
        out = []
        for r in range(1, num_exits + 1):
            labels = list(final)
            if r < num_exits and r < len(targets):
                d = self._first_diff(targets[r - 1], final)
                if d is not None:
                    labels = [-100] * len(final)
                    labels[d] = targets[r - 1][d]
            out.append(labels)
        return out

    def chain_sharp(self, ex: dict) -> bool:
        """True when every hop's answer differs from the final target in exactly one token,
        i.e. the single supervised token identifies the hop's entity completely."""
        targets = self.chain_targets(ex)
        final = targets[-1]
        return all(len(t) == len(final) and sum(x != y for x, y in zip(t, final)) <= 1 for t in targets)

    def chain_tokens(self, ex: dict) -> Optional[tuple[int, list[int]]]:
        """(index into the target, token of each hop's answer at that index) for chains of 2+ hops:
        the position where the entity is decided, and what each hop's entity looks like there."""
        targets = self.chain_targets(ex)
        if len(targets) < 2:
            return None
        d = self._first_diff(targets[0], targets[-1])
        if d is None or any(len(t) <= d for t in targets):
            return None
        return d, [t[d] for t in targets]

    def chain_batch(self, examples: list[dict], num_exits: int) -> dict[int, torch.Tensor]:
        """{exit: labels} with the same layout as `train_batch`."""
        rows = [self.encode(ex) for ex in examples]
        width = max(len(p) + len(t) for p, t in rows)
        out = {r: torch.full((len(rows), width), -100, dtype=torch.long) for r in range(1, num_exits + 1)}
        for i, (ex, (p, t)) in enumerate(zip(examples, rows)):
            for r, labels in enumerate(self.chain_labels(ex, num_exits), 1):
                out[r][i, len(p) : len(p) + len(t)] = torch.tensor(labels)
        return out

    def fits(self, ex: dict) -> bool:
        p, t = self.encode(ex)
        return len(p) + len(t) <= self.max_length

    def train_batch(self, examples: list[dict]) -> dict:
        """Right-padded prompt+target; labels are -100 everywhere except the target."""
        rows = [self.encode(ex) for ex in examples]
        width = max(len(p) + len(t) for p, t in rows)
        ids = torch.full((len(rows), width), self.tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        labels = torch.full_like(ids, -100)
        for i, (p, t) in enumerate(rows):
            n = len(p) + len(t)
            ids[i, :n] = torch.tensor(p + t)
            mask[i, :n] = 1
            labels[i, len(p):n] = torch.tensor(t)
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}

    def prompt_batch(self, examples: list[dict]) -> dict:
        """Left-padded prompts for generation."""
        rows = [self.encode(ex)[0] for ex in examples]
        width = max(len(p) for p in rows)
        ids = torch.full((len(rows), width), self.tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, p in enumerate(rows):
            ids[i, width - len(p):] = torch.tensor(p)
            mask[i, width - len(p):] = 1
        return {"input_ids": ids, "attention_mask": mask}


# -- CLI ----------------------------------------------------------------------

def _range(text: str) -> tuple[int, int]:
    lo, _, hi = text.partition("-")
    return int(lo), int(hi or lo)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Generate a synthetic A -> B -> C dataset.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=8000)
    ap.add_argument("--val", type=int, default=300)
    ap.add_argument("--test", type=int, default=600)
    ap.add_argument("--hops", type=_range, default=(2, 4), help="hop range for train/val/test, e.g. 2-4")
    ap.add_argument("--ood-hops", type=_range, default=None, help="hop range for test_ood, e.g. 5-6")
    ap.add_argument("--ood", type=int, default=300)
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS), help=f"comma-separated, from: {', '.join(TASKS)}")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    tasks = args.tasks.split(",")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seen: set = set()
    sizes = {}
    for split, n, hops in (
        ("test", args.test, args.hops),
        ("val", args.val, args.hops),
        ("train", args.train, args.hops),
        ("test_ood", args.ood if args.ood_hops else 0, args.ood_hops),
    ):
        if n:
            rows = generate(n, hops, args.seed, split, tasks, seen)
            write_jsonl(out / f"{split}.jsonl", rows)
            sizes[split] = {"n": n, "hops": list(hops)}
    (out / "dataset_card.json").write_text(
        json.dumps({"seed": args.seed, "tasks": tasks, "splits": sizes}, indent=2) + "\n"
    )
    print(f"wrote {sizes} to {out}")


if __name__ == "__main__":
    main()
