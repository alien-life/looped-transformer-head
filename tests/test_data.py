"""The generators are checked against independent solvers that read only the rendered text."""
import random
import re

import pytest
import torch

from lth import data
from lth.tiny import CharTokenizer


def solve_lookup(ex):
    links = dict(re.findall(r"(\w+-\d+) (?:is|reports|operates|falls)[\w ]*? (\w+-\d+)\.", ex["context"]))
    node = re.search(r"account (\w+-\d+)", ex["question"]).group(1)
    for _ in range(ex["hops"]):
        node = links[node]
    assert node not in links
    return {"answer": node}


def solve_rule(ex):
    ctx = ex["context"]
    order = re.search(r"order (ORD-\d+)", ex["question"]).group(1)
    amount, node = re.search(rf"Order {order} for \$([\d,]+) was placed by customer (\S+)\.", ctx).groups()
    links = dict(re.findall(r"(\w+-\d+) (?:belongs|is served|follows)[\w ]*? (\w+-\d+)\.", ctx))
    limits = {k: int(v.replace(",", "")) for k, v in re.findall(r"(\w+-\d+) carries an order limit of \$([\d,]+)\.", ctx)}
    steps = 2  # order -> customer, then the comparison
    while node not in limits:
        node, steps = links[node], steps + 1
    steps += 1
    assert steps == ex["hops"]
    limit = limits[node]
    return {"limit": limit, "decision": "approved" if int(amount.replace(",", "")) <= limit else "escalated"}


def solve_ledger(ex):
    sku = re.search(r"units of (SKU-\d+) are", ex["question"]).group(1)
    stock = int(re.search(rf"{sku}: (\d+) units", ex["context"]).group(1))
    touched = 0
    for line in ex["context"].splitlines():
        if sku not in line or not re.match(r"\d+\. ", line):
            continue
        touched += 1
        n = int(re.search(r"(\d+) units", line).group(1))
        if "Received" in line:
            stock += n
        elif "Shipped" in line:
            stock -= n
        else:
            stock = n
        assert stock >= 0
    assert touched == ex["hops"]
    return {"stock": stock}


def solve_name(ex):
    links = dict(re.findall(r"^\w+ (\w+) (?:works in|is based at|reports to|falls under|is financed by) \w+ (\w+)\.$",
                            ex["context"], flags=re.M))
    node = re.search(r"employee (\w+)", ex["question"]).group(1)
    seen = []
    for _ in range(ex["hops"]):
        node = links[node]
        seen.append({"answer": node})
    assert node not in links
    assert seen == ex["chain_answers"]
    return {"answer": node}


SOLVERS = {"lookup_chain": solve_lookup, "rule_eval": solve_rule, "ledger": solve_ledger, "name_chain": solve_name}


@pytest.mark.parametrize("task", list(data.TASKS))
def test_generators_agree_with_independent_solver(task):
    gen, lo, hi = data.TASKS[task]
    rng = random.Random(0)
    for _ in range(300):
        hops = rng.randint(lo, hi)
        ex = gen(rng, hops)
        assert SOLVERS[task]({**ex, "hops": hops}) == ex["answer"]
        assert len(ex["chain"]) == hops
        assert data.score(data.answer_text(ex["answer"]), ex["answer"])["correct"]


def test_generation_is_deterministic_and_splits_are_disjoint():
    a = data.generate(50, (2, 4), 0, "train")
    assert a == data.generate(50, (2, 4), 0, "train")
    assert a != data.generate(50, (2, 4), 1, "train")
    seen = set()
    test = data.generate(200, (2, 4), 0, "test", seen=seen)
    train = data.generate(200, (2, 4), 0, "train", seen=seen)
    assert not {(e["context"], e["question"]) for e in test} & {(e["context"], e["question"]) for e in train}
    assert {e["hops"] for e in train} == {2, 3, 4}
    assert {e["task"] for e in train} == set(data.DEFAULT_TASKS)


def test_scoring():
    gold = {"limit": 12000, "decision": "approved"}
    ok = lambda text: data.score(text, gold)
    assert ok('{"limit": 12000, "decision": "approved"}') == {"correct": True, "valid_json": True}
    assert ok('Sure! {"limit": "$12,000", "decision": "Approved"} hope that helps')["correct"]
    assert ok('{"limit": 12500, "decision": "approved"}') == {"correct": False, "valid_json": True}
    assert not ok('{"limit": 12000}')["correct"]
    assert not ok('{"limit": 12000, "decision": "approved", "note": "x"}')["correct"]
    assert ok("the limit is 12000") == {"correct": False, "valid_json": False}
    assert ok('{"limit": 12000, "decision": ')["valid_json"] is False


def test_batches():
    tok = CharTokenizer()
    enc = data.Encoder(tok)
    exs = data.generate(4, (2, 3), 0, "train")
    batch = enc.train_batch(exs)
    for i, ex in enumerate(exs):
        p, t = enc.encode(ex)
        assert t[-1] == tok.eos_token_id
        assert tok.decode(t).strip() == data.answer_text(ex["answer"])
        n = len(p) + len(t)
        assert batch["attention_mask"][i].sum() == n
        assert (batch["labels"][i, : len(p)] == -100).all() and (batch["labels"][i, n:] == -100).all()
        assert batch["labels"][i, len(p):n].tolist() == t
    prompts = enc.prompt_batch(exs)
    for i, ex in enumerate(exs):
        p = enc.encode(ex)[0]
        assert prompts["input_ids"][i, -len(p):].tolist() == p
        assert prompts["attention_mask"][i].sum() == len(p)
    assert torch.equal(prompts["attention_mask"][:, -1], torch.ones(4, dtype=torch.long))


def test_name_pools_are_disjoint():
    words = [w for pool in data.NAME_POOLS for w in pool]
    assert len(words) == len(set(words))
    assert all(len(pool) >= 5 for pool in data.NAME_POOLS)  # up to 5 parallel chains


def test_chain_labels():
    tok = CharTokenizer()
    enc = data.Encoder(tok)
    ex = {"id": "x", "context": "c", "question": "q", "answer": {"answer": "June"},
          "chain_answers": [{"answer": "River"}, {"answer": "Paris"}, {"answer": "June"}]}
    final = enc.encode(ex)[1]
    start = tok.decode(final).index("June")  # character tokenizer: index == token position
    labels = enc.chain_labels(ex, 4)
    # loops 1 and 2: only the first token of that hop's answer, at the place it departs from the final one
    for labs, word in zip(labels[:2], ["River", "Paris"]):
        assert [i for i, v in enumerate(labs) if v != -100] == [start]
        assert tok.decode([labs[start]]) == word[0]
    # loop 3 (chain finished) and loop 4 (last loop): the whole final answer
    assert labels[2] == final and labels[3] == final
    # a head with fewer loops than hops: its last loop must still produce the final answer
    assert enc.chain_labels(ex, 2)[1] == final and enc.chain_labels(ex, 1) == [final]
    assert enc.chain_tokens(ex) == (start, [tok.encode(w[0])[0] for w in ("River", "Paris", "June")])
    assert not enc.chain_sharp(ex)  # with a character tokenizer one token is not the whole entity
    assert enc.chain_sharp({**ex, "id": "y", "chain_answers": [{"answer": "Jane"}, {"answer": "June"}]})

    plain = {"id": "z", "context": "c", "question": "q", "answer": {"answer": "June"}}
    assert enc.chain_labels(plain, 3) == [enc.encode(plain)[1]] * 3 and enc.chain_tokens(plain) is None

    batch, per_exit = enc.train_batch([ex, plain]), enc.chain_batch([ex, plain], 3)
    assert set(per_exit) == {1, 2, 3}
    assert torch.equal(per_exit[3], batch["labels"])
    assert (per_exit[1][0] != -100).sum() == 1 and torch.equal(per_exit[1][1], batch["labels"][1])
