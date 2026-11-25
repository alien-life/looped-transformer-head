"""Linear probes on frozen backbone states: is what one lookup needs present, and at which depth?

    bash scripts/run_capped.sh python3 scripts/feature_probe.py

On the name_chain task (data/names), for each chosen backbone layer:

  query  at the position that decides the answer entity, can the QUESTION's employee be read?
  key    at a fact's object token ("... works in team River"), can that fact's SUBJECT be read?
  value  at a fact's object token, can the object itself be read?

A single attention lookup needs all three. Reports held-out accuracy of a linear classifier.
"""
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lth.data import NAME_POOLS, Encoder, build_prompt, read_jsonl  # noqa: E402
from lth.model import load_base  # noqa: E402

MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
LAYERS = [0, 8, 16, 24, 31, 32]  # indices into hidden_states; 32 = final, post-norm
N_EXAMPLES = 1800

base, tok = load_base(MODEL, "fp32")
enc = Encoder(tok)
examples = read_jsonl("data/names/train.jsonl")[:N_EXAMPLES]
names = {w: i for i, w in enumerate(NAME_POOLS[0])}
teams = {w: i for i, w in enumerate(NAME_POOLS[1])}
feats = {kind: {layer: [] for layer in LAYERS} for kind in ("query", "key", "value")}
labels = {kind: [] for kind in feats}


@torch.no_grad()
def collect(ex):
    p, t = enc.encode(ex)
    prefix = tok.apply_chat_template(
        [{"role": "user", "content": build_prompt(ex)}], tokenize=False, add_generation_prompt=True
    )
    encoded = tok(prefix, add_special_tokens=False, return_offsets_mapping=True)
    assert encoded["input_ids"] == p
    spans = encoded["offset_mapping"]

    def token_at(char):
        return max(i for i, (a, b) in enumerate(spans) if a <= char < b)

    d = 4  # SmolLM2 tokenises the answer as '{"', 'answer', '":', ' "', <entity>, ...
    assert tok.decode([t[d]]) == ex["answer"]["answer"]
    device = base.get_input_embeddings().weight.device
    states = base.base_model(input_ids=torch.tensor([p + t[:d]], device=device), output_hidden_states=True).hidden_states
    asked = re.search(r"employee (\w+)", ex["question"]).group(1)
    for layer in LAYERS:
        feats["query"][layer].append(states[layer][0, -1].float().cpu())
    labels["query"].append(names[asked])
    for m in re.finditer(r"Employee (\w+) works in team (\w+)\.", ex["context"]):
        pos = token_at(prefix.index(m.group(0)) + m.start(2) - m.start(0))
        assert tok.decode([p[pos]]).strip() == m.group(2)
        for layer in LAYERS:
            feats["key"][layer].append(states[layer][0, pos].float().cpu())
            feats["value"][layer].append(states[layer][0, pos].float().cpu())
        labels["key"].append(names[m.group(1)])
        labels["value"].append(teams[m.group(2)])


def linear_probe(x, y, classes):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, y = torch.stack(x).to(device), torch.tensor(y, device=device)
    perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(0)).to(device)
    train, test = perm[: int(0.8 * len(y))], perm[int(0.8 * len(y)):]
    x = (x - x[train].mean(0)) / (x[train].std(0) + 1e-6)
    w = torch.zeros(x.shape[1], classes, device=device, requires_grad=True)
    b = torch.zeros(classes, device=device, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=1e-2, weight_decay=1e-3)
    for _ in range(400):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(x[train] @ w + b, y[train]).backward()
        opt.step()
    return ((x[test] @ w + b).argmax(-1) == y[test]).float().mean().item(), len(y)


for ex in examples:
    collect(ex)
print(f"{'probe':6s} {'classes':>7s} {'n':>6s}  " + "  ".join(f"layer {layer:>2d}" for layer in LAYERS))
for kind, classes in (("query", len(names)), ("key", len(names)), ("value", len(teams))):
    row = [linear_probe(feats[kind][layer], labels[kind], classes) for layer in LAYERS]
    print(f"{kind:6s} {classes:7d} {row[0][1]:6d}  " + "  ".join(f"{acc:8.3f}" for acc, _ in row))
