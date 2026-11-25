# Looped Transformer Head (LTH)

A trainable, weight-shared transformer block (`fn_looped_attn`) that runs R times on the final
hidden states of a frozen causal LM before the frozen `lm_head`. This repo implements the spec's
module, the 5x / 10x / 50x training tiers, an A -> B -> C dataset format with a synthetic
generator, and a depth-sweep evaluation harness.

```
lth/head.py             the looped block: first-pass KV reuse, step embeddings, zero-init gate,
                        truncated BPTT, gradient checkpointing, 1x KV cache for decoding
lth/model.py            frozen backbone + head: loss, cached greedy decoding, head checkpoints
lth/data.py             A -> B -> C schema, three synthetic task families, scoring, tokenisation
lth/train.py            tiers 1x (control) / 2x / 3x / 5x / 10x / 50x, backbone hidden-state cache
lth/evaluate.py         depth sweep, latency, general-text regression, acceptance report
lth/probe.py            decodes every loop: which hop's answer does the head hold?
lth/general_prompts.py  built-in passages for the regression check
lth/tiny.py             offline random model + char tokenizer for smoke tests
scripts/demo_small.sh   the small end-to-end run reported below
scripts/lookup_ladder.sh  1x vs 2x vs 3x on lookup-only data (second experiment below)
scripts/chain_ladder.sh   per-loop supervision vs final-answer-only (third experiment below)
scripts/feature_probe.py  linear probes: which backbone layer holds what a lookup needs
scripts/run_capped.sh     runs a job under a memory cap, one at a time (use it for every run)
tests/                  49 tests; `python3 -m pytest`
```

## Usage

```bash
# 1. data: train/val/test at 2-4 hops, plus a harder out-of-distribution test set
python3 -m lth.data --out data/abc --train 8000 --val 200 --test 600 --hops 2-4 --ood-hops 5-6

# 2. train: the control and the tiers (each run saves only the head)
python3 -m lth.train --model <hf-id> --data data/abc --out runs/1x  --tier 1x
python3 -m lth.train --model <hf-id> --data data/abc --out runs/5x  --tier 5x
python3 -m lth.train --model <hf-id> --data data/abc --out runs/10x --tier 10x --init-from runs/5x/head.pt
python3 -m lth.train --model <hf-id> --data data/abc --out runs/50x --tier 50x --init-from runs/10x/head.pt

# 3. evaluate: one sweep over every head, on the same held-out examples
python3 -m lth.evaluate --model <hf-id> --data test=data/abc/test.jsonl --data ood=data/abc/test_ood.jsonl \
    --head 1x=runs/1x/head.pt --head 5x=runs/5x/head.pt --head 10x=runs/10x/head.pt --head 50x=runs/50x/head.pt \
    --control 1x --out reports/sweep --strict
```

`--model tiny-random` runs every command offline in seconds (plumbing check only).

`--cache-hidden DIR` on `lth.train` stores the frozen backbone's hidden states for the training set
once and trains the head from them, skipping the backbone forward. On the small model below that
took training from about 3 to 21 steps per second. The arrays live on disk and are memory-mapped,
so they cost disk, not RAM: 2 bytes x tokens x backbone width, 3.4 GB for 8000 examples here and
about 20 GB for the same data at width 5376. They are rebuilt in a couple of minutes; delete them
when a set of runs is done.

### Run every job through `scripts/run_capped.sh`

```bash
bash scripts/run_capped.sh bash scripts/chain_ladder.sh
LTH_MEM_MAX=6G bash scripts/run_capped.sh python3 -m lth.train ...
```

It puts the command in its own cgroup with a memory ceiling (default 8 GB, no swap), holds a lock
so a second job cannot start while one runs, and refuses to start when less than 10 GB of RAM is
available or swap is already in use. A job that outgrows the ceiling is killed on its own.

This exists because two overlapping training runs froze the development machine on 2026-11-23.
Each held a multi-gigabyte cache in RAM (the cache was not memory-mapped then), on a machine
whose 2 GB of swap was already full.

| Tier | Depth | Backprop | Extras |
|---|---|---|---|
| `1x` | R = 1 | full | control arm: same head, no looping |
| `2x`, `3x` | R = 2, 3 | full | smallest looped arms |
| `5x` | R = 5 | full BPTT + gradient checkpointing | |
| `10x` | R = 10 | full BPTT + gradient checkpointing | |
| `50x` | R ~ Uniform(1, 50) per step | truncated, window 8 | deep supervision at loops 5, 10, 25, 50 and the sampled R |

## Per-loop (chain) supervision

By default only the final answer is supervised, and the head has to discover the intermediate
steps on its own. With `--chain-supervision`, loop r is also trained to hold the answer after hop
r, read from the example's `chain_answers`:

```bash
python3 -m lth.data --out data/names --train 16000 --val 300 --test 600 --hops 1-3 --tasks name_chain
python3 -m lth.train --model <hf-id> --data data/names --out runs/3x-chain --tier 3x --chain-supervision
python3 -m lth.probe --model <hf-id> --data data/names/test.jsonl --head runs/3x-chain/head.pt --out reports/probe
```

How the labels are built (`Encoder.chain_labels`):

- The input is always the prompt plus the final answer, so an intermediate hop can only be
  supervised where its answer first departs from the final one. Loop r < R gets that one token of
  hop r's answer; everything else is ignored at that loop.
- The last loop, and any loop at or past the end of the chain, gets the whole final answer. A
  2-hop question under a 3-loop head must hold its answer through loop 3.
- A head with fewer loops than hops still has to produce the final answer on its last loop.
- Nothing changes at inference: only the last loop is decoded, and no intermediate text is emitted.

This needs each hop's answer to sit on one token, which is what the `name_chain` task is for. The
trainer prints the share of examples where that holds for the tokenizer in use. Works with
fixed-depth tiers (1x to 10x) and with `--cache-hidden`.

`lth.probe` decodes every loop through the frozen `lm_head` and reports, per hop count and loop,
how often the loop is on target and which hop's entity it is holding. Run it on a head trained
without chain supervision to see whether intermediate entities appear unprompted.

## Head options added while debugging

| Flag | What it does | Why |
|---|---|---|
| `--tap-layer N` | the block reads backbone layer N's output instead of the final hidden state; its output is still added to the final state before `lm_head` | linear probes show the final layer has lost much of what a lookup needs (third experiment) |
| `--standardize` | per-dimension standardisation of the block's input from training-set statistics (implied by `--tap-layer`) | 5 of 960 dimensions hold 52% of the final layer's energy, 99% at layer 16 |
| `--gate matrix` | zero-initialises the output projection instead of using one learned scalar for the whole output (now default) | the scalar gate collapsed to about 0.001 in training and dominated global `clip_grad_norm_`, starving block weights |
| `--norm-out` | `RMSNorm` on `(x - z0)` before `proj_out` (default on; `--no-norm-out` to disable) | keeps emitted delta magnitude invariant across loop depths `R`, preventing drift beyond trained depth |
| `--qk-norm` | per-head `RMSNorm` on Q and K before RoPE (default on; `--no-qk-norm` to disable) | prevents attention logit explosion (`scores > 100`, entropy `0.00`) that froze heads on random tokens; essential for induction-head formation |
| `--no-rope` | disables RoPE inside the head for pure content-addressable attention (optional; `--rope` is default) | lets ablations test content-only vs positional Q/K matching |
| `--schedule wsd` | holds LR constant for 75% of post-warmup steps before cosine decay (default; `--schedule cosine` for immediate decay) | avoids decaying LR before induction-head phase transitions snap |

All head options keep exact base-model parity at initialisation, remain backward-compatible with earlier
checkpoints, and are covered by the cached-decoding tests. `--tap-layer` departs from the spec, which
places the block strictly after the final layer.

## Dataset format (A -> B -> C)

One JSON object per line. Bring your own data by writing the same fields.

| Field | Role |
|---|---|
| `context`, `question` | **A**: what the model reads |
| `format` | the JSON shape the answer must take; shown to the model so the frozen baseline gets a fair attempt |
| `chain` | **B**: gold intermediate states. Never shown to the model; kept for diagnostics |
| `chain_answers` | optional **B** in answer form: the answer after each hop, last one equal to `answer`. Used only by `--chain-supervision` and `lth.probe` |
| `answer` | **C**: gold JSON object. Scored by exact match on the parsed object |
| `hops` | number of latent steps; every metric is broken down by it |
| `task`, `id`, `chance` | task family, unique id, accuracy of guessing among candidates in the context |

The generator has four families, each with a controllable hop count and distractors. The first
three are the default mix; `name_chain` is selected with `--tasks name_chain`.

| Task | What B is | Hops |
|---|---|---|
| `lookup_chain` | follow links through shuffled records (account -> rep -> team -> region ...) | 1-7 |
| `rule_eval` | resolve an order's limit through linked records, then apply an approval rule | 3-6 |
| `ledger` | track one SKU's stock through ordered events mixed with events on other SKUs | 1-12 |
| `name_chain` | `lookup_chain` with single-word entities (employee -> team -> site -> unit ...), each one token for GPT-2-style tokenizers; carries `chain_answers` | 1-5 |

Entity ids are random per example, so answers cannot be memorised, and splits are deduplicated
against each other. The tests re-solve generated examples with independent solvers that read only
the rendered text.

## What the evaluation harness decides

A trained head beating the frozen model does not show that *looping* helps: any trained adapter
would. So the comparison has three parts, all on the same examples with paired tests:

| Arm | Question it answers |
|---|---|
| R = 0 | frozen baseline (bit-identical to the base model) |
| `1x` head at R = 1 | how much does a trained head with the same parameters help without looping? |
| `5x` / `10x` / `50x` heads | does iterating add accuracy on top of that? |

Acceptance criteria per head (`--strict` exits 1 when a required one fails):

| Criterion | Test |
|---|---|
| beats frozen baseline | accuracy at the trained depth > R = 0, exact McNemar p < 0.05 |
| looping beats single-pass control | accuracy at the trained depth > control at R = 1, p < 0.05 |
| monotonic up to trained depth | no drop larger than `--mono-tol` along 0 -> 1 -> 5 -> ...; required only for heads trained with sampled depths (50x), informational for single-depth heads |
| stable beyond trained depth | informational: does over-iterating hurt? |
| latency within SLA | TTFT / ITL at the trained depth, when `--sla-ttft-ms` / `--sla-itl-ms` are given |
| no regression on general text | perplexity ratio vs R = 0 at most 1 + `--regression-tol` |

Outputs: `report.md`, `report.json`, and per-example predictions for every (dataset, head, depth).

## Where this differs from the spec

The reference code in section 4 does not run as written, and a few statements in the spec do not
hold. Verified by running the spec's module unless marked otherwise.

| Spec | Here | Why |
|---|---|---|
| `nn.MultiheadAttention(..., is_causal=True)` with no mask | explicit causal + padding mask | as written it raises `Need attn_mask if specifying the is_causal hint`; with the hint removed, positions attend to future tokens (label leakage under teacher forcing) |
| keys/values passed as `h_in` every pass | separate Q / K / V projections; K/V projected once | `MultiheadAttention` re-projects K/V on every pass, so nothing is reused; explicit projections also make the 1x decode cache possible |
| no positional signal in the head | rotary embeddings on Q and K | the head's attention has no positional encoding of its own otherwise |
| `h_base + alpha * h` | `h_base + alpha * (h - h_base)` | the spec's form also rescales the backbone state once alpha is non-zero |
| plain SiLU MLP | SwiGLU | the architecture table asks for SwiGLU |
| truncated BPTT by detaching early loops | pass-1 K/V always keep their graph | with first-pass KV reuse, detaching pass 1 leaves the K/V projections with no gradient at all |
| full-vocabulary logits everywhere | logits only at supervised positions | a B x T x 262k logits tensor is avoidable |
| `self.base.model`, `config.hidden_size` | `base.base_model`, `lm_head` width, plus a parity check at start-up | the spec's attribute names are not shared by every architecture (read from the code, not run on a 27B model) |
| head left on the CPU | head placed on the `lm_head` device | with `device_map="auto"` the spec's wrapper mixes devices (read from the code) |
| checkpoints are "tens of megabytes" | `--head-dim` bottleneck option | at hidden size 5376 a full-width 1-layer head is 347M parameters, 1.3 GB in fp32; `--head-dim 1024` gives 23.6M parameters, 45 MB in bf16 |
| "zero train-inference mismatch" | tested | `test_incremental_cache_matches_parallel` and `test_cached_generation_matches_uncached` |

Not implemented: LoRA on `lm_head` (it stays frozen), FLOP counting (latency is measured instead),
and automatic routing of non-domain queries. Calling with `num_loops=0` is exactly the base model,
so routing is a caller decision.

## Running on a 27B backbone

Nothing here has been run on a 27B model: this machine has an 8 GB GPU and 35 GB of free disk.
What the code needs for that:

- about 54 GB of backbone weights in bf16, or `--quant 4bit` (needs `bitsandbytes`, untested here);
- `transformers>=4.50` for Gemma 3 (4.40.2 is installed here);
- `--dtype bf16 --amp bf16`, and probably `--head-dim 1024`: a full-width pass costs roughly 0.6 of
  one backbone layer per token, so R = 50 at full width adds on the order of half a backbone
  forward to every generated token (an estimate from layer sizes, not a measurement).

## Small-scale results

`bash scripts/demo_small.sh`: SmolLM2-360M-Instruct, full-width 1-layer head (11.1M parameters),
four arms trained from scratch on the same budget (1200 steps, batch 8, one seed each), evaluated on
300 test examples (2-4 hops) and 300 out-of-distribution examples (5-6 hops). Full tables are in
`runs/demo/report/report.md`.

| Arm at its trained depth | Test accuracy | p vs frozen baseline (0.140) | p vs 1x control | General-text perplexity |
|---|---|---|---|---|
| 1x, R = 1 | 0.170 | 0.22 | | x1.02 |
| 5x, R = 5 | 0.193 | 0.04 | 0.32 | x2.97 |
| 10x, R = 10 | 0.173 | 0.11 | 1 | x2.47 |
| 50x, R = 50 | 0.190 | 0.024 | 0.34 | x4.75 |

All four heads fail acceptance.

- **No evidence that looping helps at this scale.** No looped arm beats the single-pass control, and
  the 50x head is flat across depth: 0.193 at R = 1, 0.190 at R = 50. Its single pass is also the
  best result on the out-of-distribution set (0.167 vs 0.090 baseline), so what gain there is comes
  from the trained head, not from iterating.
- **This is a weak test.** The heads barely learned the task: training loss plateaued at 0.26-0.30,
  accuracy stayed within a few points of the frozen baseline, and the gate ended small (mean
  |alpha| 0.040, 0.009, 0.003, 0.0008 for 1x to 50x). One seed per arm, and the p-values are not
  adjusted for the number of comparisons. It does not show that looping cannot help, only that
  this budget on a 360M backbone gives no sign of it.
- **Single-depth heads break when run deeper than trained.** Accuracy and JSON validity fall to
  zero (5x head: 0.193 at R = 5, 0.017 at R = 10, 0 at R = 25). The 50x head, trained with sampled
  depths, stays valid from R = 1 to 50. A depth dial needs sampled-depth training.
- **Every head degrades general text at its trained depth**, from x1.02 perplexity (1x) to x4.75
  (50x). Non-domain requests have to be served with `num_loops=0`.
- **Latency**: 28.2 ms per token at R = 0, about +0.6 ms per loop, x2.0-2.1 at R = 50.
  Time-to-first-token moves the same way (28.7 ms to about 58 ms), because the prompt needs only
  one full pass and the remaining loops run on the last position.

What would make the test stronger: more optimisation steps and several seeds per arm, a LoRA arm
with the same parameter count, and a backbone that can already solve some of the task.

### Second experiment: 1x vs 2x vs 3x on lookups only

`LAYERS=2 bash scripts/lookup_ladder.sh` (and the same with a 1-layer block). Lookup-chain questions
only, 1-3 hops, 8000 training examples, 600 test examples, one seed per arm, heads trained from
cached backbone states. If a loop buys a hop, the 2x head should win on 2-hop questions and the 3x
head on 3-hop questions. Guessing among the candidates in the context scores about 0.25.

| Head | Block | 1 hop | 2 hops | 3 hops | Overall |
|---|---|---|---|---|---|
| frozen baseline | | 0.641 | 0.282 | 0.000 | 0.322 |
| 1x | 1 layer | 0.699 | 0.306 | 0.253 | 0.427 |
| 2x | 1 layer | 0.703 | 0.316 | 0.319 | 0.452 |
| 1x | 2 layers | 0.766 | 0.301 | 0.253 | 0.448 |
| 2x | 2 layers | 0.794 | 0.311 | 0.308 | 0.478 |
| 3x | 2 layers | 0.732 | 0.316 | 0.264 | 0.445 |

- **No head chains lookups.** Every arm is at chance on 2-hop and 3-hop questions, with one or two
  layers and with one, two or three loops. The heads learn the output format (3-hop goes from 0 to
  chance) and a somewhat better single hop.
- **Looping still shows no benefit.** 2x vs 1x with two layers is 0.478 vs 0.448 (p = 0.14); 3x vs
  1x is 0.445 vs 0.448.
- **The heads memorise.** Training loss keeps falling (0.083-0.085 for 1x and 2x) while validation
  loss stops improving after 2000-3000 steps and then rises (0.111 to 0.116-0.121). More steps on
  the same 8000 examples will not help.
- A hypothesis that did not hold: that a single layer fails because first-pass key/value reuse
  fixes the keys before any attention runs, so a second layer would let the head bind subject to
  object inside a fact. The 2-layer block chains no better.

Full tables: `runs/lookup/report-L2/report.md` and `runs/lookup/report-L1-s0/report.md`.

### Third experiment: per-loop (chain) supervision

`bash scripts/chain_ladder.sh` with `LAYERS=1`, `LAYERS=2` and `TAP=8`. `name_chain` questions,
1-3 hops, 16000 training examples, 600 test examples, 3000 steps at batch 16, one seed per arm.
Guessing among the candidates scores about 0.25.

| Head reads | Block | 1x plain | 3x plain | 2x chain | 3x chain |
|---|---|---|---|---|---|
| final layer | 1 layer | 0.377 | 0.353 | 0.347 | 0.372 |
| final layer | 2 layers | 0.355 | 0.353 | 0.355 | 0.360 |
| layer 8 | 1 layer | 0.350 | 0.363 | 0.347 | 0.343 |

Frozen baseline: 0.140. By hop count every head is at about 0.59 / 0.30 / 0.20.

- **Per-loop supervision did not raise accuracy** in any of the three configurations.
- **The loops follow the schedule but not the content.** The probe on the chain-supervised 3x head
  shows loop 1 always emitting a hop-1-type entity, loop 2 a hop-2-type entity, and so on, but the
  right one only 36-47% of the time at loop 1 and at chance from loop 2.
- **What the heads learned is the answer's type.** The frozen backbone, restricted to candidates
  of the right type, scores 0.476 / 0.320 / 0.223 by hop count with no head at all. The heads match
  that on 2 and 3 hops and add about 11 points on 1 hop. No attention lookup was learned.
- **The final layer is a poor place to read from.** Linear probes (`scripts/feature_probe.py`,
  held-out accuracy, chance about 0.02):

  | Readable from the state | layer 8 | layer 16 | layer 24 | final |
  |---|---|---|---|---|
  | the question's entity, at the answer position | 0.953 | 0.956 | 0.725 | 0.494 |
  | a fact's subject, at its object token | 0.745 | 0.512 | 0.509 | 0.424 |
  | the object token itself | 1.000 | 1.000 | 1.000 | 1.000 |

  Reading layer 8 did not help at this budget, so information is not the only obstacle.
- **Two training defects surfaced** in a longer run (1-hop questions only, learning rate 1e-3, 7500
  steps): gradients became NaN under fp16 autocast, after which every step was silently skipped,
  and the scalar gate sat near 0.001. `--gate matrix` and `--amp none` address these.

Not yet known: whether a head can learn even one lookup with `--tap-layer 8 --gate matrix
--amp none` and the larger budget. That run was in progress when the machine froze and has not
been repeated. It is the next thing to try:

```bash
python3 -m lth.data --out data/names1 --train 16000 --val 300 --test 600 --hops 1-1 --tasks name_chain --seed 3
bash scripts/run_capped.sh python3 -m lth.train --model HuggingFaceTB/SmolLM2-360M-Instruct --dtype fp32 \
    --amp none --data data/names1 --out runs/names1/1x-L1-tap8-matrix --tier 1x --tap-layer 8 --gate matrix \
    --steps 7500 --batch-size 32 --lr 1e-3 --warmup 300 --eval-every 1500 --eval-examples 300 \
    --cache-hidden runs/names1/hidden_cache-tap8
```

A validation accuracy well above 0.56 would be the first sign of a learned lookup.

Full tables: `runs/names/report-*/report.md` and `runs/names/probe-*/probe.md`.

### Fourth experiment: optimisation fixes, 1-hop phase transition, and first looped `PASS`

Auditing the earlier runs found no sabotage, but isolated four coupled optimisation and attention bottlenecks:

1. **Scalar `alpha` dominated global gradient clipping:** `dL/d(alpha)` summed over all supervised tokens and 960 dims (`grad_norm` 15–1780) while block gradients scaled by `alpha` (`~0.001–0.007`), so joint `clip_grad_norm_` shrank Q/K/V/FFN updates by 15x–1780x. Defaulting to `--gate matrix`, separating `alpha` clipping, and excluding 1D norms from weight decay removed the bottleneck.
2. **Attention logit saturation:** without QK-Norm, `scores` hit `115` and every head collapsed to entropy `0.00` on random tokens. In a compact-vocab ablation on `data/names1`, a 2-layer head stayed at chance (`0.237`) without `--qk-norm` and snapped to `0.967` with `--qk-norm`.
3. **Unnormalised residual drift across loops:** adding `--norm-out` (`RMSNorm` on `x - z0` before `proj_out`) kept single-pass heads stable when evaluated beyond trained depth (`1x-plain` on `data/names`: `0.363 -> 0.357 -> 0.360` at `R=1,2,3`, vs `0.350 -> 0.283 -> 0.263` previously).
4. **Phase transition timing and capacity:** narrowing to `--head-dim 256 --num-heads 8 --num-layers 2` (2.07M params), running in fp32 (`--amp none`), using `--weight-decay 0.05`, and holding LR with `--schedule wsd` allowed a 1-hop head on `data/names1` (`runs/names_curr/pretrain-1hop-wd05`) to cross the induction-head phase transition between steps 3000 and 5000, reaching **0.967 validation accuracy** (`val_loss = 0.0172`).

**Matched curriculum comparison on 1–3 hop `data/names` (`runs/names_curr_v2`, `n = 600` test):** both `1x-plain` (`R=1` control) and `3x-chain` (`R=3`, chain-supervised) were initialised from that same 96.7% 1-hop checkpoint (`--init-from`) and trained identically for 3500 steps (`batch_size=32 x grad_accum=2`, `lr=3e-4`, `weight_decay=0.05`, `--schedule wsd`).

| Head | R | Accuracy | vs R=0 | 1 hop | 2 hops | 3 hops | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| frozen baseline | 0 | 0.140 | | 0.382 | 0.049 | 0.005 | |
| `1x-plain` (control) | 1* | 0.485 | +0.345 (`p=3.7e-56`) | 0.969 | 0.335 | 0.184 | PASS |
| `3x-chain` | 1 | 0.308 | +0.168 (`p=2.5e-21`) | 0.969 | 0.000 | 0.000 | |
| `3x-chain` | 2 | 0.447 | +0.307 (`p=1.5e-48`) | 0.974 | 0.404 | 0.000 | |
| `3x-chain` | 3* | **0.533** | +0.393 (`p=1.3e-64`) | 0.974 | **0.424** | **0.233** | **PASS** |

- **`looping beats single-pass control`: PASS** (`3x-chain` at `R=3`: `0.533` vs `1x-plain` at `R=1`: `0.485`, exact McNemar `p = 0.007`).
- **Per-loop probe (`runs/names_curr_v2/probe/probe.md`):** on 2-hop and 3-hop questions, `3x-chain` holds hop 1 at **0.916 / 0.908** after loop 1, then begins resolving hop 2 at loop 2 (`0.404` / `0.374`) and hop 3 at loop 3 (`0.233`). Validation loss was still dropping sharply (`0.2450 -> 0.2333 -> 0.2186`) when LR decayed at step 3000–3500, indicating hop 2 and hop 3 were mid-transition.

Full tables: `runs/names_curr_v2/report/report.md` and `runs/names_curr_v2/probe/probe.md`.
