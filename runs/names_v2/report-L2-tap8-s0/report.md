# Looped-head depth sweep

Base model: `HuggingFaceTB/SmolLM2-360M-Instruct`

## Accuracy: test (n=600)

| head | R | accuracy | 95% CI | vs R=0 | p | valid JSON | 1 hops | 2 hops | 3 hops |
|---|---|---|---|---|---|---|---|---|---|
| frozen baseline | 0 | 0.140 | 0.115-0.170 |  |  | 0.675 | 0.382 | 0.049 | 0.005 |
| 1x-plain | 1* | 0.363 | 0.326-0.403 | +0.223 | 1.3e-27 | 1.000 | 0.576 | 0.320 | 0.209 |
| 1x-plain | 2† | 0.357 | 0.319-0.396 | +0.217 | 3.3e-26 | 1.000 | 0.565 | 0.315 | 0.204 |
| 1x-plain | 3† | 0.360 | 0.323-0.399 | +0.220 | 1.1e-26 | 0.998 | 0.571 | 0.310 | 0.214 |
| 3x-chain | 1 | 0.177 | 0.148-0.209 | +0.037 | 0.028 | 1.000 | 0.555 | 0.000 | 0.000 |
| 3x-chain | 2 | 0.272 | 0.238-0.309 | +0.132 | 6.7e-12 | 1.000 | 0.550 | 0.286 | 0.000 |
| 3x-chain | 3* | 0.358 | 0.321-0.397 | +0.218 | 2e-23 | 1.000 | 0.576 | 0.315 | 0.199 |

`*` trained depth, `†` deeper than the head was trained. p: exact McNemar vs R=0.

## Acceptance

| head | criterion | result | detail |
|---|---|---|---|
| 1x-plain | beats frozen baseline | PASS | R=1: 0.363 vs R=0: 0.140 (gain +0.223, p=1.3e-27) |
| 1x-plain | monotonic up to trained depth | PASS | R0:0.140 -> R1:0.363 |
| 1x-plain | stable beyond trained depth | PASS (info) | worst at R=2: 0.357 vs R=1: 0.363 |
| 1x-plain | **verdict** | **PASS** | required criteria only |
| 3x-chain | beats frozen baseline | PASS | R=3: 0.358 vs R=0: 0.140 (gain +0.218, p=2e-23) |
| 3x-chain | monotonic up to trained depth | PASS (info) | R0:0.140 -> R1:0.177 -> R2:0.272 -> R3:0.358 | head was trained at a single depth; shallower depths are off-distribution |
| 3x-chain | looping beats single-pass control | FAIL | R=3: 0.358 vs '1x-plain' at R=1: 0.363 (p=0.84) |
| 3x-chain | deeper beats own first pass | PASS (info) | R=3: 0.358 vs R=1: 0.177 (p=7.9e-25) |
| 3x-chain | **verdict** | **FAIL** | required criteria only |
