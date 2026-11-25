# Looped-head depth sweep

Base model: `HuggingFaceTB/SmolLM2-360M-Instruct`

## Accuracy: test (n=600)

| head | R | accuracy | 95% CI | vs R=0 | p | valid JSON | 1 hops | 2 hops | 3 hops |
|---|---|---|---|---|---|---|---|---|---|
| frozen baseline | 0 | 0.140 | 0.115-0.170 |  |  | 0.675 | 0.382 | 0.049 | 0.005 |
| 1x-plain | 1* | 0.375 | 0.337-0.414 | +0.235 | 4.5e-26 | 1.000 | 0.597 | 0.310 | 0.233 |
| 1x-plain | 2† | 0.355 | 0.318-0.394 | +0.215 | 1.1e-22 | 1.000 | 0.571 | 0.300 | 0.209 |
| 1x-plain | 3† | 0.353 | 0.316-0.392 | +0.213 | 3.4e-22 | 1.000 | 0.560 | 0.296 | 0.218 |
| 3x-chain | 1 | 0.188 | 0.159-0.222 | +0.048 | 0.006 | 1.000 | 0.592 | 0.000 | 0.000 |
| 3x-chain | 2 | 0.297 | 0.262-0.334 | +0.157 | 8.9e-15 | 1.000 | 0.576 | 0.330 | 0.005 |
| 3x-chain | 3* | 0.360 | 0.323-0.399 | +0.220 | 4.2e-23 | 1.000 | 0.560 | 0.315 | 0.218 |

`*` trained depth, `†` deeper than the head was trained. p: exact McNemar vs R=0.

## Acceptance

| head | criterion | result | detail |
|---|---|---|---|
| 1x-plain | beats frozen baseline | PASS | R=1: 0.375 vs R=0: 0.140 (gain +0.235, p=4.5e-26) |
| 1x-plain | monotonic up to trained depth | PASS | R0:0.140 -> R1:0.375 |
| 1x-plain | stable beyond trained depth | FAIL (info) | worst at R=3: 0.353 vs R=1: 0.375 |
| 1x-plain | **verdict** | **PASS** | required criteria only |
| 3x-chain | beats frozen baseline | PASS | R=3: 0.360 vs R=0: 0.140 (gain +0.220, p=4.2e-23) |
| 3x-chain | monotonic up to trained depth | PASS (info) | R0:0.140 -> R1:0.188 -> R2:0.297 -> R3:0.360 | head was trained at a single depth; shallower depths are off-distribution |
| 3x-chain | looping beats single-pass control | FAIL | R=3: 0.360 vs '1x-plain' at R=1: 0.375 (p=0.44) |
| 3x-chain | deeper beats own first pass | PASS (info) | R=3: 0.360 vs R=1: 0.188 (p=5.1e-21) |
| 3x-chain | **verdict** | **FAIL** | required criteria only |
